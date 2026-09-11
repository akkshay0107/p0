"""Atomic checkpoint saving and loading for policy and training state."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    ContractCompatibility,
    checkpoint_contract_compatibility,
    load_active_runtime_manifest,
)
from p0.model.architecture_contract import CHECKPOINT_ARTIFACT_SCHEMA
from p0.model.config import ModelConfig
from p0.model.factory import (
    build_policy,
    canonical_policy_state_dict,
    load_canonical_policy_state_dict,
)
from p0.model.policy import PolicyNet
from p0.model.resources import RuntimeResources
from p0.persistence import atomic_torch_save

CHECKPOINT_SCHEMA = CHECKPOINT_ARTIFACT_SCHEMA
POLICY_ARTIFACT = "policy"
TRAINING_ARTIFACT = "training"
LOGGER = logging.getLogger(__name__)


class CheckpointStore:
    """Reader and writer for policy and training checkpoints."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
        *,
        resources: RuntimeResources | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.resolve() != DEFAULT_RUNTIME_MANIFEST.resolve():
            raise ValueError("CheckpointStore always uses the default global runtime manifest")
        self._manifest = load_active_runtime_manifest(self.manifest_path)
        self._resources = resources
        self._reused_artifact: tuple[Path, Mapping[str, Any]] | None = None

    @contextmanager
    def reuse_artifact(self, path: Path) -> Iterator[None]:
        """Cache a loaded checkpoint during initialization to avoid re-reading the file."""
        artifact = self._load_artifact(path)
        previous = self._reused_artifact
        self._reused_artifact = (path.resolve(), artifact)
        try:
            yield
        finally:
            self._reused_artifact = previous

    def preflight(self, path: Path) -> ContractCompatibility:
        """Reject incompatible checkpoint contracts before setup begins."""
        artifact = self._read_raw(path)
        return self._validate_contract(artifact, path)

    def save_policy(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write a weights-only policy checkpoint."""
        artifact = self._build_artifact(policy, POLICY_ARTIFACT, metadata)
        atomic_torch_save(path, artifact)

    def load_policy(
        self,
        path: Path,
        device: torch.device | str,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> PolicyNet:
        """Build and return a policy restored from a checkpoint."""
        artifact = self._load_artifact(path)
        self._match_metadata(artifact, path, expected_metadata)
        try:
            config = ModelConfig.from_dict(artifact["model_config"])
            policy = build_policy(config, self._runtime_resources()).to(device)
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            return policy
        except Exception as exc:
            raise ValueError(f"Invalid policy state in checkpoint {path}") from exc

    def save_training(
        self,
        path: Path,
        episode: int,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        magnet: Any = None,
        metadata: Mapping[str, Any] | None = None,
        trainer_kind: str | None = None,
    ) -> None:
        """Atomically write a full training checkpoint with optimizer and RNG states."""
        lineage = dict(metadata or {})
        if trainer_kind is not None:
            lineage["trainer_kind"] = trainer_kind

        artifact = self._build_artifact(policy, TRAINING_ARTIFACT, lineage)
        state: dict[str, Any] = {"episode": int(episode), "rng_state": _capture_rng()}

        for name, svc in (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
            ("magnet", magnet),
        ):
            if svc is not None:
                state[f"{name}_state_dict"] = svc.state_dict()

        artifact["training_state"] = state
        atomic_torch_save(path, artifact)

    def load_training(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        magnet: Any = None,
        expected_trainer_kind: str | None = None,
        expected_metadata: Mapping[str, Any] | None = None,
        require_training_state: bool = False,
    ) -> int:
        """Restore policy weights and training services. Returns completed episode."""
        if not path.exists():
            return 0

        artifact = self._load_artifact(path)
        if artifact["model_config"] != policy.config.to_dict():
            raise ValueError(f"Checkpoint {path} model configuration does not match the policy")

        if artifact["artifact_type"] == POLICY_ARTIFACT:
            if require_training_state:
                raise ValueError(f"Checkpoint {path} is weights-only and cannot resume training")
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            return 0

        if expected_trainer_kind is not None:
            trainer = artifact["provenance"].get("trainer_kind")
            if trainer != expected_trainer_kind:
                raise ValueError(
                    f"Checkpoint {path} belongs to trainer {trainer!r}, "
                    f"not {expected_trainer_kind!r}"
                )

        self._match_metadata(artifact, path, expected_metadata)

        training_state = artifact.get("training_state")
        if not isinstance(training_state, Mapping):
            raise ValueError(f"Training checkpoint {path} has no valid training_state")

        try:
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            for name, svc in (
                ("optimizer", optimizer),
                ("scheduler", scheduler),
                ("scaler", scaler),
            ):
                key = f"{name}_state_dict"
                if svc is not None and key in training_state:
                    svc.load_state_dict(training_state[key])

            if magnet is not None:
                if "magnet_state_dict" in training_state:
                    magnet.load_state_dict(training_state["magnet_state_dict"])
                else:
                    magnet.refresh(policy)

            episode = training_state["episode"]
            if type(episode) is not int or episode < 0:
                raise ValueError("episode must be a non-negative integer")

            _restore_rng(training_state.get("rng_state"))
            return episode
        except Exception as exc:
            raise ValueError(f"Invalid training state in checkpoint {path}") from exc

    save_training_state = save_training
    load_training_state = load_training

    def load_episode(self, path: Path) -> int:
        """Read completed episode count from a checkpoint header without full restoration."""
        if not path.exists():
            return 0
        artifact = self._load_artifact(path)
        if artifact.get("artifact_type") == POLICY_ARTIFACT:
            return 0
        state = artifact.get("training_state")
        if isinstance(state, Mapping) and isinstance(state.get("episode"), int):
            return max(0, state["episode"])
        return 0

    load_checkpoint_episode = load_episode

    def load_metadata(self, path: Path) -> Mapping[str, Any]:
        """Return validated checkpoint metadata without constructing a policy."""
        return self._load_artifact(path)["provenance"]

    def _build_artifact(
        self,
        policy: PolicyNet,
        artifact_type: str,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "artifact_schema": CHECKPOINT_SCHEMA,
            "artifact_type": artifact_type,
            "global_contract_sha256": self._manifest.global_sha256,
            "global_contract": self._manifest.to_dict(),
            "model_config": policy.config.to_dict(),
            "model_state_dict": canonical_policy_state_dict(policy),
            "provenance": dict(metadata or {}),
        }

    def _load_artifact(self, path: Path) -> Mapping[str, Any]:
        reused = self._reused_artifact
        if reused is not None and reused[0] == path.resolve():
            return reused[1]
        artifact = self._read_raw(path)
        self._validate_contract(artifact, path)
        try:
            ModelConfig.from_dict(artifact["model_config"])
        except Exception as exc:
            raise ValueError(f"Invalid model configuration in checkpoint {path}") from exc
        return artifact

    def _read_raw(self, path: Path) -> Mapping[str, Any]:
        try:
            artifact = torch.load(path, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise ValueError(f"Unable to read checkpoint {path}") from exc

        if not isinstance(artifact, Mapping):
            raise ValueError(f"Malformed checkpoint {path}: expected a mapping")

        schema = artifact.get("artifact_schema")
        if (
            "runtime_manifest_sha256" in artifact
            or "runtime_contract_sha256" in artifact
            or schema != CHECKPOINT_SCHEMA
        ):
            raise ValueError(f"Unsupported checkpoint schema {schema!r} at {path}")
        if artifact.get("artifact_type") not in {POLICY_ARTIFACT, TRAINING_ARTIFACT}:
            raise ValueError(f"Unsupported checkpoint artifact type at {path}")
        if not isinstance(artifact.get("provenance"), Mapping):
            raise ValueError(f"Checkpoint {path} provenance must be a mapping")

        return artifact

    def _validate_contract(self, artifact: Mapping[str, Any], path: Path) -> ContractCompatibility:
        comp = checkpoint_contract_compatibility(artifact, self.manifest_path)
        if not comp.is_compatible:
            diffs = "; ".join(comp.major_differences)
            raise ValueError(
                f"Checkpoint {path} is incompatible with the active global contract: {diffs}"
            )
        if comp.status == "warning":
            LOGGER.warning(
                "Checkpoint %s has non-breaking contract differences: %s",
                path,
                "; ".join(comp.minor_differences),
            )
        return comp

    def _runtime_resources(self) -> RuntimeResources:
        if self._resources is None:
            self._resources = RuntimeResources.from_manifest(self.manifest_path)
        return self._resources

    @staticmethod
    def _match_metadata(
        artifact: Mapping[str, Any], path: Path, expected: Mapping[str, Any] | None
    ) -> None:
        for k, v in (expected or {}).items():
            if artifact["provenance"].get(k) != v:
                raise ValueError(f"Checkpoint {path} provenance field {k!r} is incompatible")


DEFAULT_CHECKPOINT_STORE = CheckpointStore()
DEFAULT_POLICY_STORE = DEFAULT_CHECKPOINT_STORE


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    py_state = state.get("python")
    if py_state is not None:
        random.setstate(py_state)
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)
