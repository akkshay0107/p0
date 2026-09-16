"""Atomic checkpoint saving and loading for policy and training state."""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
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
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.persistence import atomic_torch_save

CHECKPOINT_SCHEMA = CHECKPOINT_ARTIFACT_SCHEMA
POLICY_ARTIFACT = "policy"
TRAINING_ARTIFACT = "training"
LOGGER = logging.getLogger(__name__)


class LoadedCheckpoint(NamedTuple):
    path: Path
    artifact: Mapping[str, Any]
    sha256: str

    def __str__(self) -> str:
        return str(self.path)


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

    def read(self, path: Path) -> LoadedCheckpoint:
        """Read and hash the same open file, even if its path is replaced."""
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            artifact = torch.load(stream, weights_only=True, map_location="cpu")
        self.validate_artifact(artifact, path)
        return LoadedCheckpoint(path.resolve(), artifact, digest)

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

    def snapshot_policy(self, policy: PolicyNet, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Copy a selected policy to CPU so later updates cannot change it."""
        artifact = self._build_artifact(policy, POLICY_ARTIFACT, metadata)
        artifact["model_state_dict"] = {
            name: tensor.detach().to("cpu", copy=True)
            for name, tensor in artifact["model_state_dict"].items()
        }
        return artifact

    def load_policy(
        self,
        path: Path | LoadedCheckpoint,
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
        run_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write a full training checkpoint with optimizer and RNG states."""
        if type(episode) is not int or episode < 0:
            raise ValueError("Completed step must be a non-negative integer")
        details = dict(metadata or {})
        if trainer_kind is not None:
            details["trainer_kind"] = trainer_kind

        artifact = self._build_artifact(policy, TRAINING_ARTIFACT, details)
        state: dict[str, Any] = {"episode": int(episode), "rng_state": _capture_rng()}

        for name, svc in (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
            ("magnet", magnet),
        ):
            if svc is not None:
                state[f"{name}_state_dict"] = svc.state_dict()

        if run_state is not None:
            state["run"] = dict(run_state)
        artifact["training_state"] = state
        atomic_torch_save(path, artifact)

    def load_training(
        self,
        path: Path | LoadedCheckpoint,
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
        if isinstance(path, Path) and not path.exists():
            if require_training_state:
                raise FileNotFoundError(path)
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

        services = (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
            ("magnet", magnet),
        )
        if require_training_state:
            required = [f"{name}_state_dict" for name, service in services if service is not None]
            missing = [key for key in required if key not in training_state]
            rng = training_state.get("rng_state")
            if (
                not isinstance(rng, Mapping)
                or not isinstance(rng.get("python"), tuple)
                or not isinstance(rng.get("torch"), torch.Tensor)
            ):
                missing.append("rng_state")
            if missing:
                raise ValueError(f"Checkpoint {path} is missing required training state: {missing}")

        try:
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            for name, service in services:
                key = f"{name}_state_dict"
                if service is not None and key in training_state:
                    service.load_state_dict(training_state[key])
                elif name == "magnet" and service is not None:
                    service.refresh(policy)

            episode = training_state["episode"]
            if type(episode) is not int or episode < 0:
                raise ValueError("episode must be a non-negative integer")

            _restore_rng(training_state.get("rng_state"))
            return episode
        except Exception as exc:
            raise ValueError(f"Invalid training state in checkpoint {path}") from exc

    save_training_state = save_training
    load_training_state = load_training

    def load_episode(self, path: Path | LoadedCheckpoint) -> int:
        """Read saved progress without constructing a policy."""
        if isinstance(path, Path) and not path.exists():
            return 0
        artifact = self._load_artifact(path)
        if artifact.get("artifact_type") == POLICY_ARTIFACT:
            return 0
        state = artifact.get("training_state")
        if isinstance(state, Mapping) and isinstance(state.get("episode"), int):
            return max(0, state["episode"])
        return 0

    load_checkpoint_episode = load_episode

    def load_metadata(self, path: Path | LoadedCheckpoint) -> Mapping[str, Any]:
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

    def _load_artifact(self, path: Path | LoadedCheckpoint) -> Mapping[str, Any]:
        return path.artifact if isinstance(path, LoadedCheckpoint) else self.read(path).artifact

    def validate_artifact(self, artifact: Any, path: Path) -> None:
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
            raise ValueError(f"Checkpoint {path} metadata must be a mapping")

        self._validate_contract(artifact, path)
        try:
            ModelConfig.from_dict(artifact["model_config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model configuration in checkpoint {path}") from exc

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
            self._resources = default_runtime_resources()
        return self._resources

    @staticmethod
    def _match_metadata(
        artifact: Mapping[str, Any],
        path: Path | LoadedCheckpoint,
        expected: Mapping[str, Any] | None,
    ) -> None:
        for k, v in (expected or {}).items():
            if artifact["provenance"].get(k) != v:
                raise ValueError(f"Checkpoint {path} provenance field {k!r} is incompatible")


DEFAULT_CHECKPOINT_STORE = CheckpointStore()
DEFAULT_POLICY_STORE = DEFAULT_CHECKPOINT_STORE


def _capture_rng() -> dict[str, Any]:
    numpy_state: Any = np.random.get_state()
    state: dict[str, Any] = {
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(
            (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        )
    py_state = state.get("python")
    if py_state is not None:
        random.setstate(py_state)
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)
