"""Atomic checkpoint saving and loading for policy and training state."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer

from p0.contracts import RuntimeContract, compare_runtime_contracts
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
from p0.training.magnet import Magnet

POLICY_ARTIFACT = "policy"
TRAINING_ARTIFACT = "training"
VALUE_TARGET_SEMANTICS = "discounted_terminal_outcome.v1"
LOGGER = logging.getLogger(__name__)


def value_objective_metadata(gamma: float) -> dict[str, Any]:
    """Return the checkpoint metadata that pins the value-target definition."""
    return {"gamma": gamma, "value_target_semantics": VALUE_TARGET_SEMANTICS}


class LoadedCheckpoint(NamedTuple):
    path: Path
    artifact: Mapping[str, Any]

    def __str__(self) -> str:
        return str(self.path)


class CheckpointStore:
    """Reader and writer for policy and training checkpoints."""

    def __init__(
        self,
        *,
        resources: RuntimeResources | None = None,
    ) -> None:
        self._resources = default_runtime_resources() if resources is None else resources
        self._contract = RuntimeContract.from_resources(self._resources.vocab, self._resources.dex)

    def read(self, path: Path) -> LoadedCheckpoint:
        """Read and validate a checkpoint once so later loads do not reopen the file."""
        artifact = torch.load(path, weights_only=True, map_location="cpu")
        self.validate_artifact(artifact, path)
        return LoadedCheckpoint(path.resolve(), artifact)

    def save_policy(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write a weights-only policy checkpoint."""
        atomic_torch_save(path, self._build_artifact(policy, POLICY_ARTIFACT, metadata))

    def load_policy(
        self,
        path: Path | LoadedCheckpoint,
        device: torch.device | str,
        *,
        expected_objective: Mapping[str, Any] | None = None,
    ) -> PolicyNet:
        """Build and return a policy restored from a checkpoint."""
        artifact = self._load_artifact(path)

        for key, value in (expected_objective or {}).items():
            saved = artifact["provenance"].get(key)
            if saved != value:
                LOGGER.warning(
                    "Checkpoint %s was trained with %s=%r; this run uses %r",
                    path,
                    key,
                    saved,
                    value,
                )

        try:
            config = ModelConfig.from_dict(artifact["model_config"])
            policy = build_policy(config, self._resources).to(device)
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            return policy
        # Any failure here means the saved policy does not fit this code; name the file.
        except Exception as exc:
            raise ValueError(f"Invalid policy state in checkpoint {path}") from exc

    def save_training(
        self,
        path: Path,
        episode: int,
        policy: PolicyNet,
        *,
        trainer_kind: str,
        optimizer: Optimizer,
        scaler: GradScaler,
        magnet: Magnet | None = None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Atomically write the weights and state needed to continue training."""
        artifact = self._build_artifact(
            policy, TRAINING_ARTIFACT, {**metadata, "trainer_kind": trainer_kind}
        )

        state: dict[str, Any] = {
            "episode": episode,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        # Behavior cloning trains without a magnet.
        if magnet is not None:
            state["magnet_state_dict"] = magnet.state_dict()

        artifact["training_state"] = state
        atomic_torch_save(path, artifact)

    def load_training(
        self,
        path: Path | LoadedCheckpoint,
        policy: PolicyNet,
        *,
        trainer_kind: str,
        optimizer: Optimizer,
        scaler: GradScaler,
        magnet: Magnet | None = None,
    ) -> int:
        """Restore weights and training state. Returns the completed episode."""
        artifact = self._load_artifact(path)
        if artifact["artifact_type"] != TRAINING_ARTIFACT:
            raise ValueError(f"Checkpoint {path} is weights-only and cannot resume training")

        saved_kind = artifact["provenance"].get("trainer_kind")
        if saved_kind != trainer_kind:
            raise ValueError(
                f"Checkpoint {path} belongs to trainer {saved_kind!r}, not {trainer_kind!r}"
            )

        if artifact["model_config"] != policy.config.to_dict():
            raise ValueError(f"Checkpoint {path} model configuration does not match the policy")

        try:
            state = artifact["training_state"]
            load_canonical_policy_state_dict(policy, artifact["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])

            if magnet is not None:
                # A checkpoint without a magnet restarts it from the restored policy.
                if "magnet_state_dict" in state:
                    magnet.load_state_dict(state["magnet_state_dict"])
                else:
                    magnet.refresh(policy)

            return state["episode"]
        # Any failure here means the saved state does not fit this run; name the file.
        except Exception as exc:
            raise ValueError(f"Invalid training state in checkpoint {path}") from exc

    def load_episode(self, path: Path | LoadedCheckpoint) -> int:
        """Return the completed episode; a weights-only checkpoint reports zero."""
        artifact = self._load_artifact(path)
        if artifact["artifact_type"] == POLICY_ARTIFACT:
            return 0

        return artifact["training_state"]["episode"]

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
            "artifact_schema": CHECKPOINT_ARTIFACT_SCHEMA,
            "artifact_type": artifact_type,
            "runtime_contract": self._contract.to_dict(),
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
        if schema != CHECKPOINT_ARTIFACT_SCHEMA:
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

    def _validate_contract(self, artifact: Mapping[str, Any], path: Path) -> None:
        historical = RuntimeContract.from_dict(artifact["runtime_contract"])
        status = compare_runtime_contracts(historical, self._contract)
        if status == "incompatible":
            raise ValueError(f"Checkpoint {path} vocabulary or encoding is incompatible")
        if status == "warning":
            LOGGER.warning("Checkpoint %s was trained with different dex data", path)


DEFAULT_CHECKPOINT_STORE = CheckpointStore()
