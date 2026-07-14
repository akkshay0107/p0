"""Atomic checkpoint persistence behind the injected policy-store boundary."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    load_active_runtime_manifest,
    validate_artifact_runtime_contract,
)
from p0.model.config import ModelConfig
from p0.model.factory import PolicyFactory
from p0.model.policy import PolicyNet
from p0.model.resources import RuntimeResources

CHECKPOINT_SCHEMA = "p0.checkpoint.v1"
POLICY_ARTIFACT = "policy"
TRAINING_ARTIFACT = "training"


class PolicyStore(Protocol):
    def save_policy(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...

    def load_policy(self, path: Path, device: torch.device | str) -> PolicyNet: ...

    def save_training_state(
        self,
        path: Path,
        episode: int,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> None: ...

    def load_training_state(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> int | None: ...


class CheckpointStore:
    """The sole reader and writer for policy and training checkpoints."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
        *,
        resources: RuntimeResources | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self._resources = resources

    def save_policy(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        artifact = self._policy_artifact(policy, POLICY_ARTIFACT, metadata)
        _atomic_torch_save(path, artifact)

    def load_policy(self, path: Path, device: torch.device | str) -> PolicyNet:
        artifact = self._load_artifact(path)
        config = self._model_config(artifact, path)
        try:
            policy = PolicyFactory(self._runtime_resources()).create(config).to(device)
            policy.load_state_dict(artifact["model_state_dict"], strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid policy state in checkpoint {path}") from exc
        return policy

    def save_training_state(
        self,
        path: Path,
        episode: int,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> None:
        artifact = self._policy_artifact(policy, TRAINING_ARTIFACT)
        training_state: dict[str, Any] = {"episode": int(episode)}
        for name, service in (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
        ):
            if service is not None:
                training_state[f"{name}_state_dict"] = service.state_dict()
        artifact["training_state"] = training_state
        _atomic_torch_save(path, artifact)

    def load_training_state(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> int | None:
        if not path.exists():
            return None
        artifact = self._load_artifact(path)
        expected = self._policy_config(policy).to_dict()
        actual = self._model_config(artifact, path).to_dict()
        if actual != expected:
            raise ValueError(f"Checkpoint {path} model configuration does not match the policy")
        if artifact["artifact_type"] == POLICY_ARTIFACT:
            return None
        training_state = artifact.get("training_state")
        if not isinstance(training_state, Mapping):
            raise ValueError(f"Training checkpoint {path} has no valid training_state")
        try:
            policy.load_state_dict(artifact["model_state_dict"], strict=True)
            for name, service in (
                ("optimizer", optimizer),
                ("scheduler", scheduler),
                ("scaler", scaler),
            ):
                key = f"{name}_state_dict"
                if service is not None and key in training_state:
                    service.load_state_dict(training_state[key])
            episode = training_state["episode"]
            if type(episode) is not int or episode < 0:
                raise ValueError("episode must be a non-negative integer")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid training state in checkpoint {path}") from exc
        return episode

    def _policy_artifact(
        self,
        policy: PolicyNet,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = load_active_runtime_manifest(self.manifest_path)
        return {
            "artifact_schema": CHECKPOINT_SCHEMA,
            "artifact_type": artifact_type,
            "runtime_contract_sha256": manifest.runtime_contract_sha256,
            "model_config": self._policy_config(policy).to_dict(),
            "model_state_dict": policy.state_dict(),
            "provenance": dict(metadata or {}),
        }

    def _load_artifact(self, path: Path) -> Mapping[str, Any]:
        try:
            artifact = torch.load(path, weights_only=True, map_location="cpu")
        except (OSError, RuntimeError, EOFError) as exc:
            raise ValueError(f"Unable to read checkpoint {path}") from exc
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Malformed checkpoint {path}: expected a mapping")
        if "runtime_manifest_sha256" in artifact or artifact.get("artifact_schema") is None:
            raise ValueError(
                f"Unsupported legacy checkpoint format at {path}; "
                f"expected artifact_schema={CHECKPOINT_SCHEMA!r}"
            )
        if artifact.get("artifact_schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"Unsupported checkpoint schema {artifact.get('artifact_schema')!r} at {path}"
            )
        if artifact.get("artifact_type") not in {POLICY_ARTIFACT, TRAINING_ARTIFACT}:
            raise ValueError(f"Unsupported checkpoint artifact type at {path}")
        provenance = artifact.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Checkpoint {path} provenance must be a mapping")
        validate_artifact_runtime_contract(artifact, self.manifest_path)
        return artifact

    def _runtime_resources(self) -> RuntimeResources:
        if self._resources is None:
            self._resources = RuntimeResources.from_manifest(self.manifest_path)
        return self._resources

    @staticmethod
    def _model_config(artifact: Mapping[str, Any], path: Path) -> ModelConfig:
        try:
            return ModelConfig.from_dict(artifact.get("model_config"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model configuration in checkpoint {path}") from exc

    @staticmethod
    def _policy_config(policy: PolicyNet) -> ModelConfig:
        config = getattr(policy, "config", None)
        if not isinstance(config, ModelConfig):
            raise ValueError("Only policies with a validated ModelConfig can be checkpointed")
        return config


def _atomic_torch_save(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(artifact), temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


DEFAULT_POLICY_STORE = CheckpointStore()
