"""Injected policy persistence seam for the pre-envelope checkpoint format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from p0.format_config import (
    policy_model_config,
    runtime_manifest_sha256,
    validate_artifact_manifest_reference,
)
from p0.model.policy import PolicyNet


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


class LegacyManifestPolicyStore:
    """Checkpoint 3 adapter for the existing manifest-backed dictionaries."""

    @staticmethod
    def _policy_state(policy: PolicyNet) -> dict[str, Any]:
        return {
            "model_state_dict": policy.state_dict(),
            "runtime_manifest_sha256": runtime_manifest_sha256(),
            "model_config": policy_model_config(policy),
        }

    def save_policy(
        self,
        path: Path,
        policy: PolicyNet,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        state = self._policy_state(policy)
        if metadata:
            overlap = state.keys() & metadata.keys()
            if overlap:
                raise ValueError(f"Policy metadata replaces reserved fields: {sorted(overlap)}")
            state.update(metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_policy(self, path: Path, device: torch.device | str) -> PolicyNet:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        validate_artifact_manifest_reference(checkpoint)
        config_value = checkpoint.get("model_config")
        if not isinstance(config_value, dict) or not config_value:
            raise ValueError(f"Checkpoint {path} has no serialized model configuration")
        config = dict(config_value)
        try:
            config["obs_dim"] = tuple(config["obs_dim"])
            policy = PolicyNet(**config).to(device)
            policy.load_state_dict(checkpoint["model_state_dict"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model configuration in checkpoint {path}") from exc
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
        state = self._policy_state(policy)
        state["episode"] = episode
        for name, service in (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
        ):
            if service is not None:
                state[f"{name}_state_dict"] = service.state_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

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
        checkpoint = torch.load(path, map_location=policy.device)
        validate_artifact_manifest_reference(checkpoint)
        config = checkpoint.get("model_config")
        if not isinstance(config, dict) or config != policy_model_config(policy):
            raise ValueError(f"Checkpoint {path} model configuration does not match the policy")
        policy.load_state_dict(checkpoint["model_state_dict"])
        for name, service in (
            ("optimizer", optimizer),
            ("scheduler", scheduler),
            ("scaler", scaler),
        ):
            key = f"{name}_state_dict"
            if service is not None and key in checkpoint:
                service.load_state_dict(checkpoint[key])
        episode = checkpoint.get("episode")
        return int(episode) if episode is not None else None


LEGACY_POLICY_STORE = LegacyManifestPolicyStore()
