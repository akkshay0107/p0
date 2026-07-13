"""Checkpoint-local policy architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture choices for constructing a new policy.

    This checkpoint intentionally contains architecture only. Compatibility
    fingerprints and resource-bundle identity remain deferred.
    """

    d_model: int
    nhead: int
    reducer_layers: int
    history_tokens: int
    dim_feedforward: int
    series_context_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "nhead",
            "reducer_layers",
            "history_tokens",
            "dim_feedforward",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("ModelConfig.d_model must be divisible by nhead")
        if self.series_context_enabled:
            raise ValueError("Series context remains disabled during the refactor baseline")

    @classmethod
    def baseline(cls) -> ModelConfig:
        return cls(
            d_model=512,
            nhead=8,
            reducer_layers=5,
            history_tokens=8,
            dim_feedforward=2048,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "reducer_layers": self.reducer_layers,
            "history_tokens": self.history_tokens,
            "dim_feedforward": self.dim_feedforward,
            "series_context_enabled": self.series_context_enabled,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        if not isinstance(value, dict):
            raise ValueError("ModelConfig must be an object")
        expected = {
            "d_model",
            "nhead",
            "reducer_layers",
            "history_tokens",
            "dim_feedforward",
            "series_context_enabled",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown or missing:
            raise ValueError(f"Invalid ModelConfig fields: missing={missing}, unknown={unknown}")
        return cls(**value)  # type: ignore[arg-type]

    @classmethod
    def from_legacy_policy(cls, policy: Any) -> ModelConfig:
        return cls(
            d_model=int(policy.d_model),
            nhead=int(policy.actor.reducer.encoder.layers[0].nhead),
            reducer_layers=len(policy.actor.reducer.encoder.layers),
            history_tokens=int(policy.actor.reducer.n_hg),
            dim_feedforward=int(policy.d_model) * 4,
        )
