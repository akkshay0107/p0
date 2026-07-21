"""Checkpoint-local policy architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

_MODEL_CONFIG_FIELDS = frozenset(
    {
        "d_model",
        "nhead",
        "reducer_layers",
        "history_tokens",
        "dim_feedforward",
        "series_context_enabled",
        "series_tokens",
    }
)


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
    series_tokens: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("nhead", self.nhead),
            ("reducer_layers", self.reducer_layers),
            ("history_tokens", self.history_tokens),
            ("dim_feedforward", self.dim_feedforward),
            ("series_tokens", self.series_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive integer")
        if type(self.series_context_enabled) is not bool:
            raise ValueError("ModelConfig.series_context_enabled must be a boolean")
        if self.d_model % self.nhead:
            raise ValueError("ModelConfig.d_model must be divisible by nhead")

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
            "series_tokens": self.series_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        if type(value) is not dict:
            raise ValueError("ModelConfig must be an object")
        unknown = sorted(key for key in value if key not in _MODEL_CONFIG_FIELDS)
        missing = sorted(_MODEL_CONFIG_FIELDS - value.keys())
        if unknown or missing:
            raise ValueError(f"Invalid ModelConfig fields: missing={missing}, unknown={unknown}")
        return cls(
            d_model=cast(int, value["d_model"]),
            nhead=cast(int, value["nhead"]),
            reducer_layers=cast(int, value["reducer_layers"]),
            history_tokens=cast(int, value["history_tokens"]),
            dim_feedforward=cast(int, value["dim_feedforward"]),
            series_context_enabled=cast(bool, value["series_context_enabled"]),
            series_tokens=cast(int, value["series_tokens"]),
        )
