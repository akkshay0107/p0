"""Checkpoint-local policy architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

_MODEL_CONFIG_FIELDS = frozenset(
    {
        "d_model",
        "nhead",
        "prelude_layers",
        "history_tokens",
        "dim_feedforward",
        "coda_layers",
        "core_repeats",
        "core_weights_tied",
        "pass_embedding_enabled",
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
    prelude_layers: int
    history_tokens: int
    dim_feedforward: int
    coda_layers: int = 1
    core_repeats: int = 1
    core_weights_tied: bool = False
    pass_embedding_enabled: bool = True
    series_context_enabled: bool = False
    series_tokens: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("nhead", self.nhead),
            ("prelude_layers", self.prelude_layers),
            ("history_tokens", self.history_tokens),
            ("dim_feedforward", self.dim_feedforward),
            ("coda_layers", self.coda_layers),
            ("core_repeats", self.core_repeats),
            ("series_tokens", self.series_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive integer")
        for name, value in (
            ("core_weights_tied", self.core_weights_tied),
            ("pass_embedding_enabled", self.pass_embedding_enabled),
            ("series_context_enabled", self.series_context_enabled),
        ):
            if type(value) is not bool:
                raise ValueError(f"ModelConfig.{name} must be a boolean")
        if self.prelude_layers != 1 or self.coda_layers != 1:
            raise ValueError("ModelConfig prelude_layers and coda_layers must both be one")
        if self.core_weights_tied and self.core_repeats == 1:
            raise ValueError("ModelConfig.core_weights_tied requires core_repeats greater than one")
        if self.d_model % self.nhead:
            raise ValueError("ModelConfig.d_model must be divisible by nhead")

    @classmethod
    def baseline(cls) -> ModelConfig:
        return cls(
            d_model=512,
            nhead=8,
            prelude_layers=1,
            history_tokens=8,
            dim_feedforward=2048,
            coda_layers=1,
            core_repeats=1,
            core_weights_tied=False,
            pass_embedding_enabled=True,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "prelude_layers": self.prelude_layers,
            "history_tokens": self.history_tokens,
            "dim_feedforward": self.dim_feedforward,
            "coda_layers": self.coda_layers,
            "core_repeats": self.core_repeats,
            "core_weights_tied": self.core_weights_tied,
            "pass_embedding_enabled": self.pass_embedding_enabled,
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
            prelude_layers=cast(int, value["prelude_layers"]),
            history_tokens=cast(int, value["history_tokens"]),
            dim_feedforward=cast(int, value["dim_feedforward"]),
            coda_layers=cast(int, value["coda_layers"]),
            core_repeats=cast(int, value["core_repeats"]),
            core_weights_tied=cast(bool, value["core_weights_tied"]),
            pass_embedding_enabled=cast(bool, value["pass_embedding_enabled"]),
            series_context_enabled=cast(bool, value["series_context_enabled"]),
            series_tokens=cast(int, value["series_tokens"]),
        )
