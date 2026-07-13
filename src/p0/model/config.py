"""Checkpoint-local policy architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture choices for constructing a new policy.

    Contract references and strict persistence are added when policy construction
    moves to ``PolicyFactory`` in checkpoint two. Application configuration does
    not own or override these values.
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
