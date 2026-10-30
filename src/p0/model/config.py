"""Checkpoint-local policy architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass

from p0.contracts import require_dataclass_fields
from p0.model.architecture_contract import EVENT_RAW_WIDTH


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture choices stored with each policy checkpoint."""

    d_model: int
    nhead: int
    reducer_layers: int
    dim_feedforward: int

    def __post_init__(self) -> None:
        for name, value in (
            ("d_model", self.d_model),
            ("nhead", self.nhead),
            ("reducer_layers", self.reducer_layers),
            ("dim_feedforward", self.dim_feedforward),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("ModelConfig.d_model must be divisible by nhead")
        if EVENT_RAW_WIDTH % self.nhead:
            raise ValueError("ModelConfig.nhead must divide the fixed low-width event channel")

    @classmethod
    def baseline(cls) -> ModelConfig:
        return cls(
            d_model=384,
            nhead=8,
            reducer_layers=8,
            dim_feedforward=1536,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "reducer_layers": self.reducer_layers,
            "dim_feedforward": self.dim_feedforward,
        }

    @classmethod
    def from_dict(cls, value: object) -> ModelConfig:
        if not isinstance(value, dict):
            raise ValueError("ModelConfig must be an object")
        require_dataclass_fields(value, cls)
        return cls(**value)  # type: ignore[arg-type]
