from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

TEAM_SIZE = 6
MOVE_SLOTS = 4
MAX_VOLATILES = 6
SEQUENCE_LENGTH = 1 + TEAM_SIZE * 2 * 2 + 3
CATEGORICAL_WIDTH = 20
NUMERICAL_WIDTH = 36


class TokenType(IntEnum):
    CLS = 0
    POKEMON_SUPER = 1
    POKEMON_NUMERIC = 2
    GLOBAL_FIELD = 3
    ALLY_SIDE = 4
    OPPONENT_SIDE = 5


class SideId(IntEnum):
    NONE = 0
    ALLY = 1
    OPPONENT = 2


@dataclass(slots=True)
class StructuredObservation:
    """Fixed battle token structure consumed by the learned encoder."""

    token_type_ids: torch.Tensor
    side_ids: torch.Tensor
    slot_ids: torch.Tensor
    categorical: torch.Tensor
    numerical: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "token_type_ids": self.token_type_ids,
            "side_ids": self.side_ids,
            "slot_ids": self.slot_ids,
            "categorical": self.categorical,
            "numerical": self.numerical,
        }

    def to(self, *args, **kwargs) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.to(*args, **kwargs),
            side_ids=self.side_ids.to(*args, **kwargs),
            slot_ids=self.slot_ids.to(*args, **kwargs),
            categorical=self.categorical.to(*args, **kwargs),
            numerical=self.numerical.to(*args, **kwargs),
        )

    def unsqueeze(self, dim: int) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.unsqueeze(dim),
            side_ids=self.side_ids.unsqueeze(dim),
            slot_ids=self.slot_ids.unsqueeze(dim),
            categorical=self.categorical.unsqueeze(dim),
            numerical=self.numerical.unsqueeze(dim),
        )

    def cpu(self) -> StructuredObservation:
        return self.to("cpu")

    def __getitem__(self, index) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids[index],
            side_ids=self.side_ids[index],
            slot_ids=self.slot_ids[index],
            categorical=self.categorical[index],
            numerical=self.numerical[index],
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.cat:
            observations = args[0]
            dim = kwargs.get("dim", 0)
            return StructuredObservation(
                token_type_ids=torch.cat([obs.token_type_ids for obs in observations], dim=dim),
                side_ids=torch.cat([obs.side_ids for obs in observations], dim=dim),
                slot_ids=torch.cat([obs.slot_ids for obs in observations], dim=dim),
                categorical=torch.cat([obs.categorical for obs in observations], dim=dim),
                numerical=torch.cat([obs.numerical for obs in observations], dim=dim),
            )
        return NotImplemented
