from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

TEAM_SIZE = 6
MOVE_SLOTS = 4
MAX_VOLATILES = 6
SEQUENCE_LENGTH = 1 + TEAM_SIZE * 2 * 2 + 3
CATEGORICAL_WIDTH = 25
NUMERICAL_WIDTH = 50

TOKEN_IDX_CLS = 0
TOKEN_IDX_GLOBAL_FIELD = 25
TOKEN_IDX_ALLY_SIDE = 26
TOKEN_IDX_OPPONENT_SIDE = 27

NUM_IDX_TEAM_PREVIEW = 2
NUM_IDX_ORIG_IDX_RATIO = 26
NUM_IDX_FAINTED = 27


ALLY_POKE_TOKENS = (1, 3, 5, 7, 9, 11)
ALLY_NUM_TOKENS = (2, 4, 6, 8, 10, 12)
ALL_NUM_TOKENS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
TARGET_SEQ_INDICES = (3, 1, TOKEN_IDX_CLS, 13, 15)


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

    def clone(self) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.clone(),
            side_ids=self.side_ids.clone(),
            slot_ids=self.slot_ids.clone(),
            categorical=self.categorical.clone(),
            numerical=self.numerical.clone(),
        )

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

    @staticmethod
    def cat(observations: list[StructuredObservation], dim: int = 0) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=torch.cat([obs.token_type_ids for obs in observations], dim=dim),
            side_ids=torch.cat([obs.side_ids for obs in observations], dim=dim),
            slot_ids=torch.cat([obs.slot_ids for obs in observations], dim=dim),
            categorical=torch.cat([obs.categorical for obs in observations], dim=dim),
            numerical=torch.cat([obs.numerical for obs in observations], dim=dim),
        )

    @staticmethod
    def stack(observations: list[StructuredObservation], dim: int = 0) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=torch.stack([obs.token_type_ids for obs in observations], dim=dim),
            side_ids=torch.stack([obs.side_ids for obs in observations], dim=dim),
            slot_ids=torch.stack([obs.slot_ids for obs in observations], dim=dim),
            categorical=torch.stack([obs.categorical for obs in observations], dim=dim),
            numerical=torch.stack([obs.numerical for obs in observations], dim=dim),
        )

    @staticmethod
    def empty_batch(batch_size: int, pin_memory: bool = False) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=torch.zeros(
                (batch_size, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=pin_memory
            ),
            side_ids=torch.zeros(
                (batch_size, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=pin_memory
            ),
            slot_ids=torch.zeros(
                (batch_size, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=pin_memory
            ),
            categorical=torch.zeros(
                (batch_size, SEQUENCE_LENGTH, CATEGORICAL_WIDTH),
                dtype=torch.long,
                pin_memory=pin_memory,
            ),
            numerical=torch.zeros(
                (batch_size, SEQUENCE_LENGTH, NUMERICAL_WIDTH),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
        )
