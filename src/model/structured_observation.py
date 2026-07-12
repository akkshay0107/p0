from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

TEAM_SIZE = 6
MOVE_SLOTS = 4
MAX_EFFECTS = 16
SEQUENCE_LENGTH = 1 + TEAM_SIZE * 2 * 2 + 3 * 2
POKEMON_IDENTITY_WIDTH = 25
CAT_KNOWNNESS_START = 25
CAT_KNOWNNESS_WIDTH = POKEMON_IDENTITY_WIDTH
CAT_EFFECT_START = 52
EFFECT_CATEGORICAL_WIDTH = 3
CATEGORICAL_WIDTH = CAT_EFFECT_START + MAX_EFFECTS * EFFECT_CATEGORICAL_WIDTH

NUM_BASE_WIDTH = 56
NUM_PROVENANCE_START = NUM_BASE_WIDTH
NUM_PROVENANCE_WIDTH = 8
NUM_EFFECT_START = NUM_PROVENANCE_START + NUM_PROVENANCE_WIDTH
EFFECT_NUMERICAL_WIDTH = 5
NUM_IDX_EFFECT_COUNT = NUM_EFFECT_START + MAX_EFFECTS * EFFECT_NUMERICAL_WIDTH
NUM_IDX_EFFECT_OVERFLOW = NUM_IDX_EFFECT_COUNT + 1
NUMERICAL_WIDTH = NUM_IDX_EFFECT_OVERFLOW + 1

EVENT_COUNT = 64
EVENT_CATEGORICAL_WIDTH = 10
EVENT_NUMERICAL_WIDTH = 3
EVENT_ORDER_VOCAB_SIZE = EVENT_COUNT + 1

TOKEN_IDX_CLS = 0
TOKEN_IDX_GLOBAL_FIELD_SUPER = 25
TOKEN_IDX_GLOBAL_FIELD_NUMERIC = 26
TOKEN_IDX_ALLY_SIDE_SUPER = 27
TOKEN_IDX_ALLY_SIDE_NUMERIC = 28
TOKEN_IDX_OPPONENT_SIDE_SUPER = 29
TOKEN_IDX_OPPONENT_SIDE_NUMERIC = 30

NUM_IDX_TEAM_PREVIEW = 2
NUM_IDX_MOVE_PP = 19  # 19-22: per-move-slot pp fraction
NUM_IDX_ORIG_IDX_RATIO = 26
NUM_IDX_FAINTED = 27
NUM_IDX_MOVE_LAST = 32  # 32-35: per-move-slot "was the last move used" (actives only)
NUM_IDX_MOVE_LEGAL = 50  # 50-53: per-move-slot "legal this step" (allies only)
NUM_IDX_CAN_SWITCH_OUT = 54  # active allies only
NUM_IDX_REVEALED = 55  # has appeared on the field this battle


ALLY_POKE_TOKENS = (1, 3, 5, 7, 9, 11)
ALLY_NUM_TOKENS = (2, 4, 6, 8, 10, 12)
ALL_NUM_TOKENS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
TARGET_SEQ_INDICES = (3, 1, TOKEN_IDX_CLS, 13, 15)


class TokenType(IntEnum):
    CLS = 0
    POKEMON_SUPER = 1
    POKEMON_NUMERIC = 2
    FIELD_SUPER = 3
    FIELD_NUMERIC = 4
    EVENT = 5


class SideId(IntEnum):
    NONE = 0
    ALLY = 1
    OPPONENT = 2


class Knownness(IntEnum):
    PAD = 0
    UNKNOWN = 1
    KNOWN_NONE = 2
    KNOWN = 3
    OOV = 4


class Provenance(IntEnum):
    PAD = 0
    UNKNOWN = 1
    OBSERVED = 2
    OPEN_TEAM_SHEET = 3
    SELF_KNOWN = 4
    IMPUTED = 5


class EffectNamespace(IntEnum):
    NONE = 0
    POKEMON = 1
    SIDE = 2
    FIELD = 3
    WEATHER = 4


class CounterKind(IntEnum):
    PRESENCE_ONLY = 0
    TURN_AGE = 1
    ACTION_COUNT = 2
    STACK_COUNT = 3
    KNOWN_REMAINING = 4


def effect_cat_slice(index: int) -> slice:
    if not 0 <= index < MAX_EFFECTS:
        raise IndexError(index)
    start = CAT_EFFECT_START + index * EFFECT_CATEGORICAL_WIDTH
    return slice(start, start + EFFECT_CATEGORICAL_WIDTH)


def effect_num_slice(index: int) -> slice:
    if not 0 <= index < MAX_EFFECTS:
        raise IndexError(index)
    start = NUM_EFFECT_START + index * EFFECT_NUMERICAL_WIDTH
    return slice(start, start + EFFECT_NUMERICAL_WIDTH)


@dataclass(slots=True)
class StructuredObservation:
    """Fixed battle token structure consumed by the learned encoder."""

    token_type_ids: torch.Tensor
    side_ids: torch.Tensor
    slot_ids: torch.Tensor
    categorical: torch.Tensor
    numerical: torch.Tensor
    events_cat: torch.Tensor
    events_num: torch.Tensor
    events_side_ids: torch.Tensor
    events_slot_ids: torch.Tensor

    def is_teampreview(self) -> torch.Tensor:
        return is_teampreview(self.numerical)

    def overflow_totals(self) -> tuple[int, int]:
        """Return effect and event overflow counts for telemetry and corpus audits."""
        effect_overflow = int(self.numerical[..., NUM_IDX_EFFECT_OVERFLOW].sum().item())
        event_overflow = int(self.events_num[..., 2].amax().item())
        return effect_overflow, event_overflow

    def validate_overflow_contract(self) -> None:
        """Reject counts that imply silent effect truncation."""
        counts = self.numerical[..., NUM_IDX_EFFECT_COUNT]
        overflow = self.numerical[..., NUM_IDX_EFFECT_OVERFLOW]
        expected = torch.clamp(counts - MAX_EFFECTS, min=0)
        if not torch.equal(overflow, expected):
            raise ValueError("Effect overflow does not match the number of dropped effects")

    def clone(self) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.clone(),
            side_ids=self.side_ids.clone(),
            slot_ids=self.slot_ids.clone(),
            categorical=self.categorical.clone(),
            numerical=self.numerical.clone(),
            events_cat=self.events_cat.clone(),
            events_num=self.events_num.clone(),
            events_side_ids=self.events_side_ids.clone(),
            events_slot_ids=self.events_slot_ids.clone(),
        )

    def to(self, *args, **kwargs) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.to(*args, **kwargs),
            side_ids=self.side_ids.to(*args, **kwargs),
            slot_ids=self.slot_ids.to(*args, **kwargs),
            categorical=self.categorical.to(*args, **kwargs),
            numerical=self.numerical.to(*args, **kwargs),
            events_cat=self.events_cat.to(*args, **kwargs),
            events_num=self.events_num.to(*args, **kwargs),
            events_side_ids=self.events_side_ids.to(*args, **kwargs),
            events_slot_ids=self.events_slot_ids.to(*args, **kwargs),
        )

    def unsqueeze(self, dim: int) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=self.token_type_ids.unsqueeze(dim),
            side_ids=self.side_ids.unsqueeze(dim),
            slot_ids=self.slot_ids.unsqueeze(dim),
            categorical=self.categorical.unsqueeze(dim),
            numerical=self.numerical.unsqueeze(dim),
            events_cat=self.events_cat.unsqueeze(dim),
            events_num=self.events_num.unsqueeze(dim),
            events_side_ids=self.events_side_ids.unsqueeze(dim),
            events_slot_ids=self.events_slot_ids.unsqueeze(dim),
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
            events_cat=self.events_cat[index],
            events_num=self.events_num[index],
            events_side_ids=self.events_side_ids[index],
            events_slot_ids=self.events_slot_ids[index],
        )

    @staticmethod
    def cat(observations: list[StructuredObservation], dim: int = 0) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=torch.cat([obs.token_type_ids for obs in observations], dim=dim),
            side_ids=torch.cat([obs.side_ids for obs in observations], dim=dim),
            slot_ids=torch.cat([obs.slot_ids for obs in observations], dim=dim),
            categorical=torch.cat([obs.categorical for obs in observations], dim=dim),
            numerical=torch.cat([obs.numerical for obs in observations], dim=dim),
            events_cat=torch.cat([obs.events_cat for obs in observations], dim=dim),
            events_num=torch.cat([obs.events_num for obs in observations], dim=dim),
            events_side_ids=torch.cat([obs.events_side_ids for obs in observations], dim=dim),
            events_slot_ids=torch.cat([obs.events_slot_ids for obs in observations], dim=dim),
        )

    @staticmethod
    def stack(observations: list[StructuredObservation], dim: int = 0) -> StructuredObservation:
        return StructuredObservation(
            token_type_ids=torch.stack([obs.token_type_ids for obs in observations], dim=dim),
            side_ids=torch.stack([obs.side_ids for obs in observations], dim=dim),
            slot_ids=torch.stack([obs.slot_ids for obs in observations], dim=dim),
            categorical=torch.stack([obs.categorical for obs in observations], dim=dim),
            numerical=torch.stack([obs.numerical for obs in observations], dim=dim),
            events_cat=torch.stack([obs.events_cat for obs in observations], dim=dim),
            events_num=torch.stack([obs.events_num for obs in observations], dim=dim),
            events_side_ids=torch.stack([obs.events_side_ids for obs in observations], dim=dim),
            events_slot_ids=torch.stack([obs.events_slot_ids for obs in observations], dim=dim),
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
            events_cat=torch.zeros(
                (batch_size, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH),
                dtype=torch.long,
                pin_memory=pin_memory,
            ),
            events_num=torch.zeros(
                (batch_size, EVENT_COUNT, EVENT_NUMERICAL_WIDTH),
                dtype=torch.float32,
                pin_memory=pin_memory,
            ),
            events_side_ids=torch.zeros(
                (batch_size, EVENT_COUNT),
                dtype=torch.long,
                pin_memory=pin_memory,
            ),
            events_slot_ids=torch.zeros(
                (batch_size, EVENT_COUNT),
                dtype=torch.long,
                pin_memory=pin_memory,
            ),
        )


def is_teampreview(numerical: torch.Tensor) -> torch.Tensor:
    return numerical[:, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, NUM_IDX_TEAM_PREVIEW] > 0.5
