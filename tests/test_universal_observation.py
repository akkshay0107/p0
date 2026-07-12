import numpy as np
import pytest
import torch

from src.model.event_builder import EventCollector, EventTypeId, RawBattleEvent
from src.model.observation_builder import _write_effects
from src.model.structured_observation import (
    CAT_EFFECT_START,
    CATEGORICAL_WIDTH,
    MAX_EFFECTS,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUMERICAL_WIDTH,
    CounterKind,
    EffectNamespace,
    StructuredObservation,
)


def test_typed_effects_are_sorted_aligned_and_report_overflow():
    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    num = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    entries = [
        (EffectNamespace.POKEMON, effect_id, CounterKind.ACTION_COUNT, effect_id, 0, False, 0)
        for effect_id in range(MAX_EFFECTS + 3, 0, -1)
    ]

    _write_effects(entries, cat, num)

    assert num[NUM_IDX_EFFECT_COUNT] == MAX_EFFECTS + 3
    assert num[NUM_IDX_EFFECT_OVERFLOW] == 3
    assert cat[CAT_EFFECT_START] == 1


def test_overflow_contract_rejects_unmarked_truncation():
    obs = StructuredObservation.empty_batch(1)
    obs.numerical[0, 1, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS + 1

    with pytest.raises(ValueError, match="overflow"):
        obs.validate_overflow_contract()


def test_event_parser_preserves_identity_target_and_evidence():
    events = EventCollector.parse_events(
        [
            RawBattleEvent(
                ("", "move", "p1a: Mew", "Metronome", "p2a: Gengar", "[from] ability: Dancer")
            ),
            RawBattleEvent(("", "-ability", "p2a: Gengar", "Cursed Body")),
            RawBattleEvent(("", "-fieldstart", "move: Trick Room")),
            RawBattleEvent(("", "-fieldend", "move: Trick Room")),
        ]
    )

    assert events[0].target_id == "p2a: Gengar"
    assert events[0].flags & 4
    assert events[1].event_type == EventTypeId.ABILITY
    assert events[1].ability_id > 0
    assert [event.event_type for event in events[2:]] == [
        EventTypeId.FIELD_START,
        EventTypeId.FIELD_END,
    ]
    assert events[2].effect_id > 0


def test_schema_v2_allocates_explicit_event_overflow_channel():
    obs = StructuredObservation.empty_batch(2)
    obs.events_num[0, 0, 2] = 7
    assert obs.overflow_totals() == (0, 7)
    assert torch.count_nonzero(obs.events_num[1]) == 0
