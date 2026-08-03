from __future__ import annotations

import pytest

from p0.battle.events import EVENT_DIAGNOSTICS, EventTypeId, parse_events, truncate_events
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions
from tests.stress.replay_fixtures import GOLDEN_EVENT_TYPES, golden_raw_events


@pytest.mark.stress
def test_protocol_event_stream_matches_showdown_golden_types() -> None:
    EVENT_DIAGNOSTICS.clear()
    events = parse_events(list(golden_raw_events()), tokenizer)

    assert tuple(event.event_type for event in events) == GOLDEN_EVENT_TYPES
    assert events[0].entity_id == "p1a: Pikachu"
    damage = next(event for event in events if event.event_type is EventTypeId.DAMAGE)
    heal = next(event for event in events if event.event_type is EventTypeId.HEAL)
    unboost = next(event for event in events if event.event_type is EventTypeId.UNBOOST)
    assert damage.value == pytest.approx(-0.25)
    assert heal.value == pytest.approx(0.15)
    assert unboost.value == pytest.approx(-1 / 6)
    assert any(event.event_type is EventTypeId.MEGA for event in events)
    assert EVENT_DIAGNOSTICS["oov_ids"] >= 1


@pytest.mark.stress
def test_event_parser_preserves_order_under_repeated_showdown_streams() -> None:
    EVENT_DIAGNOSTICS.clear()
    raw_events = list(golden_raw_events())
    expected = tuple(event.event_type for event in parse_events(raw_events, tokenizer))

    for _ in range(stress_repetitions()):
        events = parse_events(raw_events, tokenizer)
        assert tuple(event.event_type for event in events) == expected
        assert [event.order for event in events] == list(range(len(expected)))


@pytest.mark.stress
def test_event_truncation_keeps_priority_events_and_protocol_order() -> None:
    EVENT_DIAGNOSTICS.clear()
    events = parse_events(list(golden_raw_events()), tokenizer)
    truncated = truncate_events(events, limit=12)

    assert len(truncated) == 12
    assert [event.order for event in truncated] == sorted(event.order for event in truncated)
    assert any(event.event_type is EventTypeId.MOVE for event in truncated)
    assert any(event.event_type is EventTypeId.SWITCH_IN for event in truncated)
    assert any(event.event_type is EventTypeId.FAINT for event in truncated)
