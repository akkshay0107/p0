from __future__ import annotations

import pytest

from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    parse_events,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions, stress_rng
from tests.stress.replay_fixtures import golden_raw_events


@pytest.mark.stress
def test_event_parser_preserves_order_under_repeated_showdown_streams() -> None:
    EVENT_DIAGNOSTICS.clear()
    fixture = tuple(golden_raw_events())
    rng = stress_rng()

    for _ in range(stress_repetitions(default=1000)):
        raw_events = [event for _ in range(rng.randint(1, 4)) for event in fixture]
        rng.shuffle(raw_events)
        expected = tuple(event.event_type for event in parse_events(raw_events, tokenizer))
        events = parse_events(raw_events, tokenizer)
        assert tuple(event.event_type for event in events) == expected
        assert [event.order for event in events] == list(range(len(expected)))
