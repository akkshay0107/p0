from __future__ import annotations

import pytest

from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    parse_events,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_random_raw_events, stress_repetitions, stress_rng


@pytest.mark.stress
def test_event_parser_preserves_order_under_repeated_showdown_streams() -> None:
    EVENT_DIAGNOSTICS.clear()
    try:
        rng = stress_rng()
        iterations = stress_repetitions(default=1000)

        for _ in range(iterations):
            raw_events = list(stress_random_raw_events(rng))
            first = parse_events(raw_events, tokenizer)
            second = parse_events(raw_events, tokenizer)
            assert second == first
            assert [event.order for event in first] == list(range(len(first)))
            assert len(first) <= len(raw_events)
            assert first
        assert EVENT_DIAGNOSTICS["missing_pre_hp"] >= iterations
        assert EVENT_DIAGNOSTICS["oov_ids"] > 0
    finally:
        EVENT_DIAGNOSTICS.clear()
