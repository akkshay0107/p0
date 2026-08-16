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
    """Stress test the raw Showdown event parser across randomized protocol streams.

    Verifies that:
    1. Parsing is strictly deterministic given the same tokenized event sequence.
    2. Parsed events are assigned strictly contiguous 0-indexed sequential order indices.
    3. Non-event lines (such as chat) are safely filtered without breaking sequence indexing.
    4. Parser diagnostics accurately track telemetry for edge cases (missing pre-HP and OOV tokens).
    """
    EVENT_DIAGNOSTICS.clear()
    try:
        rng = stress_rng()
        iterations = stress_repetitions(default=1000)

        for _ in range(iterations):
            raw_events = list(stress_random_raw_events(rng))
            first = parse_events(raw_events, tokenizer)
            second = parse_events(raw_events, tokenizer)
            # Re-parsing the same stream must produce identical structured output
            assert second == first
            # Parsed events must receive strictly contiguous 0-indexed order values
            assert [event.order for event in first] == list(range(len(first)))
            # Length should be at most raw length (ignorable messages like chat are dropped)
            assert len(first) <= len(raw_events)
            assert first
        # Verify that error/fallback branches were actively exercised during the stress run
        assert EVENT_DIAGNOSTICS["missing_pre_hp"] >= iterations
        assert EVENT_DIAGNOSTICS["oov_ids"] > 0
    finally:
        EVENT_DIAGNOSTICS.clear()
