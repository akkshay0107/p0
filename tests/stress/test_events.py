"""Stress tests for the ordered spatial event recorder."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.battle.events import (
    MAX_EVENT_RECORDS,
    NUM_EVENT_DETAILS,
    NUM_EVENT_KINDS,
    NUM_EVENT_POSITIONS,
    EventKind,
    SpatialEventRecorder,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions

_ACTORS = st.sampled_from(("p1a", "p1b", "p2a", "p2b"))
_TURN_LINES = st.tuples(
    _ACTORS,
    _ACTORS,
    st.booleans(),
    st.booleans(),
    st.booleans(),
)


class TestSpatialEventRecorder:
    @pytest.mark.stress
    @settings(max_examples=32, deadline=None)
    @given(turns=st.lists(_TURN_LINES, min_size=1, max_size=stress_repetitions(default=1000)))
    def test_generated_streams_stay_bounded_and_valid(
        self,
        turns: list[tuple[str, str, bool, bool, bool]],
    ) -> None:
        """Every consumed interval is bounded, ordered by arrival, and uses valid codes."""
        recorder = SpatialEventRecorder(player_role="p1")

        for actor, target, include_crit, include_fail, decide in turns:
            recorder.apply_line(
                ["", "move", f"{actor}: Pokemon", "Thunderbolt", f"{target}: Opponent"],
                tokenizer,
                lambda _identifier: 1.0,
            )
            recorder.apply_line(
                ["", "-damage", f"{target}: Opponent", "45/100"], tokenizer, lambda _identifier: 1.0
            )
            if include_crit:
                recorder.apply_line(["", "-crit", f"{target}: Opponent"], tokenizer, lambda _: None)
            if include_fail:
                recorder.apply_line(["", "-fail", f"{actor}: Pokemon"], tokenizer, lambda _: None)
            recorder.apply_line(["", "turn", "2"], tokenizer, lambda _: None)
            if not decide:
                continue

            records = recorder.consume()
            assert 0 < len(records) <= MAX_EVENT_RECORDS
            assert records[0].kind == EventKind.MOVE
            for record in records:
                assert 0 < record.kind < NUM_EVENT_KINDS
                assert 0 <= record.source < NUM_EVENT_POSITIONS
                assert 0 <= record.target < NUM_EVENT_POSITIONS
                assert 0 <= record.detail < NUM_EVENT_DETAILS
            # A retry before the next line observes the same interval.
            assert recorder.pending() == records
