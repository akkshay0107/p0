"""Generated arrival-order checks for spatial event intervals."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.battle.events import (
    MAX_EVENT_RECORDS,
    EventKind,
    EventPosition,
    EventRecord,
    SpatialEventRecorder,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions

_ACTORS = st.sampled_from(("p1a", "p1b", "p2a", "p2b"))
_TURN_LINES = st.tuples(_ACTORS, _ACTORS, st.booleans(), st.booleans(), st.booleans())
_POSITIONS = {
    "p1a": EventPosition.OWN_LEFT,
    "p1b": EventPosition.OWN_RIGHT,
    "p2a": EventPosition.OPPONENT_LEFT,
    "p2b": EventPosition.OPPONENT_RIGHT,
}


class TestSpatialEventRecorder:
    @pytest.mark.stress
    @settings(max_examples=32, deadline=None)
    @given(turns=st.lists(_TURN_LINES, min_size=1, max_size=stress_repetitions(default=1000)))
    def test_generated_streams_stay_bounded_and_valid(
        self, turns: list[tuple[str, str, bool, bool, bool]]
    ) -> None:
        """Compare every retained record and the final tail to literal protocol expectations."""
        recorder = SpatialEventRecorder(player_role="p1")
        expected: list[EventRecord] = []
        records: tuple[EventRecord, ...] = ()
        for actor, target, include_crit, include_fail, decide in turns:
            source, destination = _POSITIONS[actor], _POSITIONS[target]
            recorder.apply_line(
                ["", "move", f"{actor}: Pokemon", "Thunderbolt", f"{target}: Opponent"],
                tokenizer,
                lambda _: 1.0,
            )
            expected.append(
                EventRecord(
                    EventKind.MOVE, source, destination, tokenizer.id_for("moves", "Thunderbolt")
                )
            )
            recorder.apply_line(
                ["", "-damage", f"{target}: Opponent", "45/100"], tokenizer, lambda _: 1.0
            )
            expected.append(
                EventRecord(EventKind.DAMAGE, source, destination, amount=-0.55, amount_known=1.0)
            )
            if include_crit:
                recorder.apply_line(["", "-crit", f"{target}: Opponent"], tokenizer, lambda _: None)
                expected.append(EventRecord(EventKind.CRIT, source, destination))
            if include_fail:
                recorder.apply_line(["", "-fail", f"{actor}: Pokemon"], tokenizer, lambda _: None)
                expected.append(EventRecord(EventKind.FAIL, source, source))
            recorder.apply_line(["", "turn", "2"], tokenizer, lambda _: None)
            if decide:
                records = recorder.consume()
                assert records == tuple(expected[:MAX_EVENT_RECORDS])
                assert recorder.pending() == records
                assert recorder.consume() == records
                expected.clear()

        tail = recorder.consume()
        if expected:
            assert tail == tuple(expected[:MAX_EVENT_RECORDS])
        else:
            # The final retry still sees the last consumed interval.
            assert tail == records
        assert tail
        assert len(tail) <= MAX_EVENT_RECORDS
