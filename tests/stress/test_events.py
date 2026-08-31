"""Stress tests for spatial turn recorder."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.battle.events import (
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTurnRecorder,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions

_ACTORS = st.sampled_from(("p1a", "p1b", "p2a", "p2b"))
_TURN_LINES = st.tuples(
    _ACTORS,
    _ACTORS,
    st.booleans(),
    st.booleans(),
)


@pytest.mark.stress
@settings(max_examples=32, deadline=None)
@given(turns=st.lists(_TURN_LINES, min_size=1, max_size=stress_repetitions(default=1000)))
def test_spatial_turn_recorder_under_generated_streams(
    turns: list[tuple[str, str, bool, bool]],
) -> None:
    """Stress the spatial turn recorder across generated battle-turn streams."""
    recorder = SpatialTurnRecorder(player_role="p1")

    for actor, target, include_crit, include_fail in turns:
        recorder.reset_turn()
        recorder.apply_line(
            ["", "move", f"{actor}: Pokemon", "Thunderbolt", f"{target}: Opponent"], tokenizer
        )
        recorder.apply_line(["", "-damage", f"{target}: Opponent", "45/100"], tokenizer)
        if include_crit:
            recorder.apply_line(["", "-crit", f"{target}: Opponent"], tokenizer)
        if include_fail:
            recorder.apply_line(["", "-fail", f"{actor}: Pokemon"], tokenizer)

        records = recorder.to_records()
        assert len(records) == SPATIAL_SLOT_COUNT
        assert all(0 <= r.action_type < len(SpatialActionType) for r in records)
