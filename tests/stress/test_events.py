"""Stress tests for spatial turn recorder."""

from __future__ import annotations

import pytest

from p0.battle.events import (
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTurnRecorder,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import stress_repetitions, stress_rng


@pytest.mark.stress
def test_spatial_turn_recorder_under_repeated_random_streams() -> None:
    """Stress test the spatial turn recorder across randomized battle turns."""
    rng = stress_rng()
    iterations = stress_repetitions(default=1000)
    recorder = SpatialTurnRecorder(player_role="p1")

    for _ in range(iterations):
        recorder.reset_turn()
        actor = rng.choice(("p1a", "p1b", "p2a", "p2b"))
        target = rng.choice(("p1a", "p1b", "p2a", "p2b"))
        recorder.apply_line(
            ["", "move", f"{actor}: Pokemon", "Thunderbolt", f"{target}: Opponent"], tokenizer
        )
        recorder.apply_line(["", "-damage", f"{target}: Opponent", "45/100"], tokenizer)
        if rng.random() > 0.5:
            recorder.apply_line(["", "-crit", f"{target}: Opponent"], tokenizer)
        if rng.random() > 0.8:
            recorder.apply_line(["", "-fail", f"{actor}: Pokemon"], tokenizer)

        records = recorder.to_records()
        assert len(records) == SPATIAL_SLOT_COUNT
        assert all(0 <= r.action_type < len(SpatialActionType) for r in records)
