"""Tests for the effect names the engine reads from the generated data files."""

from __future__ import annotations

from p0.replays.reconstruction.contract import ALLOWED_EFFECTS_BY_TAG, COPYABLE_VOLATILES


class TestCopyableVolatiles:
    def test_baton_pass_copies_only_conditions_without_the_no_copy_flag(self) -> None:
        # Values read from the conditions in the pinned Showdown source.
        assert {"substitute", "dragoncheer", "aquaring", "ingrain", "leechseed", "magnetrise"} <= (
            COPYABLE_VOLATILES
        )
        assert COPYABLE_VOLATILES.isdisjoint({"stockpile", "trapped", "typechange", "typeadd"})


class TestAllowedEffects:
    def test_singlemove_accepts_only_the_moves_that_emit_it(self) -> None:
        assert ALLOWED_EFFECTS_BY_TAG["-singlemove"] == {"destinybond", "glaiverush"}
