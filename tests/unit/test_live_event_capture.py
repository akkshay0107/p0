"""Tests for protocol-line capture owned by the live event capture module."""

from __future__ import annotations

import logging

from poke_env.battle import DoubleBattle

from p0.battle.events import SpatialActionType
from p0.runtime import poke_env_patches
from p0.runtime.live_event_capture import capture_message, captured_protocol_lines
from p0.runtime.poke_env_battle_adapter import battle_view


class TestLiveEventCapture:
    def test_battle_view_reads_current_turn_records(self) -> None:
        battle = DoubleBattle("turn-records", "TestPlayer", logging.getLogger("test"), gen=9)
        capture_message(
            battle,
            ("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"),
        )

        records = battle_view(battle).spatial_turn

        assert records[0].action_type == int(SpatialActionType.MOVE)
        assert records[0].order_rank == 0.25

    def test_protocol_line_capture_is_opt_in_and_survives_idempotent_install(self) -> None:
        poke_env_patches.uninstall_for_tests()
        without_capture = DoubleBattle(
            "without-capture", "TestPlayer", logging.getLogger("test"), gen=9
        )
        poke_env_patches.install()
        without_capture.parse_message(["", "gen", "9"])
        assert captured_protocol_lines(without_capture) == ()

        with_capture = DoubleBattle("with-capture", "TestPlayer", logging.getLogger("test"), gen=9)
        poke_env_patches.install(capture_protocol_lines=True)
        with_capture.parse_message(["", "gen", "9"])
        assert captured_protocol_lines(with_capture) == ("|gen|9",)
        poke_env_patches.uninstall_for_tests()

        after_uninstall = DoubleBattle(
            "after-uninstall", "TestPlayer", logging.getLogger("test"), gen=9
        )
        after_uninstall.parse_message(["", "gen", "9"])
        assert captured_protocol_lines(after_uninstall) == ()
