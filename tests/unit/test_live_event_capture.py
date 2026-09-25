"""Tests for protocol-line capture owned by the live event capture module."""

from __future__ import annotations

import logging

from poke_env.battle import DoubleBattle

from p0.battle.events import EventKind
from p0.runtime import poke_env_patches
from p0.runtime.live_event_capture import (
    capture_message,
    captured_protocol_lines,
    consume_events,
    is_retry,
)
from p0.runtime.poke_env_battle_adapter import battle_view


class TestLiveEventCapture:
    def test_events_survive_turn_markers_until_the_player_decides(self) -> None:
        battle = DoubleBattle("turn-records", "TestPlayer", logging.getLogger("test"), gen=9)
        capture_message(
            battle,
            ("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"),
        )
        capture_message(battle, ("", "turn", "2"))

        before_decision = battle_view(battle).spatial_events
        consume_events(battle)
        capture_message(battle, ("", ""))
        after_next_line = battle_view(battle).spatial_events

        assert [record.kind for record in before_decision] == [EventKind.MOVE]
        assert after_next_line == ()

    def test_decision_without_a_new_battle_line_is_a_retry(self) -> None:
        battle = DoubleBattle("retry-records", "TestPlayer", logging.getLogger("test"), gen=9)
        capture_message(battle, ("", "move", "p1a: Pikachu", "Protect", "p1a: Pikachu"))

        before_first_decision = is_retry(battle)
        consume_events(battle)
        after_rejected_choice = is_retry(battle)
        retried_events = battle_view(battle).spatial_events
        capture_message(battle, ("", ""))
        after_accepted_choice = is_retry(battle)

        assert before_first_decision is False
        assert after_rejected_choice is True
        assert [record.kind for record in retried_events] == [EventKind.MOVE]
        assert after_accepted_choice is False

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
