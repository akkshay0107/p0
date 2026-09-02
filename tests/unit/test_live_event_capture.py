"""Tests for protocol-line capture owned by the live event capture module."""

from __future__ import annotations

import logging

from poke_env.battle import DoubleBattle

from p0.runtime import poke_env_patches
from p0.runtime.live_event_capture import captured_protocol_lines


class TestLiveEventCapture:
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
