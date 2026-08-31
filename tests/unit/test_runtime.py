from __future__ import annotations

import logging
import socket
import subprocess
import sys

import pytest
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder import TeambuilderPokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from p0.battle.views import TransformedPokemonView
from p0.runtime import poke_env_patches, showdown
from p0.runtime.live_event_capture import captured_protocol_lines
from p0.runtime.poke_env_battle_adapter import battle_view


def test_event_parser_import_does_not_install_poke_env_patches() -> None:
    """Verify that importing event parsing modules does not implicitly mutate poke-env global state."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, p0.battle.events as events, p0.runtime.poke_env_patches as patches; "
                "sys.exit(0 if not patches.is_installed() else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_patch_installation_is_idempotent_reversible_and_logger_scoped() -> None:
    """Verify poke-env monkey patches can be installed repeatedly, uninstalled cleanly, and filter logs."""
    poke_env_patches.uninstall_for_tests()
    original = DoubleBattle.parse_message
    original_stop = PSClient.stop_listening
    poke_env_patches.install()
    installed = DoubleBattle.parse_message
    # Installing again should be a no-op idempotent call
    poke_env_patches.install()
    assert DoubleBattle.parse_message is installed
    assert installed is not original
    assert PSClient.stop_listening is not original_stop
    # Uninstalling must restore original method references
    poke_env_patches.uninstall_for_tests()
    assert DoubleBattle.parse_message is original
    assert PSClient.stop_listening is original_stop

    # Verify log filtering only suppresses targeted warning messages on the designated logger
    target = logging.getLogger("test.poke-env")
    other = logging.getLogger("test.other")
    poke_env_patches.install(target)
    record = logging.LogRecord("test", logging.WARNING, "", 0, "is active, but it's not", (), None)
    assert not target.filter(record)
    assert other.filter(record)
    poke_env_patches.uninstall_for_tests()


def test_protocol_line_capture_is_opt_in_and_survives_idempotent_install() -> None:
    """Verify protocol history is retained only when explicitly requested."""
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


def test_transform_patch_accepts_species_target() -> None:
    """Verify species-form Transform lines are normalized for poke-env's reference-only parser."""
    poke_env_patches.install()
    try:
        battle = DoubleBattle("transform-test", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "switch", "p2a: Pikachu", "Pikachu, L50", "100/100"])
        battle.parse_message(
            ["", "-transform", "p1a: Ditto", "Pikachu", "[from] ability: Imposter"]
        )

        transformed = battle_view(battle).active_pokemon[0]
        assert isinstance(transformed, TransformedPokemonView)
        assert transformed.species == "pikachu"
    finally:
        poke_env_patches.uninstall_for_tests()


_EV_LESS_SHEET = "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|||||50|"
_EV_BEARING_SHEET = (
    "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|252,0,0,0,4,0||||50|"
)


def _teambuilder_mon(packed: str) -> TeambuilderPokemon:
    """Parse single Pokémon record from packed Showdown team string."""
    return Teambuilder.parse_packed_team(packed)[0]


def test_poke_env_drops_the_open_team_sheet_nature_without_the_patch() -> None:
    """Demonstrate upstream poke-env bug: unpatched poke-env drops natures on OTS 0-EV Pokémon."""
    assert not poke_env_patches.is_installed()
    # Unpatched poke-env drops nature when EV string is empty
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None
    # But retains nature when explicit EV values are present
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_BEARING_SHEET)).nature == "impish"


def test_nature_patch_restores_the_open_team_sheet_nature() -> None:
    """Verify our poke_env_patches monkeypatch restores nature parsing on 0-EV Open Team Sheet Pokémon."""
    poke_env_patches.install()
    try:
        mon = Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET))
        assert mon.nature == "impish"
        no_nature = _teambuilder_mon("Incineroar||sitrusberry|intimidate|fakeout||||||50|")
        assert Pokemon(gen=9, teambuilder=no_nature).nature is None
    finally:
        poke_env_patches.uninstall_for_tests()

    # Confirm clean uninstallation leaves poke-env in original unpatched state
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None


@pytest.mark.network
def test_loopback_port_allocator_returns_distinct_reusable_ports() -> None:
    """Verify allocate_loopback_ports binds and frees distinct available TCP loopback ports."""
    ports = showdown.allocate_loopback_ports(8)
    assert len(ports) == len(set(ports))
    sockets = [socket.socket() for _ in ports]
    try:
        # Verify that all allocated ports are immediately bindable by caller
        for port, listener in zip(ports, sockets, strict=True):
            listener.bind(("127.0.0.1", port))
    finally:
        for listener in sockets:
            listener.close()
