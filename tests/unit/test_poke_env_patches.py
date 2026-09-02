"""Tests for the poke-env compatibility patches."""

from __future__ import annotations

import logging
import subprocess
import sys

from poke_env.battle import DoubleBattle, Pokemon
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder import TeambuilderPokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from p0.battle.views import TransformedPokemonView
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view


class TestPokeEnvPatches:
    def test_event_parser_import_does_not_install_poke_env_patches(self) -> None:
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

    def test_patch_installation_is_idempotent_reversible_and_logger_scoped(self) -> None:
        poke_env_patches.uninstall_for_tests()
        original = DoubleBattle.parse_message
        original_stop = PSClient.stop_listening
        poke_env_patches.install()
        installed = DoubleBattle.parse_message
        poke_env_patches.install()
        assert DoubleBattle.parse_message is installed
        assert installed is not original
        assert PSClient.stop_listening is not original_stop
        poke_env_patches.uninstall_for_tests()
        assert DoubleBattle.parse_message is original
        assert PSClient.stop_listening is original_stop

        target = logging.getLogger("test.poke-env")
        other = logging.getLogger("test.other")
        poke_env_patches.install(target)
        record = logging.LogRecord(
            "test", logging.WARNING, "", 0, "is active, but it's not", (), None
        )
        assert not target.filter(record)
        assert other.filter(record)
        poke_env_patches.uninstall_for_tests()

    def test_transform_patch_accepts_species_target(self) -> None:
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
            assert transformed.base_species == "pikachu"
        finally:
            poke_env_patches.uninstall_for_tests()


_EV_LESS_SHEET = "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|||||50|"
_EV_BEARING_SHEET = (
    "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|252,0,0,0,4,0||||50|"
)


def _teambuilder_mon(packed: str) -> TeambuilderPokemon:
    return Teambuilder.parse_packed_team(packed)[0]


class TestPokeEnvNaturePatch:
    def test_poke_env_drops_the_open_team_sheet_nature_without_the_patch(self) -> None:
        assert not poke_env_patches.is_installed()
        assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None
        assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_BEARING_SHEET)).nature == "impish"

    def test_nature_patch_restores_the_open_team_sheet_nature(self) -> None:
        poke_env_patches.install()
        try:
            mon = Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET))
            assert mon.nature == "impish"
            no_nature = _teambuilder_mon("Incineroar||sitrusberry|intimidate|fakeout||||||50|")
            assert Pokemon(gen=9, teambuilder=no_nature).nature is None
        finally:
            poke_env_patches.uninstall_for_tests()

        assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None
