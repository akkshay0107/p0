"""Tests for the poke-env compatibility patches."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from poke_env.battle import DoubleBattle, Effect, Pokemon
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
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_patch_installation_is_idempotent_reversible_and_logger_scoped(self) -> None:
        poke_env_patches.uninstall_for_tests()
        original = DoubleBattle.parse_message
        original_stop = PSClient.stop_listening
        original_start_effect = Pokemon.start_effect
        original_copy_boosts = Pokemon.copy_boosts
        poke_env_patches.install()
        installed = DoubleBattle.parse_message
        poke_env_patches.install()
        assert DoubleBattle.parse_message is installed
        assert installed is not original
        assert PSClient.stop_listening is not original_stop
        assert Pokemon.start_effect is not original_start_effect
        assert Pokemon.copy_boosts is not original_copy_boosts
        poke_env_patches.uninstall_for_tests()
        assert DoubleBattle.parse_message is original
        assert PSClient.stop_listening is original_stop
        assert Pokemon.start_effect is original_start_effect
        assert Pokemon.copy_boosts is original_copy_boosts

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

    def test_copyboost_copies_receiver_from_donor(self, tmp_path: Path) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("copyboost-test", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p1b: Donor", "Eevee, L50", "100/100"],
                ["", "-boost", "p1a: Receiver", "atk", "1"],
                ["", "-boost", "p1b: Donor", "spa", "2"],
                ["", "-copyboost", "p1a: Receiver", "p1b: Donor", "[from] move: Psych Up"],
            ):
                battle.parse_message(event)

            receiver = battle.get_pokemon("p1a: Receiver")
            donor = battle.get_pokemon("p1b: Donor")
            assert receiver.boosts["atk"] == 0
            assert receiver.boosts["spa"] == 2
            assert donor.boosts["atk"] == 0
            assert donor.boosts["spa"] == 2
            replay = battle.save_replay(tmp_path / "copyboost.html").read_text()
            assert "|-copyboost|p1a: Receiver|p1b: Donor|[from] move: Psych Up" in replay
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_copyboost_replaces_focus_energy(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("copyboost-focus", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p1b: Donor", "Eevee, L50", "100/100"],
                ["", "-start", "p1a: Receiver", "Dragon Cheer"],
                ["", "-start", "p1b: Donor", "Focus Energy"],
                ["", "-copyboost", "p1a: Receiver", "p1b: Donor"],
            ):
                battle.parse_message(event)

            receiver = battle.get_pokemon("p1a: Receiver")
            donor = battle.get_pokemon("p1b: Donor")
            assert Effect.FOCUS_ENERGY in receiver.effects
            assert Effect.DRAGON_CHEER not in receiver.effects
            assert Effect.FOCUS_ENERGY in donor.effects
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_copyboost_preserves_dragon_cheer_metadata(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle(
                "copyboost-dragon-cheer", "Alice", logging.getLogger("test"), gen=9
            )
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p1b: Donor", "Garchomp, L50", "100/100"],
                ["", "-start", "p1a: Receiver", "Focus Energy"],
                ["", "-start", "p1b: Donor", "Dragon Cheer"],
                ["", "-copyboost", "p1a: Receiver", "p1b: Donor"],
            ):
                battle.parse_message(event)

            receiver = battle.get_pokemon("p1a: Receiver")
            donor = battle.get_pokemon("p1b: Donor")
            assert receiver.effects[Effect.DRAGON_CHEER] == 1
            assert Effect.FOCUS_ENERGY not in receiver.effects
            assert donor.effects[Effect.DRAGON_CHEER] == 1
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_copyboost_preserves_gmax_chi_strike_layers(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("copyboost-gmax", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p1b: Donor", "Eevee, L50", "100/100"],
                ["", "-start", "p1a: Receiver", "Laser Focus"],
                ["", "-start", "p1b: Donor", "G-Max Chi Strike"],
                ["", "-start", "p1b: Donor", "G-Max Chi Strike"],
                ["", "-copyboost", "p1a: Receiver", "p1b: Donor"],
            ):
                battle.parse_message(event)

            receiver = battle.get_pokemon("p1a: Receiver")
            donor = battle.get_pokemon("p1b: Donor")
            assert receiver.effects[Effect.G_MAX_CHI_STRIKE] == 2
            assert Effect.LASER_FOCUS not in receiver.effects
            assert donor.effects[Effect.G_MAX_CHI_STRIKE] == 2
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_copyboost_replaces_laser_focus(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle(
                "copyboost-laser-focus", "Alice", logging.getLogger("test"), gen=9
            )
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p1b: Donor", "Eevee, L50", "100/100"],
                ["", "-start", "p1a: Receiver", "Focus Energy"],
                ["", "-start", "p1b: Donor", "Laser Focus"],
                ["", "-copyboost", "p1a: Receiver", "p1b: Donor"],
            ):
                battle.parse_message(event)

            receiver = battle.get_pokemon("p1a: Receiver")
            donor = battle.get_pokemon("p1b: Donor")
            assert Effect.LASER_FOCUS in receiver.effects
            assert Effect.FOCUS_ENERGY not in receiver.effects
            assert Effect.LASER_FOCUS in donor.effects
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_leppa_berry_uses_ripen_and_clamps_pp(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("leppa-ripen", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p2a: Donor", "Eevee, L50", "100/100"],
                ["", "-ability", "p1a: Receiver", "Ripen"],
            ):
                battle.parse_message(event)
            for _ in range(20):
                battle.parse_message(["", "move", "p1a: Receiver", "Tackle", "p2a: Donor"])

            move = battle.get_pokemon("p1a: Receiver").moves["tackle"]
            assert move.current_pp == 36
            battle.parse_message(
                ["", "-activate", "p1a: Receiver", "item: Leppa Berry", "Tackle", "[consumed]"]
            )
            assert move.current_pp == move.max_pp
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_leppa_berry_uses_ten_pp_without_ripen(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("leppa-normal", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
                ["", "switch", "p2a: Donor", "Eevee, L50", "100/100"],
            ):
                battle.parse_message(event)
            for _ in range(20):
                battle.parse_message(["", "move", "p1a: Receiver", "Tackle", "p2a: Donor"])

            move = battle.get_pokemon("p1a: Receiver").moves["tackle"]
            battle.parse_message(
                ["", "-activate", "p1a: Receiver", "item: Leppa Berry", "Tackle", "[consumed]"]
            )
            assert move.current_pp == 46
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_leppa_berry_rejects_unknown_move(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("leppa-unknown", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
            ):
                battle.parse_message(event)

            with pytest.raises(KeyError, match="unknown move"):
                battle.parse_message(
                    [
                        "",
                        "-activate",
                        "p1a: Receiver",
                        "item: Leppa Berry",
                        "Unknown Move",
                        "[consumed]",
                    ]
                )
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
