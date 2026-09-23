"""Tests for the poke-env compatibility patches."""

from __future__ import annotations

import gc
import logging
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
from poke_env.battle import DoubleBattle, Effect, Pokemon
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder import TeambuilderPokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from p0.battle.views import TransformedPokemonView
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import NUM_IDX_EFFECT_COUNT
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view


class TestPokeEnvPatches:
    def test_illusion_reveal_retains_boosts_on_actual_pokemon(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("illusion-test", "Alice", logging.getLogger("test"), gen=9)
            battle.parse_message(["", "player", "p1", "Alice", "", ""])
            battle.parse_message(["", "player", "p2", "Bob", "", ""])
            battle.parse_message(["", "switch", "p1a: Cofagrigus", "Cofagrigus, L50, M", "100/100"])
            battle.parse_message(["", "-unboost", "p1a: Cofagrigus", "spa", "1"])
            battle.parse_message(["", "-start", "p1a: Cofagrigus", "confusion"])
            battle.parse_message(["", "replace", "p1a: Zoroark", "Zoroark, L50, F"])

            revealed = battle.active_pokemon[0]
            assert revealed is not None
            assert revealed.species == "zoroark"
            assert revealed.boosts["spa"] == -1
            assert Effect.CONFUSION in revealed.effects
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_cached_battle_view_does_not_keep_battle_alive(self) -> None:
        battle = DoubleBattle("weak-view-test", "Alice", logging.getLogger("test"), gen=9)
        view = battle_view(battle)
        battle_ref = weakref.ref(battle)
        del battle
        gc.collect()
        assert battle_ref() is None
        with pytest.raises(ReferenceError):
            _ = view.team

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

    def test_transform_keeps_the_form_copied_at_transform_time(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("transform-form", "Alice", logging.getLogger("test"), gen=9)
            battle.parse_message(["", "player", "p1", "Alice", "", ""])
            battle.parse_message(["", "player", "p2", "Bob", "", ""])
            battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
            battle.parse_message(["", "switch", "p2a: Charizard", "Charizard, L50", "100/100"])
            battle.parse_message(
                ["", "-transform", "p1a: Ditto", "Charizard", "[from] ability: Imposter"]
            )
            battle.parse_message(["", "detailschange", "p2a: Charizard", "Charizard-Mega-Y, L50"])

            transformed = battle_view(battle).active_pokemon[0]
            target = battle.opponent_active_pokemon[0]
            assert isinstance(transformed, TransformedPokemonView)
            assert transformed.species == "charizard"
            assert target is not None
            assert target.species == "charizardmegay"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_transform_keeps_the_moves_known_at_transform_time(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("transform-moves", "Alice", logging.getLogger("test"), gen=9)
            battle.parse_message(["", "player", "p1", "Alice", "", ""])
            battle.parse_message(["", "player", "p2", "Bob", "", ""])
            battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
            battle.parse_message(["", "switch", "p2a: Mew", "Mew, L50", "100/100"])
            battle.parse_message(
                ["", "-transform", "p1a: Ditto", "Mew", "[from] ability: Imposter"]
            )
            battle.parse_message(["", "move", "p2a: Mew", "Psychic", "p1a: Ditto"])

            transformed = battle_view(battle).active_pokemon[0]
            target = battle.opponent_active_pokemon[0]
            assert isinstance(transformed, TransformedPokemonView)
            assert "psychic" not in transformed.moves
            assert target is not None
            assert "psychic" in target.moves
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_transform_retains_original_hp_and_tracks_pp_and_type_changes(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("transform-state", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"],
                ["", "switch", "p2a: Roserade", "Roserade, L50", "100/100"],
                ["", "move", "p2a: Roserade", "Giga Drain", "p1a: Ditto"],
                ["", "-transform", "p1a: Ditto", "Roserade", "[from] ability: Imposter"],
            ):
                battle.parse_message(event)

            transformed = battle_view(battle).active_pokemon[0]
            assert isinstance(transformed, TransformedPokemonView)
            assert transformed.species == "roserade"
            assert transformed.base_stats["hp"] == 48
            assert transformed.base_stats["spa"] == 125
            assert transformed.moves["gigadrain"].current_pp == 5

            battle.parse_message(["", "move", "p1a: Ditto", "Giga Drain", "p2a: Roserade"])
            assert battle_view(battle).active_pokemon[0].moves["gigadrain"].current_pp == 4

            battle.parse_message(
                ["", "-start", "p1a: Ditto", "typechange", "Water", "[from] move: Soak"]
            )
            current = battle_view(battle).active_pokemon[0]
            assert tuple(value.name for value in current.types) == ("WATER",)
            assert current.base_stats["hp"] == 48
            observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
            assert observation.numerical[0, 6].item() == pytest.approx(48 / 160)
            assert observation.numerical[0, 19].item() == pytest.approx(4 / 5)

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])
            battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
            returned = battle_view(battle).active_pokemon[0]
            assert not isinstance(returned, TransformedPokemonView)
            assert returned.species == "ditto"
            assert returned.weight == pytest.approx(4.0)
            assert returned.height == pytest.approx(0.3)
            assert tuple(returned.moves) == ("transform",)
            returned_observation = ObservationBuilder(default_runtime_resources()).build(
                battle_view(battle)
            )
            assert returned_observation.numerical[0, 25].item() == 0.0
            assert observation.numerical[0, 25].item() == pytest.approx(0.2)
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_faint_clears_transformed_types_and_size(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("transform-faint", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"],
                ["", "switch", "p2a: Grimmsnarl", "Grimmsnarl, L50", "100/100"],
                ["", "-transform", "p1a: Ditto", "Grimmsnarl", "[from] ability: Imposter"],
            ):
                battle.parse_message(event)
            ditto = battle.active_pokemon[0]
            assert ditto is not None
            assert [value.name for value in ditto.types] == ["DARK", "FAIRY"]

            battle.parse_message(["", "faint", "p1a: Ditto"])

            assert ditto.fainted
            assert [value.name for value in ditto.types] == ["NORMAL"]
            assert ditto.weight == pytest.approx(4.0)
            assert ditto.height == pytest.approx(0.3)
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_baton_pass_transfers_boosts_before_switch_cleanup(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("baton-pass", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Beedrill", "Beedrill, L50", "100/100"],
                ["", "-boost", "p1a: Beedrill", "atk", "3"],
                ["", "-unboost", "p1a: Beedrill", "spd", "1"],
                ["", "move", "p1a: Beedrill", "Baton Pass", "p1a: Beedrill"],
                [
                    "",
                    "switch",
                    "p1a: Crabominable",
                    "Crabominable, L50",
                    "100/100",
                    "[from] Baton Pass",
                ],
            ):
                battle.parse_message(event)

            recipient = battle_view(battle).active_pokemon[0]
            assert recipient.boosts["atk"] == 3
            assert recipient.boosts["spd"] == -1
            observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
            assert observation.numerical[0, 12].item() == pytest.approx(3 / 6)
            assert observation.numerical[0, 15].item() == pytest.approx(-1 / 6)

            for event in (
                ["", "switch", "p2a: Sylveon", "Sylveon, L50", "100/100"],
                ["", "-unboost", "p2a: Sylveon", "atk", "1"],
                ["", "-unboost", "p2a: Sylveon", "spe", "2"],
                ["", "move", "p2a: Sylveon", "Baton Pass", "p2a: Sylveon"],
                [
                    "",
                    "switch",
                    "p2a: Garchomp",
                    "Garchomp, L50",
                    "100/100",
                    "[from] Baton Pass",
                ],
            ):
                battle.parse_message(event)
            observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
            assert observation.numerical[6, 12].item() == pytest.approx(-1 / 6)
            assert observation.numerical[6, 16].item() == pytest.approx(-2 / 6)

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])
            assert battle_view(battle).active_pokemon[0].boosts["atk"] == 0
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_shed_tail_transfers_substitute_without_boosts(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("shed-tail", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Cyclizar", "Cyclizar, L50", "100/100"],
                ["", "-boost", "p1a: Cyclizar", "atk", "2"],
                ["", "-start", "p1a: Cyclizar", "Substitute"],
                ["", "move", "p1a: Cyclizar", "Shed Tail", "p1a: Cyclizar"],
                [
                    "",
                    "switch",
                    "p1a: Pikachu",
                    "Pikachu, L50",
                    "100/100",
                    "[from] Shed Tail",
                ],
            ):
                battle.parse_message(event)

            recipient = battle_view(battle).active_pokemon[0]
            assert recipient.boosts["atk"] == 0
            assert Effect.SUBSTITUTE in recipient.effects
            observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
            assert observation.numerical[0, 12].item() == 0.0
            assert observation.numerical[0, NUM_IDX_EFFECT_COUNT].item() == 1
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_forecast_form_resets_when_castform_switches_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("forecast-switch", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p2a: Castform", "Castform, L50", "100/100"],
                ["", "-ability", "p2a: Castform", "Forecast"],
                [
                    "",
                    "-formechange",
                    "p2a: Castform",
                    "Castform-Rainy",
                    "[from] ability: Forecast",
                ],
            ):
                battle.parse_message(event)
            active = battle.opponent_active_pokemon[0]
            assert active is not None
            assert active.species == "castformrainy"

            battle.parse_message(["", "switch", "p2a: Pikachu", "Pikachu, L50", "100/100"])

            castform = battle.opponent_team["p2: Castform"]
            assert castform.species == "castform"
            assert castform.type_1.name == "NORMAL"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_hunger_switch_form_resets_when_morpeko_switches_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("morpeko-switch", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Morpeko", "Morpeko, L50", "100/100"],
                ["", "-formechange", "p1a: Morpeko", "Morpeko-Hangry"],
            ):
                battle.parse_message(event)
            active = battle.active_pokemon[0]
            assert active is not None and active.species == "morpekohangry"

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

            morpeko = battle.team["p1: Morpeko"]
            assert morpeko.species == "morpeko"
            assert morpeko.type_1.name == "ELECTRIC"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_stance_change_form_resets_when_aegislash_switches_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("aegislash-switch", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Aegislash", "Aegislash, L50", "100/100"],
                ["", "-formechange", "p1a: Aegislash", "Aegislash-Blade"],
            ):
                battle.parse_message(event)
            aegislash = battle.active_pokemon[0]
            assert aegislash is not None and aegislash.species == "aegislashblade"

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

            assert aegislash.species == "aegislash"
            assert aegislash.type_1.name == "STEEL"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_unlisted_temporary_form_resets_when_switching_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("cherrim-switch", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Cherrim", "Cherrim, L50", "100/100"],
                ["", "-formechange", "p1a: Cherrim", "Cherrim-Sunshine"],
            ):
                battle.parse_message(event)
            cherrim = battle.active_pokemon[0]
            assert cherrim is not None and cherrim.species == "cherrimsunshine"

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

            assert cherrim.species == "cherrim"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_permanent_form_persists_when_switching_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("palafin-switch", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Palafin", "Palafin, L50", "100/100"],
                ["", "detailschange", "p1a: Palafin", "Palafin-Hero, L50"],
            ):
                battle.parse_message(event)
            palafin = battle.active_pokemon[0]
            assert palafin is not None and palafin.species == "palafinhero"

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

            assert palafin.species == "palafinhero"
        finally:
            poke_env_patches.uninstall_for_tests()

    def test_typeadd_updates_live_types_until_switch_out(self) -> None:
        poke_env_patches.install()
        try:
            battle = DoubleBattle("typeadd-test", "Alice", logging.getLogger("test"), gen=9)
            for event in (
                ["", "player", "p1", "Alice", "", ""],
                ["", "player", "p2", "Bob", "", ""],
                ["", "switch", "p1a: Typhlosion", "Typhlosion, L50", "100/100"],
                ["", "-start", "p1a: Typhlosion", "typeadd", "Ghost"],
            ):
                battle.parse_message(event)
            typhlosion = battle.active_pokemon[0]
            assert typhlosion is not None
            assert [value.name for value in typhlosion.types] == ["FIRE", "GHOST"]

            battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

            assert [value.name for value in typhlosion.types] == ["FIRE"]
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
