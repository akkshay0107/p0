"""Tests for the poke-env compatibility patches."""

from __future__ import annotations

import gc
import logging
import weakref
from pathlib import Path

import pytest
from poke_env.battle import DoubleBattle, Effect, Pokemon, PokemonType, SideCondition, Status
from poke_env.player.battle_order import DoubleBattleOrder
from poke_env.teambuilder import TeambuilderPokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from p0.battle.views import TransformedPokemonView
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import NUM_IDX_EFFECT_COUNT
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view


class TestPokeEnvPatches:
    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_illusion_duplicate_display_keeps_both_slots_active(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("duplicate-illusion", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.get_pokemon(f"{role}: Zoroark", details="Zoroark, L50")
        battle.parse_message(["", "switch", f"{role}a: Staraptor", "Staraptor, L50", "68/100"])
        battle.parse_message(["", "switch", f"{role}b: Staraptor", "Staraptor, L50", "87/100"])

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        first, second = active
        assert first is not None and second is not None
        assert first is not second
        assert first.current_hp == 68
        assert second.current_hp == 87
        battle.parse_message(["", "-unboost", f"{role}b: Staraptor", "atk", "2"])
        assert first.boosts["atk"] == 0
        assert second.boosts["atk"] == -2

        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark, L50"])
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        first, second = active
        assert first is not None and first.species == "zoroark"
        assert first.current_hp == 68
        assert first.boosts["atk"] == 0
        assert second is not None and second.species == "staraptor"
        assert second.current_hp == 87
        assert second.boosts["atk"] == -2

    @pytest.mark.parametrize("role,row", (("p1", 0), ("p2", 6)))
    def test_temporary_form_resets_in_bench_and_return_observations(
        self, role: str, row: int
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("form-observation", "Alice", logging.getLogger("test"), gen=9)
        builder = ObservationBuilder(default_runtime_resources())
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Castform", "Castform, L50", "100/100"])
        battle.parse_message(
            [
                "",
                "-formechange",
                f"{role}a: Castform",
                "Castform-Rainy",
                "[from] ability: Forecast",
            ]
        )
        rainy = builder.build(battle_view(battle))
        assert rainy.categorical[row, 0].item() == builder.tokenizer.id_for(
            "species", "castformrainy"
        )

        battle.parse_message(["", "switch", f"{role}a: Pikachu", "Pikachu, L50", "100/100"])
        benched = builder.build(battle_view(battle))
        bench_row = row + 2
        assert benched.categorical[bench_row, 0].item() == builder.tokenizer.id_for(
            "species", "castform"
        )
        assert benched.categorical[bench_row, 3].item() == builder.tokenizer.type_id(
            PokemonType.NORMAL
        )

        battle.parse_message(["", "switch", f"{role}a: Castform", "Castform, L50", "100/100"])
        returned = builder.build(battle_view(battle))
        assert returned.categorical[row, 0].item() == benched.categorical[bench_row, 0].item()
        assert returned.categorical[row, 3].item() == benched.categorical[bench_row, 3].item()

    def test_illusion_reveal_retains_boosts_on_actual_pokemon(self) -> None:
        poke_env_patches.install()
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

    def test_cached_battle_view_does_not_keep_battle_alive(self) -> None:
        battle = DoubleBattle("weak-view-test", "Alice", logging.getLogger("test"), gen=9)
        view = battle_view(battle)
        battle_ref = weakref.ref(battle)
        del battle
        gc.collect()
        assert battle_ref() is None
        with pytest.raises(ReferenceError):
            _ = view.team

    def test_repeated_install_applies_each_patch_once_and_scopes_the_log_filter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        message = "is active, but it's not"
        target = logging.getLogger("test.poke-env")
        other = logging.getLogger("test.other")
        poke_env_patches.install(target)
        poke_env_patches.install(target)
        battle = DoubleBattle("install-test", "Alice", logging.getLogger("test"), gen=9)
        for event in (
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", "p1a: Aegislash", "Aegislash, L50", "100/100"],
            ["", "-formechange", "p1a: Aegislash", "Aegislash-Blade"],
            ["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"],
        ):
            battle.parse_message(event)

        with caplog.at_level(logging.WARNING):
            target.warning(message)
            other.warning(message)

        assert [record.name for record in caplog.records] == ["test.other"]
        assert battle.team["p1: Aegislash"].species == "aegislash"

    def test_transform_patch_accepts_species_target(self) -> None:
        poke_env_patches.install()
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

    def test_transform_targets_a_pokemon_renamed_by_an_earlier_reveal(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("transform-reveal", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        for species in ("Ninetales-Alola", "Zoroark-Hisui"):
            battle.get_pokemon(
                f"p2: {species}",
                request={
                    "active": False,
                    "baseAbility": "illusion" if species.startswith("Zoroark") else "snowwarning",
                    "condition": "100/100",
                    "details": f"{species}, L50",
                    "ident": f"p2: {species}",
                    "item": "",
                    "moves": [],
                },
            )
        battle.parse_message(["", "teamsize", "p1", "2"])
        battle.parse_message(["", "teamsize", "p2", "2"])
        battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
        # Zoroark-Hisui leads looking like Ninetales and is revealed, then Ninetales enters.
        battle.parse_message(["", "switch", "p2a: Ninetales", "Ninetales-Alola, L50", "100/100"])
        battle.parse_message(["", "replace", "p2a: Zoroark", "Zoroark-Hisui, L50"])
        battle.parse_message(["", "switch", "p2a: Ninetales", "Ninetales-Alola, L50", "100/100"])
        battle.parse_message(["", "-transform", "p1a: Ditto", "p2a: Ninetales"])

        transformed = battle_view(battle).active_pokemon[0]
        assert isinstance(transformed, TransformedPokemonView)
        assert transformed.species == "ninetalesalola"
        assert len(battle.opponent_team) == 2

    def test_transform_keeps_the_form_copied_at_transform_time(self) -> None:
        poke_env_patches.install()
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

    def test_transform_keeps_the_moves_known_at_transform_time(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("transform-moves", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "switch", "p2a: Mew", "Mew, L50", "100/100"])
        battle.parse_message(["", "-transform", "p1a: Ditto", "Mew", "[from] ability: Imposter"])
        battle.parse_message(["", "move", "p2a: Mew", "Psychic", "p1a: Ditto"])

        transformed = battle_view(battle).active_pokemon[0]
        target = battle.opponent_active_pokemon[0]
        assert isinstance(transformed, TransformedPokemonView)
        assert "psychic" not in transformed.moves
        assert target is not None
        assert "psychic" in target.moves

    def test_transform_retains_original_hp_and_tracks_pp_and_type_changes(self) -> None:
        poke_env_patches.install()
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
        move_view = transformed.moves["gigadrain"]
        assert move_view.current_pp == 5
        assert move_view.max_pp == 5
        assert move_view.type.name == "GRASS"
        assert move_view.category.name == "SPECIAL"

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

    def test_transform_copies_the_targets_changed_types(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("transform-protean", "Alice", logging.getLogger("test"), gen=9)
        for event in (
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", "p1a: Meowscarada", "Meowscarada, L50, F", "100/100"],
            ["", "switch", "p2b: Ditto", "Ditto, L50", "100/100"],
            [
                "",
                "-start",
                "p1a: Meowscarada",
                "typechange",
                "Normal",
                "[from] ability: Protean",
            ],
            ["", "-transform", "p2b: Ditto", "p1a: Meowscarada", "[from] ability: Imposter"],
        ):
            battle.parse_message(event)

        transformed = battle_view(battle).opponent_active_pokemon[1]
        assert isinstance(transformed, TransformedPokemonView)
        assert tuple(value.name for value in transformed.types) == ("NORMAL",)

    def test_faint_clears_transformed_types_and_size(self) -> None:
        poke_env_patches.install()
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

    def test_baton_pass_transfers_boosts_before_switch_cleanup(self) -> None:
        poke_env_patches.install()
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

    def test_shed_tail_transfers_substitute_without_boosts(self) -> None:
        poke_env_patches.install()
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

    @pytest.mark.parametrize(
        ("species", "temporary_form", "expected_active_form", "base_type"),
        [
            pytest.param("Morpeko", "Morpeko-Hangry", "morpekohangry", "ELECTRIC", id="morpeko"),
            pytest.param("Aegislash", "Aegislash-Blade", "aegislashblade", "STEEL", id="aegislash"),
            pytest.param("Cherrim", "Cherrim-Sunshine", "cherrimsunshine", "GRASS", id="cherrim"),
        ],
    )
    def test_temporary_forms_reset_to_base_species_when_switching_out(
        self,
        species: str,
        temporary_form: str,
        expected_active_form: str,
        base_type: str,
    ) -> None:
        """Keep each named temporary form and base type through a real switch stream."""
        poke_env_patches.install()
        battle = DoubleBattle(
            f"{species.lower()}-switch", "Alice", logging.getLogger("test"), gen=9
        )
        for event in (
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", f"p1a: {species}", f"{species}, L50", "100/100"],
            ["", "-formechange", f"p1a: {species}", temporary_form],
        ):
            battle.parse_message(event)
        active = battle.active_pokemon[0]
        assert active is not None and active.species == expected_active_form

        battle.parse_message(["", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100"])

        returned_to_base = battle.team[f"p1: {species}"]
        assert returned_to_base.species == species.lower()
        assert returned_to_base.type_1.name == base_type

    def test_temporary_form_resets_when_a_disguise_replaces_it_in_the_same_slot(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("morpeko-illusion", "Alice", logging.getLogger("test"), gen=9)
        for event in (
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", "p2b: Morpeko", "Morpeko, L50, F", "100/100"],
            ["", "-formechange", "p2b: Morpeko", "Morpeko-Hangry"],
            ["", "switch", "p2b: Morpeko", "Morpeko, L50, F", "64/100"],
        ):
            battle.parse_message(event)

        shown = battle.opponent_active_pokemon[1]
        assert shown is not None and shown.species == "morpeko"

    def test_form_change_clears_an_earlier_type_change(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("morpeko-soak", "Alice", logging.getLogger("test"), gen=9)
        for event in (
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", "p1a: Morpeko", "Morpeko, L50", "100/100"],
            ["", "-start", "p1a: Morpeko", "typechange", "Water"],
            ["", "-formechange", "p1a: Morpeko", "Morpeko-Hangry"],
        ):
            battle.parse_message(event)

        morpeko = battle.active_pokemon[0]
        assert morpeko is not None
        assert morpeko.types == [PokemonType.ELECTRIC, PokemonType.DARK]

    def test_stance_change_form_resets_when_aegislash_switches_out(self) -> None:
        poke_env_patches.install()
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

    def test_unlisted_temporary_form_resets_when_switching_out(self) -> None:
        poke_env_patches.install()
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

    def test_permanent_form_persists_when_switching_out(self) -> None:
        poke_env_patches.install()
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
        builder = ObservationBuilder(default_runtime_resources())
        benched = builder.build(battle_view(battle))
        assert benched.categorical[2, 0].item() == builder.tokenizer.id_for(
            "species", "palafinhero"
        )

    def test_typeadd_updates_live_types_until_switch_out(self) -> None:
        poke_env_patches.install()
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

    def test_copyboost_copies_receiver_from_donor(self, tmp_path: Path) -> None:
        poke_env_patches.install()
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

    @pytest.mark.parametrize(
        (
            "receiver_effect",
            "donor_effect",
            "donor_species",
            "expected_effect",
            "expected_payload",
        ),
        [
            pytest.param(
                "Dragon Cheer",
                "Focus Energy",
                "Eevee",
                Effect.FOCUS_ENERGY,
                None,
                id="focus-energy",
            ),
            pytest.param(
                "Focus Energy",
                "Dragon Cheer",
                "Garchomp",
                Effect.DRAGON_CHEER,
                1,
                id="dragon-cheer",
            ),
        ],
    )
    def test_copyboost_replaces_critical_volatile_state_from_donor(
        self,
        receiver_effect: str,
        donor_effect: str,
        donor_species: str,
        expected_effect: Effect,
        expected_payload: int | None,
    ) -> None:
        """Copy each critical volatile while replacing receiver state and preserving the donor."""
        poke_env_patches.install()
        battle = DoubleBattle(
            "copyboost-critical-volatile", "Alice", logging.getLogger("test"), gen=9
        )
        events = [
            ["", "player", "p1", "Alice", "", ""],
            ["", "player", "p2", "Bob", "", ""],
            ["", "switch", "p1a: Receiver", "Pikachu, L50", "100/100"],
            ["", "switch", "p1b: Donor", f"{donor_species}, L50", "100/100"],
            ["", "-start", "p1a: Receiver", receiver_effect],
            ["", "-start", "p1b: Donor", donor_effect],
            ["", "-copyboost", "p1a: Receiver", "p1b: Donor"],
        ]
        for event in events:
            battle.parse_message(event)

        receiver = battle.get_pokemon("p1a: Receiver")
        donor = battle.get_pokemon("p1b: Donor")
        assert receiver_effect.lower().replace(" ", "_") not in {
            effect.name.lower() for effect in receiver.effects
        }
        if expected_payload is None:
            assert expected_effect in receiver.effects
            assert expected_effect in donor.effects
        else:
            assert receiver.effects[expected_effect] == expected_payload
            assert donor.effects[expected_effect] == expected_payload

    def test_round_moved_up_by_round_spends_the_users_unrevealed_round(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("round-boost", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Espeon", "Espeon, L50, M", "100/100"])
        battle.parse_message(["", "switch", "p2b: Politoed", "Politoed, L50, F", "100/100"])
        battle.parse_message(["", "move", "p1a: Espeon", "Round", "p2b: Politoed"])
        battle.parse_message(
            ["", "move", "p2b: Politoed", "Round", "p1a: Espeon", "[from] move: Round"]
        )

        round_move = battle.opponent_team["p2: Politoed"].moves["round"]
        assert round_move.current_pp == round_move.max_pp - 1

    def test_leppa_berry_uses_ripen_and_clamps_pp(self) -> None:
        poke_env_patches.install()
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
        assert move.current_pp == 12
        battle.parse_message(
            ["", "-activate", "p1a: Receiver", "item: Leppa Berry", "Tackle", "[consumed]"]
        )
        assert move.current_pp == move.max_pp

    def test_leppa_berry_uses_ten_pp_without_ripen(self) -> None:
        poke_env_patches.install()
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
        assert move.current_pp == 22

    def test_leppa_berry_rejects_unknown_move(self) -> None:
        poke_env_patches.install()
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


_EV_LESS_SHEET = "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|||||50|"
_EV_BEARING_SHEET = (
    "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|252,0,0,0,4,0||||50|"
)


def _teambuilder_mon(packed: str) -> TeambuilderPokemon:
    return Teambuilder.parse_packed_team(packed)[0]


class TestPokeEnvNaturePatch:
    def test_nature_patch_restores_the_open_team_sheet_nature(self) -> None:
        assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None
        assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_BEARING_SHEET)).nature == "impish"

        poke_env_patches.install()
        mon = Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET))
        assert mon.nature == "impish"
        ev_bearing = Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_BEARING_SHEET))
        assert ev_bearing.nature == "impish"
        no_nature = _teambuilder_mon("Incineroar||sitrusberry|intimidate|fakeout||||||50|")
        assert Pokemon(gen=9, teambuilder=no_nature).nature is None


def _request_pokemon(ident: str, details: str, condition: str, active: bool) -> dict[str, object]:
    return {
        "ident": ident,
        "details": details,
        "condition": condition,
        "active": active,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": ["protect"],
        "baseAbility": "illusion" if ident.endswith("Zoroark") else "noability",
        "item": "",
        "pokeball": "pokeball",
        "ability": "illusion" if ident.endswith("Zoroark") else "noability",
        "commanding": False,
        "reviving": False,
    }


def _request_move(name: str, target: str) -> dict[str, object]:
    return {
        "move": name,
        "id": name.replace(" ", "").lower(),
        "pp": 8,
        "maxpp": 8,
        "target": target,
        "disabled": False,
    }


def _rotom_houndstone_request(houndstone_moves: list[dict[str, object]]) -> dict[str, object]:
    """A move request for Rotom-Wash beside a Houndstone offered the given moves."""
    rotom = _request_pokemon("p1: Rotom", "Rotom-Wash, L50", "157/157", True)
    rotom["moves"] = ["rest", "nightshade"]
    houndstone = _request_pokemon("p1: Houndstone", "Houndstone, L50, M", "179/179", True)
    houndstone["moves"] = ["facade", "uproar"]
    return {
        "active": [
            {"moves": [_request_move("Rest", "self"), _request_move("Night Shade", "normal")]},
            {"moves": houndstone_moves},
        ],
        "side": {"name": "Alice", "id": "p1", "pokemon": [rotom, houndstone]},
    }


class TestLockedMoveChoice:
    def test_locked_move_is_sent_by_position_and_the_partner_is_unchanged(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("locked-uproar", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Rotom", "Rotom-Wash, L50", "157/157"])
        battle.parse_message(["", "switch", "p1b: Houndstone", "Houndstone, L50, M", "179/179"])
        battle.parse_message(["", "switch", "p2a: Blissey", "Blissey, L50, F", "100/100"])
        battle.parse_message(["", "switch", "p2b: Snorlax", "Snorlax, L50, F", "100/100"])
        # Showdown's request entry for a Pokemon on a later turn of Uproar.
        battle.parse_request(_rotom_houndstone_request([{"move": "Uproar", "id": "uproar"}]))

        free, locked = battle.valid_orders
        night_shade = next(order for order in free if order.message == "/choose move nightshade 1")
        order = DoubleBattleOrder(night_shade, locked[0])

        assert order.message == "/choose move nightshade 1, move uproar"
        assert (
            poke_env_patches.locked_choice(battle, order.message)
            == "/choose move nightshade 1, move 1"
        )

    def test_choice_is_unchanged_when_no_pokemon_is_locked(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("free-uproar", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Rotom", "Rotom-Wash, L50", "157/157"])
        battle.parse_message(["", "switch", "p1b: Houndstone", "Houndstone, L50, M", "179/179"])
        battle.parse_request(
            _rotom_houndstone_request(
                [_request_move("Facade", "normal"), _request_move("Uproar", "randomNormal")]
            )
        )

        message = "/choose move rest, move uproar"
        assert poke_env_patches.locked_choice(battle, message) == message

    def test_struggle_is_not_treated_as_locked(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("struggle", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Rotom", "Rotom-Wash, L50", "157/157"])
        battle.parse_message(["", "switch", "p1b: Houndstone", "Houndstone, L50, M", "179/179"])
        # Showdown gives Struggle its target, so naming it by id is accepted.
        battle.parse_request(_rotom_houndstone_request([_request_move("Struggle", "randomNormal")]))

        message = "/choose move rest, move struggle"
        assert poke_env_patches.locked_choice(battle, message) == message

    def test_default_choice_is_unchanged_beside_a_locked_pokemon(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("locked-default", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Rotom", "Rotom-Wash, L50", "157/157"])
        battle.parse_message(["", "switch", "p1b: Houndstone", "Houndstone, L50, M", "179/179"])
        battle.parse_request(_rotom_houndstone_request([{"move": "Uproar", "id": "uproar"}]))

        assert poke_env_patches.locked_choice(battle, "/choose default") == "/choose default"


class TestIllusionTracking:
    def test_public_names_keep_targeting_the_own_disguise_after_a_private_request(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-slot-reference", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Weavile", "Weavile, L50", "100/100"])
        battle.parse_message(["", "switch", "p1b: Eevee", "Eevee, L50", "100/100"])
        battle.parse_message(["", "switch", "p2a: Pikachu", "Pikachu, L50", "100/100"])
        weavile = _request_pokemon("p1: Weavile", "Weavile, L50", "100/100", False)
        weavile.update(baseAbility="pressure", ability="pressure")
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Zoroark", "Zoroark, L50", "100/100", True),
                        _request_pokemon("p1: Eevee", "Eevee, L50", "100/100", True),
                        weavile,
                    ],
                },
                "rqid": 1,
            }
        )

        battle.parse_message(["", "move", "p2a: Pikachu", "Quick Attack", "p1a: Weavile"])
        battle.parse_message(["", "-unboost", "p1a: Weavile", "atk", "1"])
        battle.parse_message(["", "-item", "p1a: Weavile", "Life Orb"])

        zoroark = battle.active_pokemon[0]
        assert zoroark is not None and zoroark.species == "zoroark"
        assert zoroark.boosts["atk"] == -1
        assert zoroark.item == "lifeorb"
        assert battle.team["p1: Weavile"].boosts["atk"] == 0
        assert battle.team["p1: Weavile"].item != "lifeorb"
        pikachu = battle.opponent_active_pokemon[0]
        assert pikachu is not None
        move = pikachu.moves["quickattack"]
        assert move.current_pp == move.max_pp - 1

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_real_pokemon_entering_beside_its_disguise_has_no_disguise_boosts(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-boosts", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}b: Toxapex", "Toxapex, L50, M", "100/100"])
        battle.parse_message(["", "-unboost", f"{role}b: Toxapex", "spe", "2"])
        battle.parse_message(["", "switch", f"{role}a: Toxapex", "Toxapex, L50, M", "100/100"])

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        real, disguise = active
        assert real is not None and disguise is not None
        assert real.boosts["spe"] == 0
        assert disguise.boosts["spe"] == -2

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_moves_the_disguise_status_to_the_illusion_user(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-status", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "drag", f"{role}b: Primarina", "Primarina, L50, F", "56/100 par"])
        battle.parse_message(["", "replace", f"{role}b: Zoroark", "Zoroark-Hisui, L50, M"])

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        revealed = active[1]
        assert revealed is not None and revealed.species == "zoroarkhisui"
        assert revealed.status == Status.PAR

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_leaves_the_active_real_pokemon_as_its_team_entry(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-reveal-team", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}b: Glimmora", "Glimmora, L50, F", "100/100"])
        battle.parse_message(["", "switch", f"{role}a: Glimmora", "Glimmora, L50, F", "100/100"])
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark, L50, M"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        glimmoras = [mon for mon in team.values() if mon.species == "glimmora"]
        assert glimmoras == [active[1]]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_revealed_zoroark_at_zero_hp_leaves_the_real_pokemon_as_its_team_entry(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-zero-hp", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}b: Victreebel", "Victreebel, L50, F", "60/100"])
        battle.parse_message(["", "switch", f"{role}a: Victreebel", "Victreebel, L50, F", "14/100"])
        battle.parse_message(["", "-damage", f"{role}a: Victreebel", "0 fnt"])
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark-Hisui, L50, M"])
        battle.parse_message(["", "faint", f"{role}a: Zoroark"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        victreebels = [mon for mon in team.values() if mon.species == "victreebel"]
        assert victreebels == [active[1]]
        assert active[1] is not None and not active[1].fainted

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_disguise_switching_out_leaves_the_active_display_as_its_team_entry(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-switch-out", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}b: Spiritomb", "Spiritomb, L50, M", "100/100"])
        battle.parse_message(["", "drag", f"{role}a: Spiritomb", "Spiritomb, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}a: Eelektross", "Eelektross, L50, M", "84/100"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        spiritombs = [mon for mon in team.values() if mon.species == "spiritomb"]
        assert spiritombs == [active[1]]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_display_fainting_leaves_the_active_display_as_its_team_entry(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-faint", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(
            ["", "switch", f"{role}b: Corviknight", "Corviknight, L50, M", "85/100"]
        )
        battle.parse_message(
            ["", "switch", f"{role}a: Corviknight", "Corviknight, L50, M", "53/100 par"]
        )
        battle.parse_message(["", "faint", f"{role}a: Corviknight"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        corviknights = [mon for mon in team.values() if mon.species == "corviknight"]
        assert corviknights == [active[1]]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_display_fainting_in_another_form_leaves_the_active_display_as_its_team_entry(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-forme-faint", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}b: Morpeko", "Morpeko, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}a: Morpeko", "Morpeko, L50, M", "38/100"])
        battle.parse_message(
            [
                "",
                "-formechange",
                f"{role}a: Morpeko",
                "Morpeko-Hangry",
                "[from] ability: Hunger Switch",
            ]
        )
        battle.parse_message(["", "faint", f"{role}a: Morpeko"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        morpekos = [mon for mon in team.values() if mon.base_species == "morpeko"]
        assert morpekos == [active[1]]
        assert not morpekos[0].fainted

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_real_pokemon_returning_beside_its_disguise_keeps_its_changed_form(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-busted", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Mimikyu", "Mimikyu, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}b: Mimikyu", "Mimikyu, L50, M", "100/100"])
        battle.parse_message(["", "detailschange", f"{role}b: Mimikyu", "Mimikyu-Busted, L50, M"])
        battle.parse_message(["", "switch", f"{role}b: Glalie", "Glalie, L50, F", "100/100"])
        battle.parse_message(
            ["", "switch", f"{role}b: Mimikyu", "Mimikyu-Busted, L50, M", "84/100"]
        )

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        assert active[0] is not None and active[0].species == "mimikyu"
        assert active[1] is not None and active[1].species == "mimikyubusted"

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_ally_switch_moves_the_display_copy_with_its_slot(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-ally-switch", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        # Zoroark looks like Houndstone in slot b, then the real Houndstone enters slot a.
        battle.parse_message(["", "switch", f"{role}b: Houndstone", "Houndstone, L50, F", "42/100"])
        battle.parse_message(["", "switch", f"{role}a: Houndstone", "Houndstone, L50, F", "24/100"])
        battle.parse_message(["", "swap", f"{role}a: Houndstone", "1", "[from] move: Ally Switch"])
        battle.parse_message(["", "-damage", f"{role}a: Houndstone", "0 fnt"])
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark-Hisui, L50, M"])
        battle.parse_message(["", "faint", f"{role}a: Zoroark"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        houndstones = [mon for mon in team.values() if mon.species == "houndstone"]
        zoroarks = [mon for mon in team.values() if mon.species == "zoroarkhisui"]
        assert houndstones == [active[1]]
        assert not houndstones[0].fainted
        assert [mon.fainted for mon in zoroarks] == [True]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_disguise_of_a_transformed_ditto_leaves_the_ditto_species(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-ditto-copy", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "switch", f"{role}b: Lapras", "Lapras, L50, F", "100/100"])
        battle.parse_message(["", "switch", f"{role}b: Ditto", "Lapras, L50, F", "100/100"])

        team = battle.team if role == "p1" else battle.opponent_team
        assert [mon.species for mon in team.values() if mon.name == "Ditto"] == ["ditto"]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_returns_the_team_entry_to_the_real_pokemon_that_left(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-hero", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        # Zoroark looks like Palafin in slot b; the real Palafin enters slot a and leaves.
        battle.parse_message(["", "switch", f"{role}b: Palafin", "Palafin, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}a: Palafin", "Palafin, L50, M", "100/100"])
        battle.parse_message(["", "detailschange", f"{role}a: Palafin", "Palafin-Hero, L50, M"])
        battle.parse_message(["", "switch", f"{role}a: Pinsir", "Pinsir, L50, F", "100/100"])
        battle.parse_message(["", "replace", f"{role}b: Zoroark", "Zoroark-Hisui, L50, F"])

        team = battle.team if role == "p1" else battle.opponent_team
        palafins = [mon for mon in team.values() if mon.base_species == "palafin"]
        assert [mon.species for mon in palafins] == ["palafinhero"]
        assert not palafins[0].active

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_keeps_a_type_the_disguise_already_showed(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-trick", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Gourgeist", "Gourgeist, L50, M", "100/100"])
        battle.parse_message(
            [
                "",
                "-start",
                f"{role}a: Gourgeist",
                "typeadd",
                "Ghost",
                "[from] move: Trick-or-Treat",
            ]
        )
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark, L50, M"])

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        assert active[0] is not None and active[0].species == "zoroark"
        assert [type_.name for type_ in active[0].types] == ["DARK", "GHOST"]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_fainted_display_returning_means_the_illusion_user_fainted(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-memento", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        zoroark = battle.get_pokemon(
            f"{role}: Zoroark",
            request=_request_pokemon(f"{role}: Zoroark", "Zoroark, L50, M", "100/100", False),
        )
        battle.parse_message(["", "switch", f"{role}b: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "faint", f"{role}b: Ditto"])
        battle.parse_message(["", "switch", f"{role}b: Ditto", "Ditto, L50", "100/100"])

        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        assert zoroark.fainted
        assert active[1] is not None and active[1].species == "ditto"
        assert not active[1].fainted

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_moves_the_disguise_type_change_to_the_illusion_user(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-reveal-soak", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "-start", f"{role}a: Ditto", "typechange", "Water"])
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark-Hisui, L50, M"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        assert active[0] is not None and active[0].species == "zoroarkhisui"
        assert [type_.name for type_ in active[0].types] == ["WATER"]
        ditto = next(mon for mon in team.values() if mon.species == "ditto")
        assert [type_.name for type_ in ditto.types] == ["NORMAL"]

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_after_a_twin_fainted_means_the_real_pokemon_fainted(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-twin-faint", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Hatterene", "Hatterene, L50, F", "74/100"])
        battle.parse_message(["", "switch", f"{role}b: Hatterene", "Hatterene, L50, F", "36/100"])
        battle.parse_message(["", "faint", f"{role}b: Hatterene"])
        battle.parse_message(["", "replace", f"{role}a: Zoroark", "Zoroark-Hisui, L50, M"])

        team = battle.team if role == "p1" else battle.opponent_team
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        hatterenes = [mon for mon in team.values() if mon.species == "hatterene"]
        assert [mon.fainted for mon in hatterenes] == [True]
        assert active[0] is not None and active[0].species == "zoroarkhisui"

    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_reveal_elsewhere_after_a_twin_fainted_means_the_real_pokemon_fainted(
        self, role: str
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-late-reveal", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        # Zoroark looks like Salazzle in slot a; the real Salazzle enters slot b and faints.
        battle.parse_message(["", "switch", f"{role}a: Salazzle", "Salazzle, L50, F", "87/100"])
        battle.parse_message(["", "switch", f"{role}b: Salazzle", "Salazzle, L50, F", "1/100"])
        battle.parse_message(["", "faint", f"{role}b: Salazzle"])
        # Zoroark leaves, then comes back looking like Glimmora and is revealed.
        battle.parse_message(["", "switch", f"{role}a: Swampert", "Swampert, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}b: Glimmora", "Glimmora, L50, F", "87/100"])
        battle.parse_message(["", "replace", f"{role}b: Zoroark", "Zoroark, L50, F"])

        team = battle.team if role == "p1" else battle.opponent_team
        salazzles = [mon for mon in team.values() if mon.species == "salazzle"]
        assert [mon.fainted for mon in salazzles] == [True]

    def test_illusion_user_entering_as_itself_after_a_twin_fainted_means_the_twin_fainted(
        self,
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-shows-itself", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.apply_teambuilder_team(
            "p2",
            Teambuilder.parse_packed_team(
                "Glimmora||FocusSash|ToxicDebris|Hex|Modest||F|||50|]"
                "Rillaboom||WideLens|GrassySurge|LowKick|Hasty||F|||50|]"
                "Zoroark-Hisui||SpellTag|Illusion|Hex|Naive||F|||50|"
            ),
            battle.teampreview_opponent_team,
        )
        # Zoroark looks like Glimmora in slot a; the real Glimmora enters slot b and faints.
        battle.parse_message(["", "switch", "p2a: Glimmora", "Glimmora, L50, F", "100/100"])
        battle.parse_message(["", "switch", "p2b: Glimmora", "Glimmora, L50, F", "100/100"])
        battle.parse_message(["", "faint", "p2b: Glimmora"])
        # Zoroark leaves, then comes back as itself because no ally is left to copy.
        battle.parse_message(["", "switch", "p2a: Rillaboom", "Rillaboom, L50, F", "100/100"])
        battle.parse_message(["", "switch", "p2b: Zoroark", "Zoroark-Hisui, L50, F", "100/100"])

        glimmoras = [mon for mon in battle.opponent_team.values() if mon.species == "glimmora"]
        assert [mon.fainted for mon in glimmoras] == [True]

    def test_off_field_target_named_by_its_base_species_is_its_team_sheet_entry(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-notarget", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "poke", "p2", "Zoroark-Hisui, L50, F", ""])
        battle.parse_message(["", "poke", "p2", "Typhlosion-Hisui, L50, F", ""])
        battle.apply_teambuilder_team(
            "p2",
            Teambuilder.parse_packed_team(
                "Zoroark-Hisui||Falinksite|Illusion|Agility,Substitute|Modest||F|||50|]"
                "Typhlosion-Hisui||Salamencite|Frisk|HyperBeam,Substitute|Impish||F|||50|"
            ),
            battle.teampreview_opponent_team,
        )
        battle.parse_message(["", "teamsize", "p1", "2"])
        battle.parse_message(["", "teamsize", "p2", "2"])
        battle.parse_message(["", "switch", "p1a: Grapploct", "Grapploct, L50, M", "158/158"])
        battle.parse_message(
            ["", "switch", "p2a: Typhlosion", "Typhlosion-Hisui, L50, F", "100/100"]
        )
        battle.parse_message(["", "faint", "p2a: Typhlosion"])
        # The disguised Zoroark-Hisui fainted, and Showdown names it by its default nickname.
        battle.parse_message(
            ["", "move", "p1a: Grapploct", "Giga Impact", "p2: Zoroark", "[notarget]"]
        )

        assert len(battle.opponent_team) == 2
        assert battle.get_pokemon("p2: Zoroark").species == "zoroarkhisui"

    def test_disguise_entering_as_a_transformed_ditto_leaves_the_ditto_transformed(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-transform", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Beedrill", "Beedrill, L50, M", "100/100"])
        battle.parse_message(["", "switch", "p2b: Ditto", "Ditto, L50", "100/100"])
        battle.parse_message(["", "-transform", "p2b: Ditto", "p1a: Beedrill"])
        # The disguise copies the name and the transformed look of the real Ditto.
        battle.parse_message(["", "switch", "p2a: Ditto", "Beedrill, L50", "94/100"])

        shown = battle_view(battle).opponent_active_pokemon
        assert [mon.species for mon in shown] == ["beedrill", "beedrill"]
        assert [(mon.type_1, mon.type_2) for mon in shown] == [
            (PokemonType.BUG, PokemonType.POISON),
            (PokemonType.BUG, PokemonType.POISON),
        ]

    def test_request_places_the_disguised_own_illusion_user_in_its_slot(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-request", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Hawlucha", "Hawlucha, L50, F", "165/165"])
        battle.parse_message(["", "switch", "p1b: Farigiraf", "Farigiraf, L50, M", "201/201"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Hawlucha", "Hawlucha, L50, F", "165/165", True),
                        _request_pokemon("p1: Farigiraf", "Farigiraf, L50, M", "201/201", True),
                        _request_pokemon("p1: Zoroark", "Zoroark, L50, M", "159/159", False),
                    ],
                },
                "rqid": 1,
            }
        )
        battle.parse_message(["", "switch", "p1b: Farigiraf", "Farigiraf, L50, M", "159/159"])
        battle.parse_message(["", "-boost", "p1b: Farigiraf", "atk", "1"])
        battle.parse_message(["", "faint", "p1a: Hawlucha"])
        battle.parse_request(
            {
                "forceSwitch": [True, False],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Hawlucha", "Hawlucha, L50, F", "0 fnt", True),
                        _request_pokemon("p1: Zoroark", "Zoroark, L50, M", "159/159", True),
                        _request_pokemon("p1: Farigiraf", "Farigiraf, L50, M", "201/201", False),
                    ],
                },
                "rqid": 2,
            }
        )

        revealed = battle.active_pokemon[1]
        assert revealed is not None and revealed.species == "zoroark"
        assert revealed.boosts["atk"] == 1
        assert not battle.team["p1: Farigiraf"].active

    def test_request_moves_the_disguise_type_change_to_the_illusion_user(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-soak", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        party = (
            ("p1: Hawlucha", "Hawlucha, L50, F", "165/165"),
            ("p1: Farigiraf", "Farigiraf, L50, M", "201/201"),
            ("p1: Zoroark", "Zoroark, L50, M", "159/159"),
        )
        battle.parse_message(["", "switch", "p1a: Hawlucha", "Hawlucha, L50, F", "165/165"])
        battle.parse_message(["", "switch", "p1b: Farigiraf", "Farigiraf, L50, M", "201/201"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(ident, details, condition, index < 2)
                        for index, (ident, details, condition) in enumerate(party)
                    ],
                },
                "rqid": 1,
            }
        )
        # Zoroark comes in looking like Farigiraf, and is soaked and boosted that turn.
        battle.parse_message(["", "switch", "p1b: Farigiraf", "Farigiraf, L50, M", "159/159"])
        battle.parse_message(["", "-start", "p1b: Farigiraf", "typechange", "Water"])
        battle.parse_message(["", "-boost", "p1b: Farigiraf", "spe", "1"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(*party[0], True),
                        _request_pokemon(*party[2], True),
                        _request_pokemon(*party[1], False),
                    ],
                },
                "rqid": 2,
            }
        )

        zoroark = battle.active_pokemon[1]
        farigiraf = battle.team["p1: Farigiraf"]
        assert zoroark is not None and zoroark.species == "zoroark"
        assert [type_.name for type_ in zoroark.types] == ["WATER"]
        assert zoroark.boosts["spe"] == 1
        assert [type_.name for type_ in farigiraf.types] == ["NORMAL", "PSYCHIC"]
        assert farigiraf.boosts["spe"] == 0
        assert not farigiraf.effects

    def test_type_change_after_a_request_reaches_the_own_disguise_beside_its_twin(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-twin-soak", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        party = [
            _request_pokemon("p1: Gourgeist", "Gourgeist, L50, M", "40/138", True),
            _request_pokemon("p1: Zoroark", "Zoroark-Hisui, L50, F", "38/132", True),
        ]
        battle.parse_request(
            {"teamPreview": True, "side": {"name": "Alice", "id": "p1", "pokemon": party}}
        )
        # Zoroark enters slot b looking like Gourgeist, then the real Gourgeist enters slot a.
        battle.parse_message(["", "switch", "p1b: Gourgeist", "Gourgeist, L50, M", "38/132"])
        battle.parse_message(["", "switch", "p1a: Gourgeist", "Gourgeist, L50, M", "40/138"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {"name": "Alice", "id": "p1", "pokemon": party},
                "rqid": 2,
            }
        )
        battle.parse_message(["", "-start", "p1b: Gourgeist", "typechange", "Water"])

        gourgeist, zoroark = battle.active_pokemon
        assert zoroark is not None and zoroark is battle.team["p1: Zoroark"]
        assert [type_.name for type_ in zoroark.types] == ["WATER"]
        assert gourgeist is not None
        assert [type_.name for type_ in gourgeist.types] == ["GHOST", "GRASS"]

    def test_added_type_of_a_twin_stays_with_it_when_the_own_disguise_enters_beside_it(
        self,
    ) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-twin-typeadd", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        gourgeist = ("p1: Gourgeist", "Gourgeist, L50, F", "130/130")
        scolipede = ("p1: Scolipede", "Scolipede, L50, M", "80/140")
        zoroark = ("p1: Zoroark", "Zoroark, L50, F", "136/136")
        battle.parse_request(
            {
                "teamPreview": True,
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(*gourgeist, True),
                        _request_pokemon(*scolipede, True),
                        _request_pokemon(*zoroark, False),
                    ],
                },
            }
        )
        battle.parse_message(["", "switch", "p1a: Gourgeist", "Gourgeist, L50, F", "130/130"])
        battle.parse_message(["", "switch", "p1b: Scolipede", "Scolipede, L50, M", "80/140"])
        battle.parse_message(["", "-start", "p1b: Scolipede", "typeadd", "Ghost"])
        battle.parse_message(["", "faint", "p1a: Gourgeist"])
        # Zoroark replaces Gourgeist looking like the Scolipede beside it.
        battle.parse_message(["", "switch", "p1a: Scolipede", "Scolipede, L50, M", "136/136"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(*zoroark, True),
                        _request_pokemon(*scolipede, True),
                        _request_pokemon("p1: Gourgeist", "Gourgeist, L50, F", "0 fnt", False),
                    ],
                },
                "rqid": 2,
            }
        )

        zoroark_shown, scolipede_shown = battle.active_pokemon
        assert zoroark_shown is not None and zoroark_shown.species == "zoroark"
        assert [type_.name for type_ in zoroark_shown.types] == ["DARK"]
        assert scolipede_shown is not None
        assert [type_.name for type_ in scolipede_shown.types] == ["BUG", "POISON", "GHOST"]

    def test_own_illusion_user_leading_from_the_first_party_slot_counts_as_seen(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-lead", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        party = (
            ("p1: Zoroark", "Zoroark-Hisui, L50, M", "150/150"),
            ("p1: Swalot", "Swalot, L50, F", "177/177"),
            ("p1: Hawlucha", "Hawlucha, L50, M", "135/135"),
        )
        # Showdown flags the first two party positions active even at team preview.
        battle.parse_request(
            {
                "teamPreview": True,
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(ident, details, condition, index < 2)
                        for index, (ident, details, condition) in enumerate(party)
                    ],
                },
                "rqid": 1,
            }
        )
        battle.parse_message(["", "switch", "p1a: Hawlucha", "Hawlucha, L50, M", "150/150"])
        battle.parse_message(["", "switch", "p1b: Swalot", "Swalot, L50, F", "177/177"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon(ident, details, condition, index < 2)
                        for index, (ident, details, condition) in enumerate(party)
                    ],
                },
                "rqid": 2,
            }
        )

        assert battle.active_pokemon[0] is battle.team["p1: Zoroark"]
        assert battle.team["p1: Zoroark"].revealed

    def test_request_keeps_the_real_own_pokemon_beside_its_disguise(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("illusion-own-pair", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Toxapex", "Toxapex, L50, M", "157/157"])
        battle.parse_message(["", "switch", "p1b: Hawlucha", "Hawlucha, L50, F", "165/165"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Toxapex", "Toxapex, L50, M", "157/157", True),
                        _request_pokemon("p1: Hawlucha", "Hawlucha, L50, F", "165/165", True),
                        _request_pokemon("p1: Zoroark", "Zoroark, L50, M", "159/159", False),
                    ],
                },
                "rqid": 1,
            }
        )
        battle.parse_message(["", "-unboost", "p1a: Toxapex", "spe", "2"])
        battle.parse_message(["", "switch", "p1b: Toxapex", "Toxapex, L50, M", "159/159"])
        battle.parse_message(["", "-boost", "p1b: Toxapex", "atk", "1"])
        battle.parse_request(
            {
                "active": [{"moves": []}, {"moves": []}],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Toxapex", "Toxapex, L50, M", "157/157", True),
                        _request_pokemon("p1: Zoroark", "Zoroark, L50, M", "159/159", True),
                        _request_pokemon("p1: Hawlucha", "Hawlucha, L50, F", "165/165", False),
                    ],
                },
                "rqid": 2,
            }
        )

        real, revealed = battle.active_pokemon
        assert real is not None and revealed is not None
        assert real is battle.team["p1: Toxapex"]
        assert real.boosts["spe"] == -2
        assert real.boosts["atk"] == 0
        assert revealed is battle.team["p1: Zoroark"]
        assert revealed.boosts["atk"] == 1
        assert revealed.boosts["spe"] == 0


class TestSideGuards:
    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_wide_guard_protects_its_side_until_upkeep(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("wide-guard", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Machamp", "Machamp, L50, M", "100/100"])
        battle.parse_message(["", "-singleturn", f"{role}a: Machamp", "Wide Guard"])

        conditions = battle.side_conditions if role == "p1" else battle.opponent_side_conditions
        other = battle.opponent_side_conditions if role == "p1" else battle.side_conditions
        machamp = battle.get_pokemon(f"{role}a: Machamp")
        assert SideCondition.WIDE_GUARD in conditions
        assert SideCondition.WIDE_GUARD not in other
        assert Effect.WIDE_GUARD not in machamp.effects

        battle.parse_message(["", "upkeep"])
        assert SideCondition.WIDE_GUARD not in conditions

    def test_quick_guard_with_move_prefix_ends_at_the_next_turn(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("quick-guard", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p2b: Hitmontop", "Hitmontop, L50, M", "100/100"])
        battle.parse_message(["", "-singleturn", "p2b: Hitmontop", "move: Quick Guard"])

        assert SideCondition.QUICK_GUARD in battle.opponent_side_conditions

        battle.parse_message(["", "turn", "2"])
        assert SideCondition.QUICK_GUARD not in battle.opponent_side_conditions


class TestRoost:
    def test_roost_removes_flying_until_upkeep(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("roost", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p2a: Dragonite", "Dragonite, L50, M", "100/100"])
        battle.parse_message(["", "-singleturn", "p2a: Dragonite", "move: Roost"])

        dragonite = battle.opponent_active_pokemon[0]
        assert dragonite is not None
        assert (dragonite.type_1, dragonite.type_2) == (PokemonType.DRAGON, None)

        battle.parse_message(["", "upkeep"])
        assert (dragonite.type_1, dragonite.type_2) == (PokemonType.DRAGON, PokemonType.FLYING)
        assert Effect.ROOST not in dragonite.effects

    def test_roost_makes_a_pure_flying_pokemon_normal(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("roost-pure", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p1a: Tornadus", "Tornadus, L50, M", "155/155"])
        battle.parse_message(["", "-singleturn", "p1a: Tornadus", "move: Roost"])

        tornadus = battle.active_pokemon[0]
        assert tornadus is not None
        assert tornadus.types == [PokemonType.NORMAL]

        battle.parse_message(["", "turn", "2"])
        assert tornadus.types == [PokemonType.FLYING]

    def test_type_change_during_roost_outlasts_it(self) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("roost-soak", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", "p2a: Dragonite", "Dragonite, L50, M", "100/100"])
        battle.parse_message(["", "-singleturn", "p2a: Dragonite", "move: Roost"])
        battle.parse_message(["", "-start", "p2a: Dragonite", "typechange", "Water"])
        battle.parse_message(["", "upkeep"])

        dragonite = battle.opponent_active_pokemon[0]
        assert dragonite is not None
        assert dragonite.types == [PokemonType.WATER]


class TestReviveTracking:
    @pytest.mark.parametrize("role", ("p1", "p2"))
    def test_revived_pokemon_is_unfainted_on_the_bench_and_can_return(self, role: str) -> None:
        poke_env_patches.install()
        battle = DoubleBattle("revive-tracking", "Alice", logging.getLogger("test"), gen=9)
        battle.parse_message(["", "player", "p1", "Alice", "", ""])
        battle.parse_message(["", "player", "p2", "Bob", "", ""])
        battle.parse_message(["", "switch", f"{role}a: Pawmot", "Pawmot, L50, M", "100/100"])
        battle.parse_message(["", "switch", f"{role}b: Vivillon", "Vivillon, L50, F", "100/100"])
        battle.parse_message(["", "faint", f"{role}b: Vivillon"])
        battle.parse_message(["", "switch", f"{role}b: Raichu", "Raichu, L50, F", "100/100"])
        vivillon = battle.get_pokemon(f"{role}: Vivillon")
        assert vivillon.fainted

        battle.parse_message(
            ["", "-heal", f"{role}: Vivillon", "49/100", "[from] move: Revival Blessing"]
        )

        assert not vivillon.fainted
        assert vivillon.current_hp == 49
        battle.parse_message(["", "switch", f"{role}b: Vivillon", "Vivillon, L50, F", "49/100"])
        active = battle.active_pokemon if role == "p1" else battle.opponent_active_pokemon
        assert active[1] is vivillon
        assert not vivillon.fainted
