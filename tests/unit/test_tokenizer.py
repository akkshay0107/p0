"""Tests for tokenizer resolution and domain values."""

from __future__ import annotations

from poke_env.battle import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.teambuilder.teambuilder import TeambuilderPokemon

from p0.model.tokenizer import PokemonTokenizer, Resolution


class TestTokenizerContracts:
    def test_tokenizer_resolution_keeps_unknown_zero_distinct_from_known_none(
        self,
    ) -> None:
        """Verify tokenizer distinguishes between KNOWN, KNOWN_NONE (valid empty entity), OOV, and UNKNOWN."""
        tokenizer_instance = PokemonTokenizer(
            {
                "weathers": {"raindance": 4},
                "status": {"brn": 5},
                "moves": {"uturn": 7},
                "species": {"pikachu": 8},
            }
        )
        assert tokenizer_instance.id_for("moves", "U-turn") == 7
        assert tokenizer_instance.effect_id_for("status", "status: brn") == 5
        assert tokenizer_instance.resolve("weathers", "raindance") == (4, Resolution.KNOWN)
        assert tokenizer_instance.resolve("status", "brn") == (5, Resolution.KNOWN)
        assert tokenizer_instance.resolve("status", "not-a-status") == (0, Resolution.OOV)
        assert tokenizer_instance.resolve("status", None) == (0, Resolution.KNOWN_NONE)
        assert tokenizer_instance.resolve("missing", "rain") == (0, Resolution.UNKNOWN)
        assert tokenizer_instance.resolve("species", None) == (0, Resolution.KNOWN_NONE)
        assert tokenizer_instance.resolve("species", "missingno") == (0, Resolution.OOV)
        assert tokenizer_instance.resolve("species", "pikachu") == (8, Resolution.KNOWN)

    def test_enum_like_tables_lazy_cache_and_missing_member_results(self) -> None:
        """Verify lazy dictionary caching on enum-like tables (weathers, status)."""
        tokenizer_instance = PokemonTokenizer({"weathers": {"raindance": 4}, "status": {"brn": 5}})
        assert tokenizer_instance.weathers["R-a-i-n-d-a-n-c-e"] == 4
        assert tokenizer_instance.weathers["raindance"] == 4
        assert tokenizer_instance.weathers == {"raindance": 4}
        assert tokenizer_instance.weathers["unknown-weather"] == 0
        assert "unknownweather" not in tokenizer_instance.weathers
        assert tokenizer_instance.status[Status.BRN] == 5
        assert tokenizer_instance.status["brn"] == 5
        assert tokenizer_instance.status["unknown-status"] == 0
        assert tokenizer_instance.status["UNKNOWN_STATUS"] == 0
        assert "unknownstatus" not in tokenizer_instance.status
        assert tokenizer_instance.status == {"brn": 5}
        assert all(isinstance(key, str) for key in tokenizer_instance.status)

    def test_tokenizer_normalization_and_table_resolution(self) -> None:
        """Verify PokemonTokenizer normalization and resolution taxonomy: KNOWN, KNOWN_NONE, OOV, UNKNOWN."""
        assert tuple(
            PokemonTokenizer.normalize_id(value)
            for value in (
                "Charizard-Mega-Y",
                "U-turn",
                "Leech Seed",
                "  Thunderbolt  ",
                "CHARIZARD-MEGA-Y",
                "Species-1",
            )
        ) == (
            "charizardmegay",
            "uturn",
            "leechseed",
            "thunderbolt",
            "charizardmegay",
            "species1",
        )
        assert PokemonTokenizer.normalize_id(None) == ""

    def test_tokenizer_domain_objects_and_missing_values(self) -> None:
        """Verify tokenizer extracts vocabulary IDs from poke-env domain objects (Pokemon, Move, Status, PokemonType, Nature)."""
        tokenizer_instance = PokemonTokenizer(
            {
                "species": {"archaludon": 2, "charizard": 3, "pikachu": 4},
                "abilities": {"intimidate": 5},
                "items": {"choicescarf": 6},
                "types": {"fire": 7, "water": 8, "fighting": 9},
                "moves": {"closecombat": 10, "aquajet": 11},
                "categories": {"special": 14, "status": 15},
                "status": {"brn": 16, "slp": 17},
            }
        )
        assert tokenizer_instance.status_id(Status.BRN) == 16
        assert tokenizer_instance.status_id(Status.SLP) == 17
        assert tokenizer_instance.status_id(None) == 0

        p1 = Pokemon(gen=9, species="archaludon")
        assert tokenizer_instance.species_id(p1) == 2

        assert tokenizer_instance.species_id(None) == 0

        p3 = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(species="charizard", ability="intimidate"),
        )
        assert tokenizer_instance.ability_id(p3) == 5
        assert tokenizer_instance.ability_id(None) == 0

        p4 = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(species="charizard", item="choicescarf"),
        )
        assert tokenizer_instance.item_id(p4) == 6
        assert tokenizer_instance.item_id(None) == 0

        assert tokenizer_instance.type_id(PokemonType.FIRE) == 7
        assert tokenizer_instance.type_id(PokemonType.WATER) == 8
        assert tokenizer_instance.type_id(None) == 0

        m1 = Move("closecombat", 9)
        assert tokenizer_instance.move_id(m1) == 10
        m_aquajet = Move("aquajet", 9)
        assert tokenizer_instance.move_id(m_aquajet) == 11
        assert tokenizer_instance.move_id(None) == 0

        assert tokenizer_instance.move_type_id(m1) == 9
        assert tokenizer_instance.move_type_id(None) == 0

        m2 = Move("thunderbolt", 9)
        assert tokenizer_instance.move_category_id(m2) == 14

        m3 = Move("protect", 9)
        assert tokenizer_instance.move_category_id(m3) == 15
        assert tokenizer_instance.move_category_id(None) == 0
        assert tokenizer_instance.nature_id(None) == 0

        for nature, expected_id in (
            ("Serious", 0),
            ("Bashful", 0),
            ("Adamant", 1),
            ("Jolly", 12),
            ("unknown_nature", 0),
        ):
            pokemon = Pokemon(
                gen=9,
                teambuilder=TeambuilderPokemon(
                    species="pikachu", nature=nature, evs=[1, 0, 0, 0, 0, 0]
                ),
            )
            assert tokenizer_instance.nature_id(pokemon) == expected_id
