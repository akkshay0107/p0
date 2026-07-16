from typing import cast

from poke_env.battle import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from p0.model.tokenizer import PokemonTokenizer, Resolution, tokenizer


def test_tokenizer_normalization_and_table_resolution():
    assert PokemonTokenizer.normalize_id("Charizard-Mega-Y") == "charizardmegay"
    assert PokemonTokenizer.normalize_id("U-turn") == "uturn"
    assert PokemonTokenizer.normalize_id("Leech Seed") == "leechseed"
    assert PokemonTokenizer.normalize_id("  Thunderbolt  ") == "thunderbolt"
    assert PokemonTokenizer.normalize_id(None) == ""
    custom_vocab = {
        "custom_table": {
            "apple": 1,
            "banana": 2,
        }
    }
    tok = PokemonTokenizer(custom_vocab)
    assert tok.id_for("custom_table", "Apple") == 1
    assert tok.id_for("custom_table", "cherry") == 0
    assert tok.id_for("missing_table", "apple") == 0
    species = PokemonTokenizer({"species": {"pikachu": 1}})
    assert species.resolve("species", None) == (0, Resolution.KNOWN_NONE)
    assert species.resolve("species", "missingno") == (0, Resolution.OOV)
    assert species.resolve("species", "pikachu") == (1, Resolution.KNOWN)


def test_tokenizer_domain_objects_and_missing_values():
    assert tokenizer.status_id(Status.BRN) == tokenizer.status[Status.BRN]
    assert tokenizer.status_id(Status.SLP) == tokenizer.status[Status.SLP]
    assert tokenizer.status_id(None) == 0
    assert tokenizer.status_id(cast(Status, "UNKNOWN_STATUS")) == 0

    p1 = Pokemon(gen=9, species="archaludon")
    assert tokenizer.species_id(p1) == tokenizer.vocab["species"]["archaludon"]

    class FallbackPokemon(Pokemon):
        @property
        def species(self) -> str:
            return ""

        @property
        def base_species(self) -> str:
            return "charizard"

    p2 = FallbackPokemon(gen=9, species="charizard")
    assert tokenizer.species_id(p2) == tokenizer.vocab["species"]["charizard"]
    assert tokenizer.species_id(None) == 0

    p3 = Pokemon(gen=9, species="charizard")
    p3._ability = "intimidate"
    assert tokenizer.ability_id(p3) == tokenizer.vocab["abilities"]["intimidate"]
    assert tokenizer.ability_id(None) == 0

    p4 = Pokemon(gen=9, species="charizard")
    p4._item = "choicescarf"
    assert tokenizer.item_id(p4) == tokenizer.vocab["items"]["choicescarf"]
    assert tokenizer.item_id(None) == 0

    assert tokenizer.type_id(PokemonType.FIRE) == tokenizer.vocab["types"]["fire"]
    assert tokenizer.type_id(PokemonType.WATER) == tokenizer.vocab["types"]["water"]
    assert tokenizer.type_id(None) == 0

    m1 = Move("closecombat", 9)
    assert tokenizer.move_id(m1) == tokenizer.vocab["moves"]["closecombat"]
    m_aquajet = Move("aquajet", 9)
    assert tokenizer.move_id(m_aquajet) == tokenizer.vocab["moves"]["aquajet"]
    assert tokenizer.move_id(None) == 0

    assert tokenizer.move_type_id(m1) == tokenizer.vocab["types"]["fighting"]
    assert tokenizer.move_type_id(None) == 0

    m2 = Move("thunderbolt", 9)
    assert tokenizer.move_category_id(m2) == 2

    m3 = Move("protect", 9)
    assert tokenizer.move_category_id(m3) == 3
    assert tokenizer.move_category_id(None) == 0
    assert tokenizer.nature_id(None) == 0

    p = Pokemon(gen=9, species="pikachu")
    assert tokenizer.nature_id(p) == 0  # no nature set yet

    p._nature = "Serious"
    serious_id = tokenizer.nature_id(p)
    assert serious_id == 0
    p._nature = "Bashful"
    assert serious_id == tokenizer.nature_id(p)

    p._nature = "Jolly"
    jolly_id = tokenizer.nature_id(p)
    assert jolly_id > 0
    assert tokenizer.natures_list[jolly_id] == "jolly"

    p._nature = "Adamant"
    adamant_id = tokenizer.nature_id(p)
    assert adamant_id > 0
    assert tokenizer.natures_list[adamant_id] == "adamant"

    p._nature = "unknown_nature"
    assert tokenizer.nature_id(p) == 0
