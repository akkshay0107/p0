from poke_env.battle import Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from src.model.tokenizer import PokemonTokenizer, tokenizer


def test_tokenizer_normalization():
    """Verify that normalize_id handles punctuation, case-insensitivity, and None."""
    assert PokemonTokenizer.normalize_id("Charizard-Mega-Y") == "charizardmegay"
    assert PokemonTokenizer.normalize_id("U-turn") == "uturn"
    assert PokemonTokenizer.normalize_id("Leech Seed") == "leechseed"
    assert PokemonTokenizer.normalize_id("  Thunderbolt  ") == "thunderbolt"
    assert PokemonTokenizer.normalize_id(None) == ""


def test_tokenizer_id_for():
    """Verify id_for retrieves elements from custom vocab dicts or defaults to 0."""
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


def test_tokenizer_status_id():
    """Verify Status enum translation mapped to vocab values."""
    assert tokenizer.status_id(Status.BRN) == 1
    assert tokenizer.status_id(Status.SLP) == 5
    assert tokenizer.status_id(None) == 0
    # Test unrecognized/invalid status (not in vocab mapping)
    assert tokenizer.status_id("UNKNOWN_STATUS") == 0


def test_tokenizer_volatile_ids():
    """Verify mapping, sorting, deduplication, truncation, and padding of volatile IDs."""
    assert tokenizer.volatile_ids(None) == [0] * 6
    assert tokenizer.volatile_ids({}) == [0] * 6

    # CONFUSION -> 1, THROAT_CHOP -> 5, ENCORE -> 3
    # Sorted unique active: [1, 3, 5], padded: [1, 3, 5, 0, 0, 0]
    effects = {Effect.THROAT_CHOP: 1, Effect.CONFUSION: 1, Effect.ENCORE: 1}
    assert tokenizer.volatile_ids(effects) == [1, 3, 5, 0, 0, 0]

    effects_dup = {Effect.CONFUSION: 1}
    assert tokenizer.volatile_ids(effects_dup) == [1, 0, 0, 0, 0, 0]

    custom_vocab = {
        "volatiles": {
            "taunt": 1,
            "yawn": 2,
            "nightmare": 3,
            "infestation": 4,
            "flinch": 5,
            "torment": 6,
            "healblock": 7,
            "embargo": 8,
        }
    }
    custom_tok = PokemonTokenizer(custom_vocab)

    # 8 real Effect enums that normalize to the vocab keys defined above
    dummy_effects = {
        Effect.EMBARGO: 1,
        Effect.HEAL_BLOCK: 1,
        Effect.TORMENT: 1,
        Effect.FLINCH: 1,
        Effect.INFESTATION: 1,
        Effect.NIGHTMARE: 1,
        Effect.YAWN: 1,
        Effect.TAUNT: 1,
    }
    assert custom_tok.volatile_ids(dummy_effects) == [1, 2, 3, 4, 5, 6]


def test_tokenizer_pokemon_attributes():
    """Verify species, ability, item, type, and move attributes parsing using real poke_env objects."""

    p1 = Pokemon(gen=9, species="archaludon")
    assert tokenizer.species_id(p1) == 3

    class FallbackPokemon(Pokemon):
        @property
        def species(self) -> str | None:
            return None

        @property
        def base_species(self) -> str:
            return "charizard"

    p2 = FallbackPokemon(gen=9, species="charizard")
    assert tokenizer.species_id(p2) == 8
    assert tokenizer.species_id(None) == 0

    p3 = Pokemon(gen=9, species="charizard")
    p3._ability = "intimidate"
    assert tokenizer.ability_id(p3) == 14
    assert tokenizer.ability_id(None) == 0

    p4 = Pokemon(gen=9, species="charizard")
    p4._item = "choicescarf"
    assert tokenizer.item_id(p4) == 5
    assert tokenizer.item_id(None) == 0

    assert tokenizer.type_id(PokemonType.FIRE) == 2
    assert tokenizer.type_id(PokemonType.WATER) == 3
    assert tokenizer.type_id(None) == 0

    m1 = Move("closecombat", 9)
    assert tokenizer.move_id(m1) == 11
    m_aquajet = Move("aquajet", 9)
    assert tokenizer.move_id(m_aquajet) == 2
    assert tokenizer.move_id(None) == 0

    assert tokenizer.move_type_id(m1) == 7
    assert tokenizer.move_type_id(None) == 0

    m2 = Move("thunderbolt", 9)
    assert tokenizer.move_category_id(m2) == 2

    m3 = Move("protect", 9)
    assert tokenizer.move_category_id(m3) == 3
    assert tokenizer.move_category_id(None) == 0
