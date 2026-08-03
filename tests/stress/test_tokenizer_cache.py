from __future__ import annotations

import pytest

from p0.model.tokenizer import PokemonTokenizer, Resolution


@pytest.mark.stress
def test_normalization_cache_is_stable_across_repeated_protocol_ids() -> None:
    PokemonTokenizer._cached_normalize.cache_clear()
    values = ("Charizard-Mega-Y", "U-turn", "Leech Seed", "CHARIZARD-MEGA-Y")
    for _ in range(64):
        assert [PokemonTokenizer.normalize_id(value) for value in values] == [
            "charizardmegay",
            "uturn",
            "leechseed",
            "charizardmegay",
        ]
    info = PokemonTokenizer._cached_normalize.cache_info()
    assert info.misses == len(values)
    assert info.hits >= 64 * len(values) - len(values)


@pytest.mark.stress
def test_tokenizer_aliases_and_resolution_keep_unknown_zero_distinct_from_known_none() -> None:
    tokenizer = PokemonTokenizer(
        {
            "weathers": {"raindance": 4},
            "status": {"brn": 5},
            "moves": {"uturn": 7},
        }
    )
    assert tokenizer.id_for("moves", "U-turn") == 7
    assert tokenizer.effect_id_for("status", "status: brn") == 5
    assert tokenizer.resolve("weathers", "rain") == (4, Resolution.KNOWN)
    assert tokenizer.resolve("status", "burn") == (5, Resolution.KNOWN)
    assert tokenizer.resolve("status", "not-a-status") == (0, Resolution.OOV)
    assert tokenizer.resolve("status", None) == (0, Resolution.KNOWN_NONE)
    assert tokenizer.resolve("missing", "rain") == (0, Resolution.UNKNOWN)


@pytest.mark.stress
def test_enum_like_tables_lazy_cache_alias_and_missing_member_results() -> None:
    tokenizer = PokemonTokenizer({"weathers": {"raindance": 4}, "status": {"brn": 5}})
    assert tokenizer.weathers["rain"] == 4
    assert tokenizer.weathers["rain"] == 4
    assert tokenizer.weathers["unknown-weather"] == 0
    assert tokenizer.status["burn"] == 5
    assert tokenizer.status["unknown-status"] == 0
