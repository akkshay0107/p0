from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore
from p0.model.tokenizer import PokemonTokenizer
from tests.stress._helpers import stress_count, stress_repetitions, stress_rng


@pytest.mark.stress
def test_normalization_cache_is_stable_across_repeated_protocol_ids() -> None:
    PokemonTokenizer._cached_normalize.cache_clear()
    try:
        rng = stress_rng()
        values = (
            "Charizard-Mega-Y",
            "U-turn",
            "Leech Seed",
            "CHARIZARD-MEGA-Y",
            *(f"Species-{index}-{rng.randrange(1_000_000)}" for index in range(1024)),
        )
        expected = tuple(
            "".join(
                character.lower()
                for character in value
                if character.isascii() and character.isalnum()
            )
            for value in values
        )
        for _ in range(stress_repetitions(default=1024)):
            assert tuple(PokemonTokenizer.normalize_id(value) for value in values) == expected
        info = PokemonTokenizer._cached_normalize.cache_info()
        assert info.misses == len(values)
        assert info.hits >= stress_repetitions(default=1024) * len(values) - len(values)
    finally:
        PokemonTokenizer._cached_normalize.cache_clear()


@pytest.mark.stress
def test_token_store_append_drop_clear_and_high_cardinality_keys() -> None:
    store = SeriesTokenStore(d_model=3, max_games=2)
    keys = tuple(
        SeriesPerspectiveKey(f"series-{i}", i % 2)
        for i in range(stress_count("P0_STRESS_SERIES_KEYS", 1024))
    )
    game = torch.arange(SERIES_TOKENS_PER_GAME * 3, dtype=torch.float32).reshape(
        SERIES_TOKENS_PER_GAME, 3
    )
    for key in keys:
        store.append(key, game)
        store.append(key, game + 1)
        store.append(key, game + 2)
    tokens, mask = store.get_tokens(keys, torch.device("cpu"))
    assert tokens.shape[0] == len(keys)
    assert mask.sum(dim=1).tolist() == [2 * SERIES_TOKENS_PER_GAME] * len(keys)
    torch.testing.assert_close(
        tokens[:, :SERIES_TOKENS_PER_GAME],
        (game + 1).expand(len(keys), -1, -1),
    )
    torch.testing.assert_close(
        tokens[:, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME],
        (game + 2).expand(len(keys), -1, -1),
    )
    missing_tokens, missing_mask = store.get_tokens(
        (SeriesPerspectiveKey("missing-series", 0),), torch.device("cpu")
    )
    assert missing_tokens.shape == (1, SERIES_TOKENS_PER_GAME * 2, 3)
    assert not missing_mask.any()
    store.drop(keys[0])
    assert not store.get_tokens((keys[0],), torch.device("cpu"))[1].any()
    store.clear()
    assert not store._store
