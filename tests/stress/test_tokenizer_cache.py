from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore
from p0.model.tokenizer import PokemonTokenizer


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
def test_token_store_append_drop_clear_and_high_cardinality_keys() -> None:
    store = SeriesTokenStore(d_model=3, max_games=2)
    keys = tuple(SeriesPerspectiveKey(f"series-{i}", i % 2) for i in range(32))
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
    store.drop(keys[0])
    assert not store.get_tokens((keys[0],), torch.device("cpu"))[1].any()
    store.clear()
    assert not store._store
