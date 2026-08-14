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
    """Verify that string ID normalization produces deterministic canonical strings and utilizes LRU cache.
    
    Checks that:
    1. PokemonTokenizer.normalize_id strips non-alphanumeric ASCII characters and lowercases.
    2. Repeated lookups over 1024+ unique strings hit the internal LRU cache with exactly N misses initially.
    """
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
        # Expected canonical normalization: lowercase ASCII alphanumeric characters only
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
        # Verify first pass missed on each distinct input and subsequent repetitions were pure cache hits
        assert info.misses == len(values)
        assert info.hits >= stress_repetitions(default=1024) * len(values) - len(values)
    finally:
        PokemonTokenizer._cached_normalize.cache_clear()


@pytest.mark.stress
def test_token_store_append_drop_clear_and_high_cardinality_keys() -> None:
    """Stress test SeriesTokenStore across high key volumes, FIFO eviction, and masking.
    
    Verifies that:
    1. Appending more games than `max_games=2` correctly evicts oldest entries (FIFO).
    2. Batch retrieval (`get_tokens`) returns aligned token tensors and accurate boolean attention masks.
    3. Missing or dropped series keys return empty zero-padded tensors with all-False mask bits.
    4. `drop` and `clear` cleanly deallocate entries without memory leaks.
    """
    store = SeriesTokenStore(d_model=3, max_games=2)
    keys = tuple(
        SeriesPerspectiveKey(f"series-{i}", i % 2)
        for i in range(stress_count("P0_STRESS_SERIES_KEYS", 1024))
    )
    # Synthetic single-game token representation of shape (SERIES_TOKENS_PER_GAME, d_model)
    game = torch.arange(SERIES_TOKENS_PER_GAME * 3, dtype=torch.float32).reshape(
        SERIES_TOKENS_PER_GAME, 3
    )
    # Append 3 successive games (game, game+1, game+2) to test max_games=2 FIFO capacity limit
    for key in keys:
        store.append(key, game)
        store.append(key, game + 1)
        store.append(key, game + 2)
    tokens, mask = store.get_tokens(keys, torch.device("cpu"))
    assert tokens.shape[0] == len(keys)
    # Valid mask count must be 2 games * tokens_per_game per key
    assert mask.sum(dim=1).tolist() == [2 * SERIES_TOKENS_PER_GAME] * len(keys)
    # Verify the oldest game was evicted and only (game+1) and (game+2) remain in slots
    torch.testing.assert_close(
        tokens[:, :SERIES_TOKENS_PER_GAME],
        (game + 1).expand(len(keys), -1, -1),
    )
    torch.testing.assert_close(
        tokens[:, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME],
        (game + 2).expand(len(keys), -1, -1),
    )
    # Validate missing key behavior: returned tensor is valid shape but attention mask is entirely False
    missing_tokens, missing_mask = store.get_tokens(
        (SeriesPerspectiveKey("missing-series", 0),), torch.device("cpu")
    )
    assert missing_tokens.shape == (1, SERIES_TOKENS_PER_GAME * 2, 3)
    assert not missing_mask.any()
    # Validate explicit key eviction and full store purge
    store.drop(keys[0])
    assert not store.get_tokens((keys[0],), torch.device("cpu"))[1].any()
    store.clear()
    assert not store._store
