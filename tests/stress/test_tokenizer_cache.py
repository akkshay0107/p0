from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore
from tests.stress._helpers import stress_count


class TestTokenizerCache:
    @pytest.mark.stress
    def test_token_store_append_drop_clear_and_high_cardinality_keys(self) -> None:
        """
        Stress test SeriesTokenStore across high key volumes, FIFO eviction, and masking.

        Verifies that:
        1. Appending more games than max_games=2 correctly evicts oldest entries (FIFO).
        2. Batch retrieval (get_tokens) returns aligned token tensors and accurate boolean attention masks.
        3. Missing or dropped series keys return empty zero-padded tensors with all-False mask bits.
        4. drop and clear cleanly deallocate entries without memory leaks.
        """
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
        _, cleared_mask = store.get_tokens(keys[:1], torch.device("cpu"))
        assert not cleared_mask.any()
