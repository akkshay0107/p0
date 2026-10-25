"""Tests for completed-game token storage."""

import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore


class TestSeriesTokenStore:
    def test_append_keeps_independent_game_and_perspective_snapshots(self) -> None:
        store = SeriesTokenStore(d_model=2, max_games=2)
        keys = (SeriesPerspectiveKey("match", 0), SeriesPerspectiveKey("match", 1))
        source = torch.ones(SERIES_TOKENS_PER_GAME, 2)
        store.append(keys[0], source)
        source.fill_(2.0)
        store.append(keys[1], source)
        source.fill_(3.0)

        tokens, mask = store.get_tokens(keys, torch.device("cpu"))

        assert tokens.shape == (2, SERIES_SLOTS, 2)
        assert mask.shape == (2, SERIES_SLOTS)
        torch.testing.assert_close(tokens[0, :SERIES_TOKENS_PER_GAME], torch.ones_like(source))
        torch.testing.assert_close(tokens[1, :SERIES_TOKENS_PER_GAME], torch.full_like(source, 2.0))
        assert not tokens[:, SERIES_TOKENS_PER_GAME:].any()
        assert mask[:, :SERIES_TOKENS_PER_GAME].all()
        assert not mask[:, SERIES_TOKENS_PER_GAME:].any()

        store.append(keys[0], source)
        source.fill_(4.0)
        tokens, mask = store.get_tokens(keys, torch.device("cpu"))

        torch.testing.assert_close(tokens[0, :SERIES_TOKENS_PER_GAME], torch.ones_like(source))
        torch.testing.assert_close(tokens[0, SERIES_TOKENS_PER_GAME:], torch.full_like(source, 3.0))
        torch.testing.assert_close(tokens[1, :SERIES_TOKENS_PER_GAME], torch.full_like(source, 2.0))
        assert not tokens[1, SERIES_TOKENS_PER_GAME:].any()
        assert mask[0].all()
        assert mask[1, :SERIES_TOKENS_PER_GAME].all()
        assert not mask[1, SERIES_TOKENS_PER_GAME:].any()

    def test_token_store_max_games_truncation(self) -> None:
        """Verify SeriesTokenStore retains only the most recent max_games=2 entries using FIFO eviction."""
        store = SeriesTokenStore(d_model=16, max_games=2)

        tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        tokens3 = torch.randn(SERIES_TOKENS_PER_GAME, 16)

        store.append("series-B", tokens1)
        store.append("series-B", tokens2)
        store.append("series-B", tokens3)

        out_tokens, out_mask = store.get_tokens(["series-B"], device=torch.device("cpu"))

        # Oldest tokens (tokens1) must be evicted; tokens2 and tokens3 retained
        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens2)
        assert torch.allclose(
            out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens3
        )

    def test_token_store_drop_and_clear(self) -> None:
        """Verify dropping one key preserves another until clear removes both stored values."""
        store = SeriesTokenStore(d_model=8)
        first = torch.full((SERIES_TOKENS_PER_GAME, 8), 2.0)
        second = torch.full((SERIES_TOKENS_PER_GAME, 8), 9.0)
        store.append("s1", first)
        store.append("s2", second)

        before_tokens, before_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
        store.drop("s1")

        dropped_tokens, dropped_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
        assert not dropped_mask[0].any()
        assert not dropped_tokens[0].any()
        assert torch.equal(dropped_tokens[1], before_tokens[1])
        assert torch.equal(dropped_mask[1], before_mask[1])
        assert dropped_mask[1, :SERIES_TOKENS_PER_GAME].all()
        assert torch.equal(
            dropped_tokens[1, :SERIES_TOKENS_PER_GAME],
            torch.full((SERIES_TOKENS_PER_GAME, 8), 9.0),
        )

        store.clear()
        cleared_tokens, cleared_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
        assert not cleared_mask.any()
        assert not cleared_tokens.any()
