"""Tests for completed-game token storage."""

import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore


class TestSeriesTokenStore:
    def test_reused_source_storage_preserves_each_perspective_snapshot(self) -> None:
        store = SeriesTokenStore(d_model=2)
        first_player = SeriesPerspectiveKey("match", 0)
        second_player = SeriesPerspectiveKey("match", 1)
        source = torch.ones(4, 2)

        store.append(first_player, source)
        source.fill_(2.0)
        store.append(second_player, source)
        source.fill_(3.0)
        tokens, mask = store.get_tokens([first_player, second_player], torch.device("cpu"))

        torch.testing.assert_close(tokens[0, :4], torch.ones(4, 2))
        torch.testing.assert_close(tokens[1, :4], torch.full((4, 2), 2.0))
        assert mask[:, :4].all()
        assert not mask[:, 4:].any()

    def test_token_store_append_and_get(self) -> None:
        """Verify SeriesTokenStore appends tokens, pads remaining slots with zeros, and computes active boolean masks."""
        store = SeriesTokenStore(d_model=16, max_games=2)

        tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        store.append("series-A", tokens1)

        out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
        assert out_tokens.shape == (1, SERIES_SLOTS, 16)
        assert out_mask.shape == (1, SERIES_SLOTS)

        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
        assert torch.all(out_tokens[0, SERIES_TOKENS_PER_GAME:] == 0)
        assert torch.all(out_mask[0, :SERIES_TOKENS_PER_GAME])
        assert not torch.any(out_mask[0, SERIES_TOKENS_PER_GAME:])

        tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        store.append("series-A", tokens2)

        out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
        assert torch.allclose(
            out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens2
        )
        assert torch.all(out_mask[0, : 2 * SERIES_TOKENS_PER_GAME])

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

    def test_token_store_training_state_round_trip(self) -> None:
        """Verify SeriesTokenStore state dictionary roundtrips losslessly into a fresh instance."""
        store = SeriesTokenStore(d_model=8)
        first = torch.randn(SERIES_TOKENS_PER_GAME, 8)
        second = torch.randn(SERIES_TOKENS_PER_GAME, 8)
        store.append("series-1", first)
        store.append("series-1", second)

        restored = SeriesTokenStore(d_model=8)
        restored.restore_training_state(store.training_state())

        tokens, mask = restored.get_tokens(["series-1"], device=torch.device("cpu"))
        assert torch.all(mask[0, : 2 * SERIES_TOKENS_PER_GAME])
        assert torch.allclose(tokens[0, :SERIES_TOKENS_PER_GAME], first)
        assert torch.allclose(
            tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], second
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
