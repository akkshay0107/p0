from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.training.series_history import (
    SeriesHistoryStore,
    advance_series_state,
    pack_series_context,
)


class TestSeriesHistoryStore:
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
    def test_snapshots_are_detached_cpu_float32_copies_that_survive_drop(
        self, dtype: torch.dtype
    ) -> None:
        store = SeriesHistoryStore(d_model=2)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, requires_grad=True)

        store.append("series", 1, values, is_series_end=False)
        with torch.no_grad():
            values[0, 0] = 99.0
        (game,) = store.snapshot("series")
        store.drop("series")

        assert store.snapshot("series") == ()
        assert game.dtype is torch.float32
        assert game.device.type == "cpu"
        assert game.requires_grad is False
        torch.testing.assert_close(game, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_series_history_store_isolates_perspectives_and_truncates_old_games(self) -> None:
        store = SeriesHistoryStore(d_model=2, max_games=2)
        key = SeriesPerspectiveKey("series", 0)
        other_key = SeriesPerspectiveKey("series", 1)

        for number in (1, 2, 3):
            store.append(
                key,
                number,
                torch.full((number, 2), float(number)),
                is_series_end=False,
            )
        store.append(other_key, 1, torch.full((1, 2), 9.0), is_series_end=False)

        snapshot = store.snapshot(key)
        assert [item.size(0) for item in snapshot] == [2, 3]
        assert store.snapshot(other_key)[0][0, 0].item() == 9.0
        assert store.next_game_number(key) == 4

    def test_series_history_store_rejects_invalid_chronology_and_data_after_series_end(
        self,
    ) -> None:
        store = SeriesHistoryStore(d_model=2)
        values = torch.ones(1, 2)

        store.append("series", 1, values, is_series_end=False)
        with pytest.raises(ValueError, match="consecutive"):
            store.append("series", 1, values, is_series_end=False)
        with pytest.raises(ValueError, match="consecutive"):
            store.append("series", 3, values, is_series_end=False)

        store.append("series", 2, values, is_series_end=True)
        with pytest.raises(ValueError, match="after it ended"):
            store.append("series", 3, values, is_series_end=False)


class TestAdvanceSeriesState:
    def test_rejects_a_game_change_before_the_active_game_ends(self) -> None:
        fragment = torch.ones(1, 2)
        active_game_one = ((), 1, (fragment,), False)

        with pytest.raises(ValueError, match="changed"):
            advance_series_state(
                active_game_one, 2, fragment, is_game_end=True, is_series_end=False
            )

        completed, active_game, fragments, ended = advance_series_state(
            active_game_one, 1, fragment, is_game_end=True, is_series_end=False
        )
        assert [number for number, _ in completed] == [1]
        assert active_game is None
        assert fragments == ()
        assert ended is False


class TestPackSeriesContext:
    def test_shared_game_is_summarized_once_and_repeated_over_each_group(self) -> None:
        short = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        long = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 5.0]])
        resampled_games: list[int] = []

        def sum_real_rows(padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            resampled_games.append(padded.size(0))
            total = (padded * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            return total.expand(-1, SERIES_TOKENS_PER_GAME, -1)

        # Three groups of 2, 1 and 1 rows: two prior games, one shared game, none.
        tokens, mask = pack_series_context(
            [(short, long), (short,), ()],
            torch.tensor([2, 1, 1]),
            sum_real_rows,
            torch.zeros(4, 2),
        )

        assert resampled_games == [2]
        assert tokens.shape == (4, 2 * SERIES_TOKENS_PER_GAME, 2)
        short_sum, long_sum = torch.tensor([4.0, 6.0]), torch.tensor([3.0, 5.0])
        for row in (0, 1):
            assert mask[row].all()
            assert (tokens[row, :SERIES_TOKENS_PER_GAME] == short_sum).all()
            assert (tokens[row, SERIES_TOKENS_PER_GAME:] == long_sum).all()
        assert (
            mask[2].tolist() == [True] * SERIES_TOKENS_PER_GAME + [False] * SERIES_TOKENS_PER_GAME
        )
        assert (tokens[2, :SERIES_TOKENS_PER_GAME] == short_sum).all()
        assert not tokens[2, SERIES_TOKENS_PER_GAME:].any()
        assert not mask[3].any() and not tokens[3].any()
