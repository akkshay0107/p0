from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.training._bc_batch import BCGameWindow
from p0.training._bc_history import (
    ActiveGames,
    commit_history_updates,
    prepare_series_context,
    window_history_tokens,
)
from p0.training.series_history import SeriesHistoryStore


class TestPrepareSeriesContext:
    @pytest.mark.parametrize(
        "device",
        ["cpu", "meta", *(["cuda"] if torch.cuda.is_available() else [])],
    )
    def test_active_cpu_fragment_joins_current_batch_device(self, device: str) -> None:
        key = SeriesPerspectiveKey("fragmented-series", 0)
        store = SeriesHistoryStore(d_model=2)
        active: ActiveGames = {key: (1, (torch.tensor([[1.0, 2.0]]),))}
        current = torch.ones((2, 2), device=device, requires_grad=True)
        windows = (
            BCGameWindow(key, 1, 0, 1, True, False, 1),
            BCGameWindow(key, 2, 1, 2, False, False, 2),
        )

        context, mask = prepare_series_context(
            store,
            active,
            windows,
            (current[0:1], current[1:2]),
            current,
            lambda values, present: values[:, 1:2].repeat(1, SERIES_TOKENS_PER_GAME, 1),
        )

        assert context.device.type == device
        assert mask.device.type == device
        assert active[key][1][0].device.type == "cpu"
        if device == "cpu":
            context[1].sum().backward()
            assert current.grad is not None
            assert current.grad[0].abs().sum() > 0
            assert current.grad[1].abs().sum() == 0

    def test_rejects_a_game_change_before_the_active_game_ends(self) -> None:
        key = SeriesPerspectiveKey("series", 0)
        current = torch.ones((1, 2))

        with pytest.raises(ValueError, match="changed"):
            prepare_series_context(
                SeriesHistoryStore(d_model=2),
                {key: (1, (torch.ones(1, 2),))},
                (BCGameWindow(key, 2, 0, 1, True, False, 1),),
                (current,),
                current,
                lambda values, present: values,
            )


class TestWindowHistoryTokens:
    def test_only_a_game_ending_window_adds_the_final_board_summary(self) -> None:
        key = SeriesPerspectiveKey("final-board", 0)
        # Context rows: game 1 targets 0-1, its final board 2, then game 2's target 3.
        context = torch.arange(8.0).reshape(4, 2)
        targets = context[[0, 1, 3]]
        windows = (
            BCGameWindow(key, 1, 0, 2, True, False, 2),
            BCGameWindow(key, 2, 2, 3, False, False, 4),
        )

        first, second = window_history_tokens(windows, targets, context)

        torch.testing.assert_close(first, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
        torch.testing.assert_close(second, torch.tensor([[6.0, 7.0]]))


class TestCommitHistoryUpdates:
    def test_fragments_of_a_split_game_join_into_one_completed_game(self) -> None:
        key = SeriesPerspectiveKey("series", 0)
        store = SeriesHistoryStore(d_model=3)
        active: ActiveGames = {}
        first = torch.ones(2, 3, requires_grad=True)
        second = torch.full((1, 3), 2.0, requires_grad=True)

        commit_history_updates(
            store, active, (BCGameWindow(key, 1, 0, 2, False, False, 2),), (first,)
        )
        assert store.snapshot(key) == ()
        assert active[key][0] == 1
        commit_history_updates(
            store, active, (BCGameWindow(key, 1, 0, 1, True, False, 1),), (second,)
        )

        assert active == {}
        (game,) = store.snapshot(key)
        assert game.requires_grad is False
        torch.testing.assert_close(game, torch.tensor([[1.0] * 3, [1.0] * 3, [2.0] * 3]))
