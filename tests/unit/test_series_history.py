from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.training._bc_batch import BCGameWindow
from p0.training._bc_history import prepare_series_context, window_history_tokens
from p0.training.series_history import SeriesHistoryStore


class TestSeriesHistory:
    @pytest.mark.parametrize(
        "device",
        ["cpu", "meta", *(["cuda"] if torch.cuda.is_available() else [])],
    )
    def test_active_cpu_fragment_joins_current_batch_device(self, device: str) -> None:
        key = SeriesPerspectiveKey("fragmented-series", 0)
        store = SeriesHistoryStore(d_model=2)
        saved = torch.tensor([[1.0, 2.0]])
        store.append(key, 1, saved, is_game_end=False, is_series_end=False)
        current = torch.ones((2, 2), device=device, requires_grad=True)
        windows = (
            BCGameWindow(key, 1, 0, 1, True, False, 1),
            BCGameWindow(key, 2, 1, 2, False, False, 2),
        )

        context, mask = prepare_series_context(
            store,
            windows,
            (current[0:1], current[1:2]),
            current,
            lambda values, present: values[:, 1:2].repeat(1, SERIES_TOKENS_PER_GAME, 1),
        )

        assert context.device.type == device
        assert mask.device.type == device
        assert store.planning_state(key)[2][0].device.type == "cpu"
        if device == "cpu":
            context[1].sum().backward()
            assert current.grad is not None
            assert current.grad[0].abs().sum() > 0
            assert current.grad[1].abs().sum() == 0

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

    def test_series_history_store_commits_fragments_and_keeps_snapshots_after_drop(self) -> None:
        store = SeriesHistoryStore(d_model=3)
        first = torch.ones(2, 3, requires_grad=True)
        second = torch.full((1, 3), 2.0, requires_grad=True)

        store.append("series", 1, first, is_game_end=False, is_series_end=False)
        assert store.snapshot("series") == ()
        store.append("series", 1, second, is_game_end=True, is_series_end=False)

        snapshot = store.snapshot("series")
        assert len(snapshot) == 1
        assert snapshot[0].shape == (3, 3)
        assert snapshot[0].dtype is torch.float32
        assert snapshot[0].device.type == "cpu"
        assert snapshot[0].requires_grad is False

        store.drop("series")
        assert store.snapshot("series") == ()
        torch.testing.assert_close(snapshot[0], torch.cat((first, second)))

    def test_series_history_store_isolates_perspectives_and_truncates_old_games(self) -> None:
        store = SeriesHistoryStore(d_model=2, max_games=2)
        key = SeriesPerspectiveKey("series", 0)
        other_key = SeriesPerspectiveKey("series", 1)

        for number in (1, 2, 3):
            store.append(
                key,
                number,
                torch.full((number, 2), float(number)),
                is_game_end=True,
                is_series_end=False,
            )
        store.append(
            other_key,
            1,
            torch.full((1, 2), 9.0),
            is_game_end=True,
            is_series_end=False,
        )

        snapshot = store.snapshot(key)
        assert [item.size(0) for item in snapshot] == [2, 3]
        assert store.snapshot(other_key)[0][0, 0].item() == 9.0
        assert store.next_game_number(key) == 4

    def test_series_history_store_rejects_invalid_chronology_and_series_boundaries(self) -> None:
        store = SeriesHistoryStore(d_model=2)
        values = torch.ones(1, 2)

        with pytest.raises(ValueError, match="only at a game boundary"):
            store.append("series", 1, values, is_game_end=False, is_series_end=True)

        store.append("series", 1, values, is_game_end=False, is_series_end=False)
        with pytest.raises(ValueError, match="changed"):
            store.append("series", 2, values, is_game_end=True, is_series_end=False)
        store.append("series", 1, values, is_game_end=True, is_series_end=False)
        with pytest.raises(ValueError, match="consecutive"):
            store.append("series", 1, values, is_game_end=True, is_series_end=False)
        with pytest.raises(ValueError, match="consecutive"):
            store.append("series", 3, values, is_game_end=True, is_series_end=False)

        store.append("series", 2, values, is_game_end=False, is_series_end=False)
        store.discard_active_games()
        assert not store.has_partial_games
        assert store.next_game_number("series") == 2
        assert len(store.snapshot("series")) == 1

    def test_series_history_store_checkpoint_round_trip_preserves_active_state(self) -> None:
        store = SeriesHistoryStore(d_model=2)
        store.append(
            "series",
            1,
            torch.arange(4, dtype=torch.float64).reshape(2, 2),
            is_game_end=False,
            is_series_end=False,
        )

        restored = SeriesHistoryStore(d_model=2)
        restored.restore_training_state(store.training_state())

        assert restored.has_partial_games
        assert restored.next_game_number("series") == 1
        state = restored.training_state()["series"]
        assert state["active_game_number"] == 1
        fragments = state["active_fragments"]
        assert isinstance(fragments, tuple)
        torch.testing.assert_close(fragments[0], torch.arange(4, dtype=torch.float32).reshape(2, 2))

    def test_series_history_store_perspective_key_checkpoint_round_trip(self) -> None:
        """Verify SeriesPerspectiveKey is supported in checkpoint state capture and restore."""
        store = SeriesHistoryStore(d_model=2)
        key = SeriesPerspectiveKey("series_123", 1)
        store.append(
            key,
            1,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            is_game_end=True,
            is_series_end=False,
        )
        store.append(
            key,
            2,
            torch.tensor([[5.0, 6.0]]),
            is_game_end=False,
            is_series_end=False,
        )

        state = store.training_state()
        restored = SeriesHistoryStore(d_model=2)
        restored.restore_training_state(state)

        snapshot = restored.snapshot(key)
        assert len(snapshot) == 1
        torch.testing.assert_close(snapshot[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert restored.next_game_number(key) == 2

    def test_selected_key_snapshot_restores_existing_and_absent_states(self) -> None:
        store = SeriesHistoryStore(d_model=2)
        values = torch.ones(1, 2)
        store.append("existing", 1, values, is_game_end=True, is_series_end=False)
        rollback = store.snapshot_keys(("existing", "new", "existing"))

        store.append("existing", 2, values, is_game_end=True, is_series_end=True)
        store.append("new", 1, values, is_game_end=True, is_series_end=True)
        store.restore_keys(rollback)

        assert len(store.snapshot("existing")) == 1
        assert store.next_game_number("existing") == 2
        store.append("existing", 2, values, is_game_end=True, is_series_end=True)
        store.append("new", 1, values, is_game_end=True, is_series_end=True)
