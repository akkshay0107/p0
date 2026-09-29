from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.training.series_history import SeriesHistoryStore


class TestSeriesHistoryStore:
    def test_snapshots_are_detached_cpu_float32_copies_that_survive_drop(self) -> None:
        store = SeriesHistoryStore(d_model=2)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)

        store.append("series", 1, values, is_series_end=False)
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

    def test_series_history_store_loads_checkpoints_with_empty_active_game_fields(self) -> None:
        """Checkpoints written before active games left the store carry empty active fields."""
        restored = SeriesHistoryStore(d_model=2)
        restored.restore_training_state(
            {
                "series": {
                    "completed_games": ((1, torch.ones(1, 2)),),
                    "active_game_number": None,
                    "active_fragments": (),
                    "ended": False,
                }
            }
        )

        assert restored.next_game_number("series") == 2
        torch.testing.assert_close(restored.snapshot("series")[0], torch.ones(1, 2))

    def test_series_history_store_perspective_key_checkpoint_round_trip(self) -> None:
        """Verify SeriesPerspectiveKey is supported in checkpoint state capture and restore."""
        store = SeriesHistoryStore(d_model=2)
        key = SeriesPerspectiveKey("series_123", 1)
        store.append(
            key,
            1,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            is_series_end=False,
        )

        state = store.training_state()
        restored = SeriesHistoryStore(d_model=2)
        restored.restore_training_state(state)

        snapshot = restored.snapshot(key)
        assert len(snapshot) == 1
        torch.testing.assert_close(snapshot[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert restored.next_game_number(key) == 2
