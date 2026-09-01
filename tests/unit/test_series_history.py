from __future__ import annotations

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.training.series_history import SeriesHistoryStore


def test_series_history_store_commits_fragments_and_keeps_snapshots_after_drop() -> None:
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


def test_series_history_store_isolates_perspectives_and_truncates_old_games() -> None:
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


def test_series_history_store_rejects_invalid_chronology_and_series_boundaries() -> None:
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


def test_series_history_store_checkpoint_round_trip_preserves_active_state() -> None:
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
