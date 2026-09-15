"""Tests for rollout memory and trajectory storage."""

from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.structured_observation import StructuredObservation
from p0.training.trajectory import (
    TrajectoryStorage,
)


class TestRollout:
    def test_storage_keeps_the_whole_game_but_windows_reducer_inputs(self) -> None:
        storage = TrajectoryStorage.allocate(1, 200, 1, "cpu")
        env_ids = torch.tensor([0])
        total = HISTORY_WINDOW + 5
        for step in range(total):
            storage.record(
                env_ids,
                StructuredObservation.empty_batch(1),
                torch.zeros((1, 2), dtype=torch.long),
                torch.zeros(1),
                torch.zeros(1),
                torch.ones((1, 2, 49), dtype=torch.bool),
                torch.full((1, 1), float(step)),
            )

        assert storage.step_counts.tolist() == [total]

        history, mask = storage.history_inputs(env_ids, torch.device("cpu"), torch.float32)
        assert history.shape == (1, HISTORY_WINDOW, 1)
        assert bool(mask.all())
        assert history[0, -1, 0].item() == float(total - 1)
        assert history[0, 0, 0].item() == float(total - HISTORY_WINDOW)

    def test_storage_gathers_independent_environment_windows(self) -> None:
        storage = TrajectoryStorage.allocate(2, 3, 1, "cpu")
        storage.record(
            torch.tensor([0, 1]),
            StructuredObservation.empty_batch(2),
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros(2),
            torch.zeros(2),
            torch.ones((2, 2, 49), dtype=torch.bool),
            torch.tensor([[1.0], [10.0]]),
        )
        storage.record(
            torch.tensor([1]),
            StructuredObservation.empty_batch(1),
            torch.zeros((1, 2), dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones((1, 2, 49), dtype=torch.bool),
            torch.tensor([[11.0]]),
        )

        history, mask = storage.history_inputs(
            torch.tensor([0, 1]),
            torch.device("cpu"),
            torch.float32,
        )

        assert mask.sum(dim=1).tolist() == [1, 2]
        assert history[0, -1, 0].item() == 1.0
        assert history[1, -2:, 0].tolist() == [10.0, 11.0]

        storage.complete(0, 0.0, ())
        assert storage.full_history(0) is None
        second_history = storage.full_history(1)
        assert second_history is not None
        assert second_history[0, :, 0].tolist() == [10.0, 11.0]

    def test_storage_allocates_completes_and_resets_one_environment(self) -> None:
        """Verify TrajectoryStorage allocation, completion slicing, and reset for a single environment index."""
        storage = TrajectoryStorage.allocate(2, 3, 1, "cpu")
        storage.step_counts[1] = 2
        storage.actions[1, :2] = 7
        completed = storage.complete(1, 0.0, ())
        assert completed.length == 2
        assert torch.all(completed.actions == 7)
        assert storage.step_counts.tolist() == [0, 0]

    def test_storage_reports_explicit_overflow(self) -> None:
        """Verify TrajectoryStorage raises OverflowError when environment step count exceeds capacity."""
        storage = TrajectoryStorage.allocate(1, 1, 1, "cpu")
        storage.step_counts[0] = 1
        with pytest.raises(OverflowError, match="exceeded"):
            storage.ensure_capacity(torch.tensor([0]))
