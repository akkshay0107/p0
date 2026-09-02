"""Tests for rollout memory and trajectory storage."""

from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.training.rollout import BattleMemoryBuffer
from p0.training.trajectory import (
    TrajectoryStorage,
)


class TestRollout:
    def test_battle_memory_keeps_the_whole_game_but_windows_the_reducer_inputs(self) -> None:
        """Verify BattleMemoryBuffer retains all game tokens for series summary while windowing reducer inputs to HISTORY_WINDOW."""
        memory = BattleMemoryBuffer(1, d_model=1)
        env_ids = torch.tensor([0])
        total = HISTORY_WINDOW + 5
        for step in range(total):
            memory.append(env_ids, torch.full((1, 1), float(step)))

        # The end-of-game series summary compresses every decision.
        assert memory.step_counts.tolist() == [total]

        history, mask = memory.inputs(env_ids, torch.device("cpu"), torch.float32)
        # Reducer input shape is restricted to HISTORY_WINDOW
        assert history.shape == (1, HISTORY_WINDOW, 1)
        assert bool(mask.all())
        assert history[0, -1, 0].item() == float(total - 1)
        assert history[0, 0, 0].item() == float(total - HISTORY_WINDOW)

    def test_battle_memory_reports_explicit_overflow(self) -> None:
        """Verify BattleMemoryBuffer raises OverflowError if step count exceeds preallocated max_steps."""
        memory = BattleMemoryBuffer(1, d_model=1, max_steps=1)
        env_ids = torch.tensor([0])
        memory.append(env_ids, torch.ones((1, 1)))

        with pytest.raises(OverflowError, match="exceeded"):
            memory.append(env_ids, torch.ones((1, 1)))

    def test_battle_memory_gathers_independent_environment_windows(self) -> None:
        """Verify BattleMemoryBuffer slices independent variable-length histories across distinct environment rows."""
        memory = BattleMemoryBuffer(2, d_model=1, max_steps=3)
        memory.append(torch.tensor([0, 1]), torch.tensor([[1.0], [10.0]]))
        memory.append(torch.tensor([1]), torch.tensor([[11.0]]))

        history, mask = memory.inputs(
            torch.tensor([0, 1]),
            torch.device("cpu"),
            torch.float32,
        )

        assert mask.sum(dim=1).tolist() == [1, 2]
        assert history[0, -1, 0].item() == 1.0
        assert history[1, -2:, 0].tolist() == [10.0, 11.0]

        memory.reset(0)
        assert memory.full_values(0) is None
        second_history = memory.full_values(1)
        assert second_history is not None
        assert second_history[0, :, 0].tolist() == [10.0, 11.0]

    def test_storage_allocates_completes_and_resets_one_environment(self) -> None:
        """Verify TrajectoryStorage allocation, completion slicing, and reset for a single environment index."""
        storage = TrajectoryStorage.allocate(2, 3)
        storage.step_counts[1] = 2
        storage.actions[1, :2] = 7
        completed = storage.complete(1, 0.0, ())
        assert completed.length == 2
        assert torch.all(completed.actions == 7)
        assert storage.step_counts.tolist() == [0, 0]

    def test_storage_reports_explicit_overflow(self) -> None:
        """Verify TrajectoryStorage raises OverflowError when environment step count exceeds capacity."""
        storage = TrajectoryStorage.allocate(1, 1)
        storage.step_counts[0] = 1
        with pytest.raises(OverflowError, match="exceeded"):
            storage.ensure_capacity(torch.tensor([0]))
