from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.training.trajectory import TrajectoryStorage
from tests.stress._helpers import stress_repetitions


class TestSelfPlay:
    @pytest.mark.stress
    def test_battle_memory_window_is_bounded_and_reset_is_local(self) -> None:
        """Stress bounded history gathering, overflow, and environment isolation."""
        repetitions = max(HISTORY_WINDOW, stress_repetitions(default=2048))
        retained_steps = HISTORY_WINDOW + 5
        storage = TrajectoryStorage.allocate(2, retained_steps, 3, "cpu")
        values = torch.arange(retained_steps, dtype=torch.float32)
        storage.history_tokens[:] = values[None, :, None]
        storage.step_counts[:] = retained_steps

        history, mask = storage.history_inputs(
            torch.tensor([0, 1]), torch.device("cpu"), torch.float32
        )
        for _ in range(repetitions - 1):
            history, mask = storage.history_inputs(
                torch.tensor([0, 1]), torch.device("cpu"), torch.float32
            )
        assert history.shape[0] == mask.shape[0] == 2
        assert mask[0].sum().item() == mask[1].sum().item()
        assert torch.equal(history[0], history[1])
        assert torch.equal(
            history[0, -HISTORY_WINDOW:, 0],
            torch.arange(retained_steps - HISTORY_WINDOW, retained_steps, dtype=torch.float32),
        )

        with pytest.raises(OverflowError, match="exceeded"):
            storage.ensure_capacity(torch.tensor([0, 1]))

        storage.complete(0, 0.0, ())
        empty_history, empty_mask = storage.history_inputs(
            torch.tensor([0]), torch.device("cpu"), torch.float32
        )
        assert not empty_mask.any()
        assert not empty_history.any()
        assert mask[1].any()
