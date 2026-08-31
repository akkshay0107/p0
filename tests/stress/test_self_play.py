from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.training.rollout import BattleMemoryBuffer
from tests.stress._helpers import stress_repetitions


@pytest.mark.stress
def test_battle_memory_window_is_bounded_and_reset_is_local() -> None:
    """
    Stress test recurrent BattleMemoryBuffer window slicing, overflow handling, and env isolation.

    Verifies that:
    1. BattleMemoryBuffer inputs() returns the latest HISTORY_WINDOW steps.
    2. Appending beyond max_steps raises OverflowError.
    3. Resetting a specific env slot purges only that environment's history, leaving other slots intact.
    """
    repetitions = max(HISTORY_WINDOW, stress_repetitions(default=2048))
    buffer = BattleMemoryBuffer(2, d_model=3, max_steps=repetitions)
    # Fill memory buffer with sequential index values across both environments
    for index in range(repetitions):
        buffer.append(torch.tensor([0, 1]), torch.full((2, 3), float(index)))

    history, mask, ages = buffer.inputs(torch.tensor([0, 1]), torch.device("cpu"), torch.float32)
    assert history.shape[0] == mask.shape[0] == ages.shape[0] == 2
    assert mask[0].sum().item() == mask[1].sum().item()
    assert torch.equal(history[0], history[1])
    # The history slice must correspond strictly to the most recent HISTORY_WINDOW turns
    assert torch.equal(
        history[0, -HISTORY_WINDOW:, 0],
        torch.arange(repetitions - HISTORY_WINDOW, repetitions, dtype=torch.float32),
    )

    # Exceeding allocated buffer capacity must raise an explicit OverflowError
    with pytest.raises(OverflowError, match="exceeded"):
        buffer.append(torch.tensor([0, 1]), torch.full((2, 3), float(repetitions)))

    # Reset environment 0 and verify environment 1 is unaffected
    buffer.reset(0)
    empty_history, empty_mask, empty_ages = buffer.inputs(
        torch.tensor([0]), torch.device("cpu"), torch.float32
    )
    assert not empty_mask.any()
    assert not empty_history.any()
    assert not empty_ages.any()
    assert mask[1].any()
