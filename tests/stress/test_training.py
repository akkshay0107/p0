from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.model.structured_observation import StructuredObservation
from p0.training.trajectory import (
    CollectedTrajectory,
    compute_gae_batch,
    prepare_trajectory_batches,
)
from tests.stress._helpers import stress_count

_FINITE_FLOAT = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _reference_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[float],
    bootstrap: float,
    gamma: float,
    gae_lambda: float,
) -> list[float]:
    """Pure-Python reference implementation of Generalized Advantage Estimation."""
    next_value = bootstrap
    gae = 0.0
    output = [0.0] * len(rewards)
    for index in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        output[index] = gae
        next_value = values[index]
    return output


@st.composite
def _gae_cases(draw: st.DrawFn) -> tuple[torch.Tensor, ...]:
    batch_size = draw(st.integers(min_value=1, max_value=stress_count("P0_STRESS_GAE_BATCH", 128)))
    max_steps = draw(st.integers(min_value=1, max_value=stress_count("P0_STRESS_GAE_STEPS", 64)))
    tensor_size = batch_size * max_steps
    rewards = torch.tensor(
        draw(st.lists(_FINITE_FLOAT, min_size=tensor_size, max_size=tensor_size)),
        dtype=torch.float32,
    ).reshape(batch_size, max_steps)
    values = torch.tensor(
        draw(st.lists(_FINITE_FLOAT, min_size=tensor_size, max_size=tensor_size)),
        dtype=torch.float32,
    ).reshape(batch_size, max_steps)
    dones = torch.tensor(
        draw(st.lists(st.booleans(), min_size=tensor_size, max_size=tensor_size)),
        dtype=torch.float32,
    ).reshape(batch_size, max_steps)
    lengths = torch.tensor(
        draw(st.lists(st.integers(1, max_steps), min_size=batch_size, max_size=batch_size)),
        dtype=torch.long,
    )
    terminal = draw(st.lists(st.booleans(), min_size=batch_size, max_size=batch_size))
    for index, length in enumerate(lengths.tolist()):
        dones[index, length - 1] = float(terminal[index])
    bootstraps = torch.tensor(
        draw(st.lists(_FINITE_FLOAT, min_size=batch_size, max_size=batch_size)),
        dtype=torch.float32,
    )
    return rewards, values, dones, lengths, bootstraps


@st.composite
def _prepared_cases(
    draw: st.DrawFn,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    max_steps = draw(
        st.integers(min_value=2, max_value=stress_count("P0_STRESS_TRAJECTORY_STEPS", 64))
    )
    trajectory_count = draw(
        st.integers(min_value=1, max_value=stress_count("P0_STRESS_TRAJECTORIES", 64))
    )
    lengths = [draw(st.integers(min_value=2, max_value=max_steps))]
    lengths.extend(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=max_steps),
                min_size=trajectory_count - 1,
                max_size=trajectory_count - 1,
            )
        )
    )
    terminal_flags = draw(
        st.lists(st.booleans(), min_size=trajectory_count, max_size=trajectory_count)
    )
    bootstraps = torch.tensor(
        draw(st.lists(_FINITE_FLOAT, min_size=trajectory_count, max_size=trajectory_count)),
        dtype=torch.float32,
    )
    rewards = [torch.arange(1, length + 1, dtype=torch.float32) for length in lengths]
    values = [torch.zeros(length, dtype=torch.float32) for length in lengths]
    dones = []
    for length, terminal in zip(lengths, terminal_flags, strict=True):
        done = torch.zeros(length, dtype=torch.float32)
        done[-1] = float(terminal)
        dones.append(done)
    return rewards, values, dones, bootstraps


@pytest.mark.stress
@settings(max_examples=32, deadline=None)
@given(case=_gae_cases())
def test_gae_matches_independent_reference_for_generated_batches(
    case: tuple[torch.Tensor, ...],
) -> None:
    """Check vectorized GAE across generated lengths, terminals, and truncations."""
    rewards, values, dones, lengths, bootstraps = case
    gamma, gae_lambda = 0.97, 0.91

    actual = compute_gae_batch(rewards, values, dones, lengths, gamma, gae_lambda, bootstraps)
    expected = torch.zeros_like(actual)
    for index, length in enumerate(lengths.tolist()):
        expected[index, :length] = torch.tensor(
            _reference_gae(
                rewards[index, :length].tolist(),
                values[index, :length].tolist(),
                dones[index, :length].tolist(),
                float(bootstraps[index]),
                gamma,
                gae_lambda,
            )
        )

    torch.testing.assert_close(actual, expected)
    active = torch.arange(rewards.size(1)).expand(rewards.size(0), -1) < lengths.unsqueeze(1)
    assert not actual[~active].any()


@pytest.mark.stress
@settings(max_examples=32, deadline=None)
@given(case=_prepared_cases())
def test_prepared_training_batches_keep_returns_and_normalize_active_steps(
    case: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor],
) -> None:
    """Check global advantage normalization while preserving trajectory-local returns."""
    rewards, values, dones, bootstraps = case
    trajectories = [
        CollectedTrajectory(
            observations=StructuredObservation.empty_batch(reward.numel()),
            action_masks=torch.ones((reward.numel(), 2, 49), dtype=torch.bool),
            actions=torch.zeros((reward.numel(), 2), dtype=torch.long),
            log_probs=torch.zeros(reward.numel()),
            values=value,
            rewards=reward,
            dones=done,
            length=reward.numel(),
            bootstrap_value=float(bootstrap),
            series_history=(),
        )
        for reward, value, done, bootstrap in zip(rewards, values, dones, bootstraps, strict=True)
    ]

    gamma, gae_lambda = 0.99, 0.95
    raw_advantages = [
        torch.tensor(
            _reference_gae(
                reward.tolist(),
                value.tolist(),
                done.tolist(),
                trajectory.bootstrap_value,
                gamma,
                gae_lambda,
            )
        )
        for trajectory, reward, value, done in zip(
            trajectories, rewards, values, dones, strict=True
        )
    ]
    raw_flat = torch.cat(raw_advantages)
    mean = raw_flat.mean()
    std = raw_flat.std(unbiased=False).clamp_min(1e-8)

    prepared = prepare_trajectory_batches(
        trajectories,
        torch.device("cpu"),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    actual_advantages = torch.cat([batch.advantages for batch in prepared])
    torch.testing.assert_close(actual_advantages, (raw_flat - mean) / std)
    for batch, raw_advantage, value in zip(prepared, raw_advantages, values, strict=True):
        torch.testing.assert_close(batch.returns, raw_advantage + value)
        assert batch.length == len(batch.rewards)
    assert sum(batch.length for batch in prepared) == sum(
        trajectory.length for trajectory in trajectories
    )
