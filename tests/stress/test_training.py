from __future__ import annotations

import pytest
import torch

from p0.training.config import TrainingConfig
from p0.training.ppo import compute_ppo_objective
from p0.training.trajectory import compute_gae_batch, prepare_trajectory_batches
from tests.stress._helpers import stress_repetitions


def _reference_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    bootstrap: float,
    gamma: float,
    gae_lambda: float,
) -> list[float]:
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


@pytest.mark.stress
def test_gae_matches_independent_reference_for_terminated_and_truncated_batches() -> None:
    rewards = torch.tensor([[1.0, 0.5, 0.0], [-1.0, 2.0, 0.0]])
    values = torch.tensor([[0.2, 0.3, 0.0], [0.4, 0.6, 0.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    lengths = torch.tensor([2, 2])
    bootstraps = torch.tensor([0.0, 1.5])
    gamma, gae_lambda = 0.97, 0.91

    actual = compute_gae_batch(rewards, values, dones, lengths, gamma, gae_lambda, bootstraps)
    expected = torch.tensor(
        [
            _reference_gae([1.0, 0.5], [0.2, 0.3], [0.0, 1.0], 0.0, gamma, gae_lambda),
            _reference_gae([-1.0, 2.0], [0.4, 0.6], [0.0, 0.0], 1.5, gamma, gae_lambda),
        ]
    )
    torch.testing.assert_close(actual[:, :2], expected)
    assert not actual[:, 2].any()


@pytest.mark.stress
def test_prepared_training_batches_keep_returns_and_normalize_only_active_steps() -> None:
    from p0.model.structured_observation import StructuredObservation
    from p0.training.trajectory import TrajectoryBatch

    trajectories = []
    for length, reward in ((1, 2.0), (stress_repetitions(default=4), -1.0)):
        trajectories.append(
            TrajectoryBatch(
                observations=StructuredObservation.empty_batch(length),
                action_masks=torch.ones((length, 2, 49), dtype=torch.bool),
                actions=torch.zeros((length, 2), dtype=torch.long),
                log_probs=torch.zeros(length),
                values=torch.zeros(length),
                rewards=torch.full((length,), reward),
                dones=torch.tensor([1.0] + [0.0] * (length - 1)),
                length=length,
            )
        )

    prepared = prepare_trajectory_batches(
        trajectories,
        torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
    )
    active_advantages = torch.cat(
        [batch.advantages for batch in prepared if batch.advantages is not None]
    )
    assert active_advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert active_advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)
    assert all(batch.returns is not None for batch in prepared)
    assert all(batch.length == len(batch.rewards) for batch in prepared)


@pytest.mark.stress
def test_ppo_objective_matches_reference_clipping_and_preview_weights() -> None:
    config = TrainingConfig(
        clip_low=0.2,
        clip_high=0.1,
        value_coef=0.5,
        teampreview_loss_mult=3.0,
        teampreview_alpha_mult=4.0,
        residual_entropy_coef=0.2,
    )
    current_log_probs = torch.log(torch.tensor([1.4, 0.7]))
    old_log_probs = torch.zeros(2)
    advantages = torch.tensor([1.0, -2.0])
    values = torch.tensor([0.5, -0.5])
    returns = torch.tensor([1.5, 0.5])
    entropy = torch.tensor([0.25, 0.5])
    kl = torch.tensor([0.1, 0.2])
    preview = torch.tensor([True, False])

    total, policy, value, ratio, log_ratio = compute_ppo_objective(
        current_log_probs,
        values,
        entropy,
        kl,
        old_log_probs,
        advantages,
        returns,
        preview,
        config,
        alpha=0.3,
        critic_only=False,
    )
    expected_ratio = torch.tensor([1.4, 0.7])
    expected_policy = torch.tensor([-1.1, 1.6])
    expected_value = torch.tensor([1.0, 1.0])
    expected_total = torch.tensor(
        [
            3.0 * (-1.1 + 0.3 * 4.0 * 0.1 - 0.2 * 0.25 + 0.5 * 1.0),
            1.6 + 0.3 * 0.2 - 0.2 * 0.5 + 0.5 * 1.0,
        ]
    )
    torch.testing.assert_close(ratio, expected_ratio)
    torch.testing.assert_close(log_ratio, current_log_probs)
    torch.testing.assert_close(policy, expected_policy)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(total, expected_total)
