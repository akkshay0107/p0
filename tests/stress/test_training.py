from __future__ import annotations

import pytest
import torch

from p0.training.config import TrainingConfig
from p0.training.ppo import compute_ppo_objective
from p0.training.trajectory import compute_gae_batch, prepare_trajectory_batches
from tests.stress._helpers import stress_count


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
    batch_size = stress_count("P0_STRESS_GAE_BATCH", 128)
    max_steps = stress_count("P0_STRESS_GAE_STEPS", 64)
    generator = torch.Generator().manual_seed(20260805)
    rewards = torch.randn((batch_size, max_steps), generator=generator)
    values = torch.randn((batch_size, max_steps), generator=generator)
    dones = torch.randint(0, 2, (batch_size, max_steps), generator=generator).float()
    lengths = torch.randint(1, max_steps + 1, (batch_size,), generator=generator)
    bootstraps = torch.randn(batch_size, generator=generator)
    for index, length in enumerate(lengths.tolist()):
        dones[index, length - 1] = float(index % 2 == 0)
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
    active = torch.arange(max_steps).expand(batch_size, -1) < lengths.unsqueeze(1)
    assert not actual[~active].any()


@pytest.mark.stress
def test_prepared_training_batches_keep_returns_and_normalize_only_active_steps() -> None:
    from p0.model.structured_observation import StructuredObservation
    from p0.training.trajectory import TrajectoryBatch

    generator = torch.Generator().manual_seed(20260806)
    trajectory_count = stress_count("P0_STRESS_TRAJECTORIES", 64)
    trajectories = []
    for index in range(trajectory_count):
        length = int(
            torch.randint(
                1, stress_count("P0_STRESS_TRAJECTORY_STEPS", 64) + 1, (1,), generator=generator
            ).item()
        )
        reward = torch.randn(length, generator=generator)
        dones = torch.zeros(length)
        dones[-1] = 1.0 if index % 2 == 0 else 0.0
        trajectories.append(
            TrajectoryBatch(
                observations=StructuredObservation.empty_batch(length),
                action_masks=torch.ones((length, 2, 49), dtype=torch.bool),
                actions=torch.zeros((length, 2), dtype=torch.long),
                log_probs=torch.zeros(length),
                values=torch.zeros(length),
                rewards=reward,
                dones=dones,
                length=length,
                bootstrap_value=float(torch.randn((), generator=generator)),
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
    assert sum(batch.length for batch in prepared) == sum(
        trajectory.length for trajectory in trajectories
    )


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
    batch_size = stress_count("P0_STRESS_PPO_STEPS", 4096)
    generator = torch.Generator().manual_seed(20260807)
    current_log_probs = torch.randn(batch_size, generator=generator)
    old_log_probs = torch.randn(batch_size, generator=generator)
    advantages = torch.randn(batch_size, generator=generator)
    values = torch.randn(batch_size, generator=generator)
    returns = torch.randn(batch_size, generator=generator)
    entropy = torch.rand(batch_size, generator=generator)
    kl = torch.rand(batch_size, generator=generator)
    preview = torch.rand(batch_size, generator=generator) > 0.75

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
    expected_ratio = torch.exp(current_log_probs - old_log_probs)
    expected_clipped = torch.clamp(expected_ratio, 1.0 - config.clip_low, 1.0 + config.clip_high)
    expected_policy = -torch.minimum(
        expected_ratio * advantages,
        expected_clipped * advantages,
    )
    expected_value = (values - returns).square()
    expected_total = config.value_coef * expected_value
    expected_total = (
        expected_total
        + expected_policy
        + 0.3 * torch.where(preview, config.teampreview_alpha_mult, 1.0) * kl
    )
    expected_total = expected_total - config.residual_entropy_coef * entropy
    expected_total = torch.where(
        preview,
        expected_total * config.teampreview_loss_mult,
        expected_total,
    )
    torch.testing.assert_close(ratio, expected_ratio)
    torch.testing.assert_close(log_ratio, current_log_probs - old_log_probs)
    torch.testing.assert_close(policy, expected_policy)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(total, expected_total)
