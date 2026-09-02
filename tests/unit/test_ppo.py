"""Tests for PPO objective updates and series-context gradients."""

from __future__ import annotations

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import compute_ppo_objective, ppo_update
from p0.training.trajectory import (
    CollectedTrajectory,
    prepare_trajectory_batches,
)


class TestPPO:
    def test_ppo_recomputes_prior_game_series_context_with_live_gradients(self) -> None:
        """Verify PPO updates the live series resampler from detached prior-game histories."""
        torch.manual_seed(4)
        policy = build_policy(
            ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64),
            default_runtime_resources(),
        )
        history = torch.randn(3, policy.d_model, requires_grad=True)
        length = 2
        trajectory = CollectedTrajectory(
            observations=StructuredObservation.empty_batch(length),
            action_masks=torch.ones((length, 2, 49), dtype=torch.bool),
            actions=torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            log_probs=torch.zeros(length),
            values=torch.zeros(length),
            rewards=torch.tensor([0.0, 1.0]),
            dones=torch.ones(length),
            length=length,
            bootstrap_value=0.0,
            series_history=(history,),
        )
        prepared = prepare_trajectory_batches(
            [trajectory], torch.device("cpu"), gamma=0.99, gae_lambda=0.95
        )
        optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
        magnet = Magnet(policy)
        config = TrainingConfig(
            num_episodes=20,
            n_envs=1,
            rollout_steps=1,
            batch_size=1,
            minibatch_size=1,
            ppo_epochs=1,
            target_kl=1.0e9,
            enable_optim=False,
        )
        before = {
            name: parameter.detach().clone() for name, parameter in policy.series.named_parameters()
        }

        ppo_update(
            prepared,
            policy,
            magnet,
            optimizer,
            torch.amp.GradScaler("cpu", enabled=False),
            config,
            episode=0,
            alpha=0.0,
            cancel_requested=lambda: False,
        )

        assert all(torch.isfinite(parameter).all() for parameter in policy.series.parameters())
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in policy.series.named_parameters()
        )
        assert history.grad is None

    def test_pure_ppo_objective_uses_symmetric_clipping(self) -> None:
        """Verify conventional PPO ratio clipping and value loss."""
        config = TrainingConfig(
            clip_range=0.2,
            entropy_coef=0.0,
        )
        total, policy, value, ratio, log_ratio = compute_ppo_objective(
            torch.log(torch.tensor([2.0, 0.5])),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 0.5]),
            torch.zeros(2),
            torch.zeros(2),
            torch.ones(2),
            torch.ones(2),
            config,
            alpha=0.1,
        )
        assert total.shape == policy.shape == value.shape == ratio.shape == log_ratio.shape == (2,)
        assert ratio.tolist() == pytest.approx([2.0, 0.5])
        # For element 0: ratio=2.0, clipped to 1.2, adv=1.0 -> policy_loss = -1.2
        # value_loss = (0 - 1)^2 = 1.0 -> total = 0.5 * 1.0 - 1.2 = -0.7
        # For element 1: ratio=0.5, clipped to 0.8, adv=1.0 -> policy_loss = -0.5 (min of unclipped=0.5, clipped=0.8)
        # value_loss = (1 - 1)^2 = 0.0 -> total = -0.5 (no team preview scaling)
        assert policy[0].item() == pytest.approx(-1.2)
        assert policy[1].item() == pytest.approx(-0.5)
        assert total[0].item() == pytest.approx(config.value_coef * 1.0 - 1.2)
        assert total[1].item() == pytest.approx(-0.5)

    def test_ppo_objective_matches_simple_reference(self) -> None:
        """Verify symmetric PPO clipping, value loss, MMD penalty, and entropy bonus."""
        config = TrainingConfig(
            clip_range=0.2,
            value_coef=0.5,
            entropy_coef=0.2,
        )
        batch_size = 64
        generator = torch.Generator().manual_seed(20260807)
        current_log_probs = torch.randn(batch_size, generator=generator)
        old_log_probs = torch.randn(batch_size, generator=generator)
        advantages = torch.randn(batch_size, generator=generator)
        values = torch.randn(batch_size, generator=generator)
        returns = torch.randn(batch_size, generator=generator)
        entropy = torch.rand(batch_size, generator=generator)
        kl = torch.rand(batch_size, generator=generator)
        total, policy, value, ratio, log_ratio = compute_ppo_objective(
            current_log_probs,
            values,
            entropy,
            kl,
            old_log_probs,
            advantages,
            returns,
            config,
            alpha=0.3,
        )
        expected_ratio = torch.exp(current_log_probs - old_log_probs)
        expected_clipped = torch.clamp(
            expected_ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range,
        )
        expected_policy = -torch.minimum(
            expected_ratio * advantages,
            expected_clipped * advantages,
        )
        expected_value = (values - returns).square()
        expected_total = expected_policy + config.value_coef * expected_value + 0.3 * kl
        expected_total = expected_total - config.entropy_coef * entropy
        torch.testing.assert_close(ratio, expected_ratio)
        torch.testing.assert_close(log_ratio, current_log_probs - old_log_probs)
        torch.testing.assert_close(policy, expected_policy)
        torch.testing.assert_close(value, expected_value)
        torch.testing.assert_close(total, expected_total)
