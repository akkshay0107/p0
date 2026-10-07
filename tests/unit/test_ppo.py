"""Tests for PPO objective updates and series-context gradients."""

from __future__ import annotations

import math

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import compute_ppo_objective, magnet_kl_per_step, ppo_update
from p0.training.trajectory import (
    CollectedTrajectory,
    prepare_trajectory_batches,
)


class TestPPO:
    def test_update_honors_batch_size_without_mutating_input_order(self) -> None:
        torch.manual_seed(3)
        policy = build_policy(
            ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64),
            default_runtime_resources(),
        )
        collected = [
            CollectedTrajectory(
                observations=StructuredObservation.empty_batch(1),
                action_masks=torch.ones((1, 2, 49), dtype=torch.bool),
                actions=torch.tensor([[1, 2]], dtype=torch.long),
                log_probs=torch.zeros(1),
                values=torch.zeros(1),
                rewards=torch.tensor([float(index - 3)]),
                dones=torch.ones(1),
                length=1,
                bootstrap_value=0.0,
                series_history=(),
            )
            for index in range(7)
        ]
        prepared = prepare_trajectory_batches(
            collected, torch.device("cpu"), gamma=0.99, gae_lambda=0.95
        )
        original_order = tuple(id(trajectory) for trajectory in prepared)
        config = TrainingConfig(
            num_episodes=20,
            n_envs=1,
            rollout_steps=1,
            batch_size=3,
            minibatch_size=2,
            ppo_epochs=1,
            target_kl=1.0e9,
            enable_optim=False,
        )

        stats = ppo_update(
            prepared,
            policy,
            Magnet(policy),
            torch.optim.SGD(policy.parameters(), lr=1e-3),
            torch.amp.GradScaler("cpu", enabled=False),
            config,
            episode=0,
            alpha=0.0,
            cancel_requested=lambda: False,
        )

        assert stats["optimizer_updates"] == 3
        assert tuple(id(trajectory) for trajectory in prepared) == original_order

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

    def test_ppo_objective_weights_each_row_independently(self) -> None:
        """Verify distinct rows keep their own clipping, value, entropy and Magnet terms."""
        config = TrainingConfig(clip_range=0.2, value_coef=0.5, entropy_coef=0.1)
        total, policy, value, ratio, log_ratio = compute_ppo_objective(
            torch.log(torch.tensor([2.0, 0.5, 1.5])),
            torch.tensor([0.0, 3.0, 0.5]),
            torch.tensor([0.5, 0.25, 1.0]),
            torch.tensor([0.1, 0.2, 0.4]),
            torch.zeros(3),
            torch.tensor([1.0, 2.0, -2.0]),
            torch.tensor([1.0, 1.0, 2.0]),
            config,
            alpha=0.5,
        )

        assert ratio.tolist() == pytest.approx([2.0, 0.5, 1.5])
        assert log_ratio.tolist() == pytest.approx([math.log(2.0), math.log(0.5), math.log(1.5)])
        # Row 0 clips high (-1.2), row 1 keeps the unclipped lower bound (-1.0), and row 2 keeps
        # the unclipped pessimistic branch for a negative advantage (3.0).
        assert policy.tolist() == pytest.approx([-1.2, -1.0, 3.0])
        assert value.tolist() == pytest.approx([1.0, 4.0, 2.25])
        # policy + 0.5 * value + 0.5 * magnet_kl - 0.1 * entropy
        assert total.tolist() == pytest.approx([-0.7, 1.075, 4.225])

    def test_magnet_kl_is_exactly_zero_for_single_action_masks(self) -> None:
        """Verify -inf padding contributes no 0 * inf NaN and real divergence is still measured."""
        inf = float("inf")
        live = torch.tensor(
            [
                [[0.0, -inf, -inf], [2.0, -inf, -inf]],
                [[0.0, 0.0, -inf], [0.0, 0.0, -inf]],
            ]
        )
        magnet = torch.tensor(
            [
                [[5.0, -inf, -inf], [-1.0, -inf, -inf]],
                [[0.0, math.log(3.0), -inf], [0.0, math.log(3.0), -inf]],
            ]
        )

        kl = magnet_kl_per_step(live, magnet)

        assert kl[0].item() == 0.0
        # Each slot is 0.5 * log(0.5 / 0.25) + 0.5 * log(0.5 / 0.75) = 0.5 * log(4 / 3).
        assert kl[1].item() == pytest.approx(math.log(4.0 / 3.0))

    @pytest.mark.parametrize(
        ("ratio", "advantage", "expected_policy"),
        (
            (0.5, 2.0, -1.0),
            (0.8, 2.0, -1.6),
            (1.0, 2.0, -2.0),
            (1.2, 2.0, -2.4),
            (1.5, 2.0, -2.4),
            (0.5, -2.0, 1.6),
            (0.8, -2.0, 1.6),
            (1.0, -2.0, 2.0),
            (1.2, -2.0, 2.4),
            (1.5, -2.0, 3.0),
        ),
    )
    def test_ppo_clipping_respects_advantage_sign(
        self, ratio: float, advantage: float, expected_policy: float
    ) -> None:
        config = TrainingConfig(clip_range=0.2, value_coef=0.0, entropy_coef=0.0)
        total, policy, value, actual_ratio, log_ratio = compute_ppo_objective(
            torch.tensor([math.log(ratio)]),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.tensor([advantage]),
            torch.zeros(1),
            config,
            alpha=0.0,
        )

        assert policy.item() == pytest.approx(expected_policy)
        assert total.item() == pytest.approx(expected_policy)
        assert value.item() == 0.0
        assert actual_ratio.item() == pytest.approx(ratio)
        assert log_ratio.item() == pytest.approx(math.log(ratio))

    @pytest.mark.parametrize(
        ("value_coef", "entropy_coef", "alpha", "expected_total"),
        (
            (0.0, 0.0, 0.0, -2.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.2, 0.0, -2.1),
            (0.0, 0.0, 0.3, -1.7),
            (0.5, 0.2, 0.3, 0.2),
        ),
    )
    def test_ppo_loss_terms_have_the_expected_sign_and_weight(
        self, value_coef: float, entropy_coef: float, alpha: float, expected_total: float
    ) -> None:
        config = TrainingConfig(value_coef=value_coef, entropy_coef=entropy_coef)
        total, policy, value, _, _ = compute_ppo_objective(
            torch.zeros(1),
            torch.tensor([3.0]),
            torch.tensor([0.5]),
            torch.ones(1),
            torch.zeros(1),
            torch.tensor([2.0]),
            torch.ones(1),
            config,
            alpha=alpha,
        )

        assert policy.item() == -2.0
        assert value.item() == 4.0
        assert total.item() == pytest.approx(expected_total, abs=1e-6)
