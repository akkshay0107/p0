"""Tests for trajectory advantage and batch preparation."""

from __future__ import annotations

import torch

from p0.model.structured_observation import StructuredObservation
from p0.training.trajectory import (
    CollectedTrajectory,
    compute_gae_batch,
    prepare_trajectory_batches,
)


class TestTrajectory:
    def test_compute_gae_batch_matches_single_episode_reference(self) -> None:
        """Verify vectorized compute_gae_batch matches pure-Python sequential GAE calculation for variable episode lengths."""

        def compute_gae_reference(
            rewards: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
            gamma: float,
            gae_lambda: float,
        ) -> torch.Tensor:
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(rewards.size(0))):
                next_value = values[t + 1] if t + 1 < rewards.size(0) else 0.0
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value * nonterminal - values[t]
                gae = delta + gamma * gae_lambda * nonterminal * gae
                advantages[t] = gae
            return advantages

        rewards = [
            torch.tensor([1.0, 0.5, -0.25, 2.0]),
            torch.tensor([0.25, 0.75]),
            torch.tensor([-1.0, 0.0, 1.0]),
        ]
        values = [
            torch.tensor([0.2, 0.3, 0.4, 0.5]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.5, 0.25, -0.1]),
        ]
        dones = [
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ]
        lengths = torch.tensor([len(row) for row in rewards])
        rewards_padded = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True)
        values_padded = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        dones_padded = torch.nn.utils.rnn.pad_sequence(dones, batch_first=True)

        actual = compute_gae_batch(
            rewards_padded,
            values_padded,
            dones_padded,
            lengths,
            gamma=0.99,
            gae_lambda=0.95,
            bootstrap_values=torch.zeros(len(rewards)),
        )

        for episode_idx, length in enumerate(lengths.tolist()):
            expected = compute_gae_reference(
                rewards[episode_idx],
                values[episode_idx],
                dones[episode_idx],
                gamma=0.99,
                gae_lambda=0.95,
            )
            assert torch.equal(actual[episode_idx, :length], expected)
            assert torch.count_nonzero(actual[episode_idx, length:]) == 0

    def test_completed_batch_prepares_returns_advantages_and_chunks(self) -> None:
        """Verify prepare_trajectory_batches calculates returns and GAE advantages on completed trajectory batches."""
        batch = CollectedTrajectory(
            observations=StructuredObservation.empty_batch(3),
            action_masks=torch.ones((3, 2, 49), dtype=torch.bool),
            actions=torch.zeros((3, 2), dtype=torch.long),
            log_probs=torch.zeros(3),
            values=torch.tensor([0.2, 0.1, 0.0]),
            rewards=torch.tensor([0.0, 1.0, 0.5]),
            dones=torch.tensor([0.0, 1.0, 1.0]),
            length=3,
            bootstrap_value=0.0,
            series_history=(),
        )
        prepared = prepare_trajectory_batches(
            [batch], torch.device("cpu"), gamma=0.99, gae_lambda=0.95
        )
        assert prepared[0].returns.shape == (3,)
        assert prepared[0].advantages.shape == (3,)

    def test_completed_batch_only_retains_ppo_inputs_on_target_device(self) -> None:
        """Verify preparation drops collection-only tensors and moves PPO inputs."""
        batch = CollectedTrajectory(
            observations=StructuredObservation.empty_batch(1),
            action_masks=torch.ones((1, 2, 49), dtype=torch.bool),
            actions=torch.zeros((1, 2), dtype=torch.long),
            log_probs=torch.zeros(1),
            values=torch.zeros(1),
            rewards=torch.ones(1),
            dones=torch.ones(1),
            length=1,
            bootstrap_value=0.0,
            series_history=(),
        )

        prepared = prepare_trajectory_batches(
            [batch], torch.device("meta"), gamma=0.99, gae_lambda=0.95
        )
        result = prepared[0]

        assert result.observations.categorical.device.type == "meta"
        assert result.returns.device.type == "meta"
        assert result.advantages.device.type == "meta"
        assert result.truncated is False
        assert not hasattr(result, "values")
        assert not hasattr(result, "rewards")
        assert not hasattr(result, "dones")

    def test_truncated_trajectory_bootstraps_its_return(self) -> None:
        batch = CollectedTrajectory(
            observations=StructuredObservation.empty_batch(1),
            action_masks=torch.ones((1, 2, 49), dtype=torch.bool),
            actions=torch.zeros((1, 2), dtype=torch.long),
            log_probs=torch.zeros(1),
            values=torch.tensor([0.25]),
            rewards=torch.tensor([0.5]),
            dones=torch.zeros(1),
            length=1,
            bootstrap_value=0.75,
            series_history=(),
        )

        result = prepare_trajectory_batches(
            [batch], torch.device("cpu"), gamma=0.9, gae_lambda=0.95
        )[0]

        assert result.truncated is True
        torch.testing.assert_close(result.returns, torch.tensor([0.5 + 0.9 * 0.75]))

    def test_preparing_no_trajectories_is_a_noop(self) -> None:
        """Verify prepare_trajectory_batches handles empty input gracefully."""
        assert (
            prepare_trajectory_batches(
                [],
                torch.device("cpu"),
                gamma=0.99,
                gae_lambda=0.95,
            )
            == []
        )
