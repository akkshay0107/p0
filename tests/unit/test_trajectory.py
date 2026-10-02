"""Tests for trajectory advantage and batch preparation."""

from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.structured_observation import StructuredObservation
from p0.training.trajectory import (
    CollectedTrajectory,
    TrajectoryStorage,
    compute_gae_batch,
    prepare_trajectory_batches,
)


class TestTrajectory:
    def test_gae_handles_terminal_truncated_and_padded_trajectories(self) -> None:
        # Rows cover terminal, truncated, an internal terminal boundary, and one-step episodes.
        # Padding is deliberately nonzero so forgetting the length mask cannot pass.
        rewards = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 99.0], [2.0, 4.0, 8.0], [2.0, 99.0, 99.0]]
        )
        values = torch.tensor(
            [[0.5, 1.0, 1.5], [0.5, 1.0, 99.0], [1.0, 2.0, 3.0], [1.0, 99.0, 99.0]]
        )
        dones = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        actual = compute_gae_batch(
            rewards,
            values,
            dones,
            torch.tensor([3, 2, 3, 1]),
            gamma=0.5,
            gae_lambda=0.5,
            bootstrap_values=torch.tensor([9.0, 4.0, 9.0, 9.0]),
        )

        # Terminal row: residuals 1, 1.75, 1.5, with future advantages discounted by 0.25.
        # Truncated row: the final residual includes 0.5 * bootstrap(4), giving 3.
        expected = torch.tensor(
            [
                [1.53125, 2.125, 1.5],
                [1.75, 3.0, 0.0],
                [1.0, 4.75, 5.0],
                [1.0, 0.0, 0.0],
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

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
        result = prepared[0]
        torch.testing.assert_close(
            result.returns,
            torch.tensor([0.94545, 1.0, 0.5]),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.advantages,
            torch.tensor([0.18397214, 1.12235148, -1.30632362]),
            rtol=0,
            atol=1e-6,
        )
        assert result.advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert result.advantages.var(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)

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

        ppo_tensors = (
            *result.observations.tensors(),
            result.action_masks,
            result.actions,
            result.log_probs,
            result.returns,
            result.advantages,
        )
        assert all(tensor.device.type == "meta" for tensor in ppo_tensors)
        assert result.truncated is False

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


class TestTrajectoryStorage:
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
        torch.testing.assert_close(
            history[0, :, 0],
            torch.arange(total - HISTORY_WINDOW, total, dtype=torch.float32),
            rtol=0,
            atol=0,
        )

        full_history = storage.full_history(0)
        assert full_history is not None
        torch.testing.assert_close(
            full_history[0, :, 0], torch.arange(total, dtype=torch.float32), rtol=0, atol=0
        )
        completed = storage.complete(0, 0.0, ())
        assert completed.length == total
        assert completed.observations.categorical.shape[0] == total
        assert storage.full_history(0) is None

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
        reset_history, reset_mask = storage.history_inputs(
            torch.tensor([0, 1]), torch.device("cpu"), torch.float32
        )
        assert not reset_mask[0].any()
        assert not reset_history[0].any()
        assert reset_mask[1].sum().item() == 2
        assert reset_history[1, -2:, 0].tolist() == [10.0, 11.0]
        second_history = storage.full_history(1)
        assert second_history is not None
        assert second_history[0, :, 0].tolist() == [10.0, 11.0]

    def test_storage_allocates_completes_and_resets_one_environment(self) -> None:
        """Verify TrajectoryStorage allocation, completion slicing, and reset for a single environment index."""
        storage = TrajectoryStorage.allocate(2, 3, 1, "cpu")

        observations = StructuredObservation.empty_batch(2)
        observations.categorical[:, 0, 0] = torch.tensor([11, 22])
        storage.record(
            torch.tensor([0, 1]),
            observations,
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([1.0, 2.0]),
            torch.ones((2, 2, 49), dtype=torch.bool),
            torch.tensor([[10.0], [20.0]]),
        )
        second_observation = StructuredObservation.empty_batch(1)
        second_observation.categorical[0, 0, 0] = 23
        storage.record(
            torch.tensor([1]),
            second_observation,
            torch.tensor([[5, 6]]),
            torch.tensor([0.3]),
            torch.tensor([3.0]),
            torch.ones((1, 2, 49), dtype=torch.bool),
            torch.tensor([[30.0]]),
        )

        completed = storage.complete(1, 0.0, ())
        assert completed.length == 2
        assert completed.actions.tolist() == [[3, 4], [5, 6]]
        assert completed.observations.categorical[:, 0, 0].tolist() == [22, 23]
        assert storage.step_counts.tolist() == [1, 0]
        neighbor_history = storage.full_history(0)
        assert neighbor_history is not None
        torch.testing.assert_close(neighbor_history, torch.tensor([[[10.0]]]))

        reused_observation = StructuredObservation.empty_batch(1)
        reused_observation.categorical[0, 0, 0] = 99
        storage.record(
            torch.tensor([1]),
            reused_observation,
            torch.tensor([[7, 8]]),
            torch.tensor([0.4]),
            torch.tensor([4.0]),
            torch.ones((1, 2, 49), dtype=torch.bool),
            torch.tensor([[40.0]]),
        )
        assert completed.actions.tolist() == [[3, 4], [5, 6]]
        assert completed.observations.categorical[:, 0, 0].tolist() == [22, 23]

    def test_storage_reports_explicit_overflow(self) -> None:
        """Verify record rejects overflow before replacing the stored step."""
        storage = TrajectoryStorage.allocate(1, 1, 1, "cpu")
        observations = StructuredObservation.empty_batch(1)
        observations.categorical[0, 0, 0] = 12
        storage.record(
            torch.tensor([0]),
            observations,
            torch.tensor([[4, 5]]),
            torch.tensor([0.1]),
            torch.tensor([1.0]),
            torch.ones((1, 2, 49), dtype=torch.bool),
            torch.tensor([[12.0]]),
        )
        replacement = StructuredObservation.empty_batch(1)
        replacement.categorical[0, 0, 0] = 99
        with pytest.raises(OverflowError, match="exceeded"):
            storage.record(
                torch.tensor([0]),
                replacement,
                torch.tensor([[8, 9]]),
                torch.tensor([0.9]),
                torch.tensor([9.0]),
                torch.zeros((1, 2, 49), dtype=torch.bool),
                torch.tensor([[99.0]]),
            )
        assert storage.step_counts.tolist() == [1]
        assert storage.actions[0, 0].tolist() == [4, 5]
        assert storage.observations.categorical[0, 0, 0, 0].item() == 12
        assert storage.history_tokens[0, 0, 0].item() == 12.0
