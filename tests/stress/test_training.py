"""Generated return and normalization invariants for trajectory preparation."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.format_config import FORMAT
from p0.model.structured_observation import StructuredObservation
from p0.training.trajectory import CollectedTrajectory, prepare_trajectory_batches
from tests.stress._helpers import stress_count


class TestTrajectoryNormalization:
    @pytest.mark.stress
    @settings(max_examples=32, deadline=None)
    @given(
        lengths=st.lists(
            st.integers(min_value=1, max_value=stress_count("P0_STRESS_TRAJECTORY_STEPS", 64)),
            min_size=1,
            max_size=stress_count("P0_STRESS_TRAJECTORIES", 64),
        )
    )
    def test_zero_discount_preserves_rewards_and_normalizes_across_trajectories(
        self, lengths: list[int]
    ) -> None:
        rewards = torch.arange(1, sum(lengths) + 1, dtype=torch.float32).split(lengths)
        trajectories = [
            CollectedTrajectory(
                observations=StructuredObservation.empty_batch(length),
                action_masks=torch.ones((length, 2, FORMAT.action_size), dtype=torch.bool),
                actions=torch.zeros((length, 2), dtype=torch.long),
                log_probs=torch.zeros(length),
                values=-reward,
                rewards=reward,
                dones=torch.zeros(length),
                length=length,
                bootstrap_value=100.0,
                series_history=(),
            )
            for length, reward in zip(lengths, rewards, strict=True)
        ]

        prepared = prepare_trajectory_batches(
            trajectories, torch.device("cpu"), gamma=0.0, gae_lambda=0.95
        )

        assert [batch.length for batch in prepared] == lengths
        for batch, reward in zip(prepared, rewards, strict=True):
            # With no future discount, each return is exactly its immediate reward.
            torch.testing.assert_close(batch.returns, reward, rtol=0, atol=0)
        advantages = torch.cat([batch.advantages for batch in prepared])
        assert torch.isfinite(advantages).all()
        assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
        expected_variance = 1.0 if sum(lengths) > 1 else 0.0
        assert advantages.var(unbiased=False).item() == pytest.approx(expected_variance, abs=1e-6)
        # A shared normalization preserves order across trajectory boundaries too.
        assert torch.all(advantages[1:] > advantages[:-1])
