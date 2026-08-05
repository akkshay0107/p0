from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.model.policy import ActOutput
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import TrainingConfig
from p0.training.rollout import BattleMemoryBuffer, collect_rollouts
from p0.training.trajectory import TrajectoryStorage
from tests.stress._helpers import stress_repetitions


class _SelfPlayPolicy:
    device = torch.device("cpu")
    d_model = 2

    def __init__(self) -> None:
        self.calls = 0
        self.series = SimpleNamespace(
            resample_single_game=lambda tokens: torch.full((tokens.size(0), 4, self.d_model), 9.0)
        )

    def act_obs(self, obs, action_mask, series_tokens, series_mask, *memory):
        del obs, series_tokens, series_mask, memory
        self.calls += 1
        batch = action_mask.size(0)
        return ActOutput(
            actions=torch.zeros((batch, 2), dtype=torch.long),
            log_probs=torch.full((batch,), -0.25),
            value=torch.arange(batch, dtype=torch.float32),
            history_token=torch.full((batch, self.d_model), float(self.calls)),
        )


class _SelfPlayVecEnv:
    n_envs = 2

    def __init__(self) -> None:
        self.last_masks1 = np.ones((self.n_envs, 2, ACT_SIZE), dtype=np.bool_)
        self.last_masks2 = self.last_masks1.copy()
        self.obs1_buffers = StructuredObservation.empty_batch(self.n_envs)
        self.obs2_buffers = StructuredObservation.empty_batch(self.n_envs)
        self.last_infos = [{"series_id": f"series-{i}"} for i in range(self.n_envs)]
        self.envs = [
            SimpleNamespace(
                agent1=SimpleNamespace(username=f"first-{i}"),
                agent2=SimpleNamespace(username=f"second-{i}"),
            )
            for i in range(self.n_envs)
        ]

    def get_batched_obs1(self, device: torch.device) -> StructuredObservation:
        return self.obs1_buffers.to(device)

    def get_batched_obs2(self, device: torch.device) -> StructuredObservation:
        return self.obs2_buffers.to(device)

    def step(self, actions: list[dict[str, np.ndarray]]):
        assert all(set(action) == {f"first-{i}", f"second-{i}"} for i, action in enumerate(actions))
        return (
            self.last_masks1,
            self.last_masks2,
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32),
            np.array([1, 1], dtype=np.int8),
            [
                {"series_id": "series-0", "series_complete": True},
                {"series_id": "series-1", "series_complete": True},
            ],
        )


@pytest.mark.stress
def test_self_play_rollout_keeps_two_perspectives_and_clears_game_state() -> None:
    config = TrainingConfig(n_envs=2, rollout_steps=stress_repetitions(default=3))
    vec_env = _SelfPlayVecEnv()
    policy = _SelfPlayPolicy()
    completed: list[Any] = []
    first = TrajectoryStorage.allocate(config.n_envs, config.rollout_steps, policy.d_model)
    second = TrajectoryStorage.allocate(config.n_envs, config.rollout_steps, policy.d_model)
    memory1 = BattleMemoryBuffer(config.n_envs, policy.d_model)
    memory2 = BattleMemoryBuffer(config.n_envs, policy.d_model)
    series1 = SeriesTokenStore(policy.d_model)
    series2 = SeriesTokenStore(policy.d_model)

    collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        completed,
        config,
        first,
        second,
        memory1,
        memory2,
        series1,
        series2,
    )

    assert len(completed) == 2 * vec_env.n_envs * config.rollout_steps
    assert [batch.length for batch in completed] == [1] * len(completed)
    assert all(
        torch.equal(batch.actions, torch.zeros((1, 2), dtype=torch.long)) for batch in completed
    )
    assert first.step_counts.tolist() == [0, 0]
    assert second.step_counts.tolist() == [0, 0]
    assert all(not tokens for tokens in memory1.tokens + memory2.tokens)
    assert series1._store == {}
    assert series2._store == {}


@pytest.mark.stress
def test_battle_memory_window_is_bounded_and_reset_is_local() -> None:
    buffer = BattleMemoryBuffer(2, d_model=3)
    for index in range(stress_repetitions(default=8)):
        buffer.append(torch.tensor([0, 1]), torch.full((2, 3), float(index)))

    history, mask, ages = buffer.inputs(torch.tensor([0, 1]), torch.device("cpu"), torch.float32)
    assert history.shape[0] == mask.shape[0] == ages.shape[0] == 2
    assert mask[0].sum().item() == mask[1].sum().item()
    assert torch.equal(history[0], history[1])

    buffer.reset(0)
    empty_history, empty_mask, empty_ages = buffer.inputs(
        torch.tensor([0]), torch.device("cpu"), torch.float32
    )
    assert not empty_mask.any()
    assert not empty_history.any()
    assert not empty_ages.any()
    assert mask[1].any()
