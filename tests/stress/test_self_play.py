from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.policy import ActOutput
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import TrainingConfig
from p0.training.rollout import BattleMemoryBuffer, collect_rollouts
from p0.training.trajectory import TrajectoryStorage
from tests.stress._helpers import stress_repetitions


class _SelfPlaySeries:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model

    def resample_single_game(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.full((tokens.size(0), 4, self.d_model), 9.0)


class _SelfPlayPolicy:
    device = torch.device("cpu")
    d_model = 2

    def __init__(self) -> None:
        self.calls = 0
        self.series = _SelfPlaySeries(self.d_model)

    def encode(self, obs, action_mask):
        del action_mask
        return obs

    def prepare(self, encoded, memory):
        return encoded, memory

    def act(self, prepared, action_mask, *, top_p=1.0, deterministic=False):
        del prepared, top_p, deterministic
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

    def __init__(
        self,
        *,
        done_status: tuple[int, int] = (1, 1),
        series_complete: tuple[bool, bool] = (True, True),
    ) -> None:
        self.done_status = np.asarray(done_status, dtype=np.int8)
        self.series_complete = series_complete
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
        infos = []
        for index, status in enumerate(self.done_status):
            info: dict[str, Any] = {
                "series_id": f"series-{index}",
                "series_complete": self.series_complete[index],
            }
            if status == 2:
                info["terminal_observation1"] = self.obs1_buffers[index].clone()
                info["terminal_observation2"] = self.obs2_buffers[index].clone()
            infos.append(info)
        return (
            self.last_masks1,
            self.last_masks2,
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32),
            self.done_status,
            infos,
        )


@pytest.mark.stress
def test_self_play_rollout_keeps_two_perspectives_and_clears_game_state() -> None:
    config = TrainingConfig(n_envs=2, rollout_steps=stress_repetitions(default=128))
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
    assert not memory1.step_counts.any()
    assert not memory2.step_counts.any()
    assert series1._store == {}
    assert series2._store == {}


@pytest.mark.stress
def test_battle_memory_window_is_bounded_and_reset_is_local() -> None:
    repetitions = max(HISTORY_WINDOW, stress_repetitions(default=2048))
    buffer = BattleMemoryBuffer(2, d_model=3, max_steps=repetitions)
    for index in range(repetitions):
        buffer.append(torch.tensor([0, 1]), torch.full((2, 3), float(index)))

    history, mask, ages = buffer.inputs(torch.tensor([0, 1]), torch.device("cpu"), torch.float32)
    assert history.shape[0] == mask.shape[0] == ages.shape[0] == 2
    assert mask[0].sum().item() == mask[1].sum().item()
    assert torch.equal(history[0], history[1])
    assert torch.equal(
        history[0, -HISTORY_WINDOW:, 0],
        torch.arange(repetitions - HISTORY_WINDOW, repetitions, dtype=torch.float32),
    )

    with pytest.raises(OverflowError, match="exceeded"):
        buffer.append(torch.tensor([0, 1]), torch.full((2, 3), float(repetitions)))

    buffer.reset(0)
    empty_history, empty_mask, empty_ages = buffer.inputs(
        torch.tensor([0]), torch.device("cpu"), torch.float32
    )
    assert not empty_mask.any()
    assert not empty_history.any()
    assert not empty_ages.any()
    assert mask[1].any()


@pytest.mark.stress
def test_self_play_rollout_bootstraps_truncation_and_keeps_open_series() -> None:
    config = TrainingConfig(n_envs=2, rollout_steps=1)
    vec_env = _SelfPlayVecEnv(done_status=(2, 1), series_complete=(False, True))
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

    assert policy.calls == 2
    assert [batch.bootstrap_value for batch in completed] == [0.0, 1.0, 0.0, 0.0]
    assert set(series1._store) == {"series-0"}
    assert set(series2._store) == {"series-0"}
