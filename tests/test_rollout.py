from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from src.lookups import ACT_SIZE
from src.model.policy import ActOutput
from src.model.structured_observation import StructuredObservation
from src.train.config import PPOConfig
from src.train.rollout import (
    RolloutBuffer,
    build_partition,
    collect_rollouts,
    compute_gae_batch,
    create_trajectory_buffers,
)
from src.train.vec_env import ThreadVecEnv


class FakePolicy:
    def __init__(self, action: int):
        self.action = action
        self.device = torch.device("cpu")
        self.batch_sizes: list[int] = []

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 1))

    def act_obs(
        self,
        obs: StructuredObservation,
        action_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> ActOutput:
        batch_size = action_mask.size(0)
        self.batch_sizes.append(batch_size)
        actions = torch.full((batch_size, 2), self.action, dtype=torch.long)
        return ActOutput(
            actions=actions,
            log_probs=torch.full((batch_size,), -0.5),
            value=torch.full((batch_size,), 0.25),
            state=state + 1,
        )


class FakePool:
    def __init__(self, opponent_ids: list[str]):
        self.opponent_ids = opponent_ids
        self.updates: list[tuple[str, int, int]] = []
        self.loaded: dict[str, FakePolicy] = {}

    def __len__(self) -> int:
        return len(self.opponent_ids)

    def sample_many(self, count: int) -> list[str]:
        return self.opponent_ids[:count]

    def load_policy(self, opponent_id: str, device: str) -> FakePolicy:
        del device
        policy = FakePolicy(action=8)
        self.loaded[opponent_id] = policy
        return policy

    def update_win_rate(self, opponent_id: str, agent_wins: int, num_games: int = 1) -> None:
        self.updates.append((opponent_id, agent_wins, num_games))


class RotatingFakePool(FakePool):
    def __init__(self):
        super().__init__(["old", "new"])
        self.sample_calls = 0

    def sample_many(self, count: int) -> list[str]:
        del count
        opponent_id = "old" if self.sample_calls == 0 else "new"
        self.sample_calls += 1
        return [opponent_id]


class FakeVecEnv:
    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.last_masks1 = np.ones((n_envs, 2, ACT_SIZE), dtype=np.bool_)
        self.last_masks2 = np.ones((n_envs, 2, ACT_SIZE), dtype=np.bool_)
        self.obs1_buffers = StructuredObservation.empty_batch(n_envs)
        self.obs2_buffers = StructuredObservation.empty_batch(n_envs)
        self.envs = [
            SimpleNamespace(
                agent1=SimpleNamespace(username=f"agent1-{i}"),
                agent2=SimpleNamespace(username=f"agent2-{i}"),
            )
            for i in range(n_envs)
        ]
        self.received_actions: list[list[dict[str, np.ndarray]]] = []

    def get_batched_obs1(self, device: torch.device) -> StructuredObservation:
        return self.obs1_buffers.to(device)

    def get_batched_obs2(self, device: torch.device) -> StructuredObservation:
        return self.obs2_buffers.to(device)

    def step(
        self, actions: list[dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self.received_actions.append(actions)
        rewards1 = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        rewards2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        dones = np.ones(self.n_envs, dtype=np.bool_)
        return (
            self.last_masks1,
            self.last_masks2,
            rewards1,
            rewards2,
            dones,
            [{} for _ in range(self.n_envs)],
        )


class BufferBindingEnv:
    def __init__(self):
        self.targets: tuple[StructuredObservation, StructuredObservation] | None = None

    def set_observation_targets(
        self,
        obs1: StructuredObservation,
        obs2: StructuredObservation,
    ) -> None:
        self.targets = (obs1, obs2)


def test_thread_vec_env_binds_each_env_to_its_preallocated_rows():
    envs = [BufferBindingEnv(), BufferBindingEnv()]
    vec_env = ThreadVecEnv(cast(Any, envs))
    try:
        for env_id, env in enumerate(envs):
            assert env.targets is not None
            obs1, obs2 = env.targets
            assert obs1.numerical.data_ptr() == vec_env.obs1_buffers[env_id].numerical.data_ptr()
            assert obs2.numerical.data_ptr() == vec_env.obs2_buffers[env_id].numerical.data_ptr()
    finally:
        vec_env.shutdown()


def test_compute_gae_batch_matches_single_episode_reference():
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


def test_build_partition_assigns_static_pool_groups():
    config = PPOConfig(n_envs=6, n_self_envs=2, n_pool_opponents=2)
    pool = FakePool(["opp-a", "opp-b"])

    partition = build_partition(
        config,
        cast(Any, pool),
        torch.device("cpu"),
    )

    assert partition.self_idx.tolist() == [0, 1]
    assert partition.pool_idx.tolist() == [2, 3, 4, 5]
    assert partition.opponent_ids == ["self", "self", "opp-a", "opp-b", "opp-a", "opp-b"]
    assert [(name, idx.tolist()) for name, idx in partition.pool_groups()] == [
        ("opp-a", [2, 4]),
        ("opp-b", [3, 5]),
    ]
    assert partition.self_mask_cpu.tolist() == [True, True, False, False, False, False]


def test_build_partition_falls_back_to_all_self_play_for_empty_pool():
    config = PPOConfig(n_envs=4, n_self_envs=1)
    partition = build_partition(
        config,
        cast(Any, FakePool([])),
        torch.device("cpu"),
    )

    assert partition.self_idx.tolist() == [0, 1, 2, 3]
    assert partition.pool_idx.numel() == 0
    assert partition.pool_groups() == ()
    assert partition.opponent_ids == ["self", "self", "self", "self"]


def test_collect_rollouts_counts_pool_games_and_excludes_pool_side_two():
    config = PPOConfig(
        n_envs=3,
        n_self_envs=1,
        n_pool_opponents=1,
        rollout_steps=1,
    )
    pool = FakePool(["opp"])
    partition = build_partition(config, cast(Any, pool), torch.device("cpu"))
    vec_env = FakeVecEnv(config.n_envs)
    policy = FakePolicy(action=7)
    buffer = RolloutBuffer()
    trajectories1 = create_trajectory_buffers(config.n_envs, max_steps=4)
    trajectories2 = create_trajectory_buffers(config.n_envs, max_steps=4)
    state1 = torch.ones((config.n_envs, 1, 1))
    state2 = torch.ones((config.n_envs, 1, 1))

    stats, next_state1, next_state2 = collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        buffer,
        cast(Any, pool),
        config,
        cast(Any, {}),
        trajectories1,
        trajectories2,
        state1,
        state2,
        partition,
    )

    assert stats == (1, 1)
    assert pool.updates == [("opp", 1, 1)]
    assert len(buffer.trajectories) == 4
    assert all(torch.all(episode["actions"] == 7) for episode in buffer.trajectories)
    assert trajectories1["step_counts"].tolist() == [0, 0, 0]
    assert trajectories2["step_counts"].tolist() == [0, 0, 0]
    assert torch.count_nonzero(next_state1) == 0
    assert torch.count_nonzero(next_state2) == 0

    side_two_actions = [
        actions[f"agent2-{env_id}"] for env_id, actions in enumerate(vec_env.received_actions[0])
    ]
    assert side_two_actions[0].tolist() == [7, 7]
    assert side_two_actions[1].tolist() == [8, 8]
    assert side_two_actions[2].tolist() == [8, 8]
    assert policy.batch_sizes == [4]
    assert pool.loaded["opp"].batch_sizes == [2]


def test_pool_opponent_rotates_only_after_completed_battle():
    config = PPOConfig(
        n_envs=3,
        n_self_envs=1,
        n_pool_opponents=1,
        rollout_steps=1,
    )
    pool = RotatingFakePool()
    partition = build_partition(config, cast(Any, pool), torch.device("cpu"))
    assert partition.opponent_ids == ["self", "old", "old"]

    vec_env = FakeVecEnv(config.n_envs)
    policy = FakePolicy(action=7)
    stats, _, _ = collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        RolloutBuffer(),
        cast(Any, pool),
        config,
        cast(Any, {}),
        create_trajectory_buffers(config.n_envs, max_steps=4),
        create_trajectory_buffers(config.n_envs, max_steps=4),
        torch.ones((config.n_envs, 1, 1)),
        torch.ones((config.n_envs, 1, 1)),
        partition,
    )

    assert stats == (1, 1)
    assert pool.updates == [("old", 1, 1)]
    assert partition.opponent_ids == ["self", "new", "new"]
