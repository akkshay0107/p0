from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import SERIES_SLOTS
from p0.model.policy import ActOutput
from p0.model.structured_observation import StructuredObservation
from p0.training.config import TrainingConfig
from p0.training.rollout import (
    BattleMemoryBuffer,
    RolloutBuffer,
    collect_rollouts,
)
from p0.training.trajectory import TrajectoryStorage, compute_gae_batch
from p0.training.vector_env import ThreadVecEnv

ACT_SIZE = FORMAT.action_size


class FakePolicy:
    def __init__(self, action: int):
        self.action = action
        self.device = torch.device("cpu")
        self.d_model = 1
        self.batch_sizes: list[int] = []

    def act_obs(
        self,
        obs: StructuredObservation,
        action_mask: torch.Tensor,
        series_tokens: torch.Tensor,
        series_mask: torch.Tensor,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        history_age_ids: torch.Tensor,
    ) -> ActOutput:
        del series_tokens, series_mask, history_tokens, history_mask, history_age_ids
        batch_size = action_mask.size(0)
        self.batch_sizes.append(batch_size)
        actions = torch.full((batch_size, 2), self.action, dtype=torch.long)
        return ActOutput(
            actions=actions,
            log_probs=torch.full((batch_size,), -0.5),
            value=torch.full((batch_size,), 0.25),
            history_token=torch.ones((batch_size, 1)),
        )

    def encode_series(self, histories):
        batch_size = len(histories) if isinstance(histories, list) and histories else 1
        return (
            torch.zeros((batch_size, SERIES_SLOTS, self.d_model)),
            torch.zeros((batch_size, SERIES_SLOTS), dtype=torch.bool),
        )


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


class FakeSeriesProvider:
    def __init__(self, n_envs: int):
        self.game_numbers = [1 for _ in range(n_envs)]
        self.completed: list[int] = []
        self.features = torch.randn(1, 5, 1)

    def current(self, env_id: int, player: int):
        from p0.training.rollout import RolloutSeriesContext

        del player
        game_number = self.game_numbers[env_id]
        return RolloutSeriesContext(
            series_id=f"series-{env_id}",
            game_number=game_number,
            features=self.features if game_number > 1 else None,
        )

    def on_game_end(self, env_id: int, info):
        del info
        self.completed.append(env_id)
        self.game_numbers[env_id] += 1


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


def test_collect_rollouts_records_both_self_play_streams():
    config = TrainingConfig(n_envs=3, rollout_steps=1)
    vec_env = FakeVecEnv(config.n_envs)
    policy = FakePolicy(action=7)
    buffer = RolloutBuffer()
    trajectories1 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1, player_index=0)
    trajectories2 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1, player_index=1)
    memory1 = BattleMemoryBuffer(config.n_envs, 1)
    memory2 = BattleMemoryBuffer(config.n_envs, 1)

    collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        buffer,
        config,
        trajectories1,
        trajectories2,
        memory1,
        memory2,
    )

    assert len(buffer.trajectories) == 2 * config.n_envs
    assert all(torch.all(episode.actions == 7) for episode in buffer.trajectories)
    assert trajectories1.step_counts.tolist() == [0, 0, 0]
    assert trajectories2.step_counts.tolist() == [0, 0, 0]
    assert all(not entries for entries in memory1.tokens)
    assert all(not entries for entries in memory2.tokens)
    side_two_actions = [
        actions[f"agent2-{env_id}"] for env_id, actions in enumerate(vec_env.received_actions[0])
    ]
    assert all(action.tolist() == [7, 7] for action in side_two_actions)
    assert policy.batch_sizes == [2 * config.n_envs]


def test_rollout_provider_preserves_series_context_without_cross_game_history():
    config = TrainingConfig(n_envs=3, rollout_steps=2)
    vec_env = FakeVecEnv(config.n_envs)
    provider = FakeSeriesProvider(config.n_envs)
    trajectories1 = TrajectoryStorage.allocate(
        config.n_envs, max_steps=4, d_model=1, player_index=0
    )
    trajectories2 = TrajectoryStorage.allocate(
        config.n_envs, max_steps=4, d_model=1, player_index=1
    )
    memory1 = BattleMemoryBuffer(config.n_envs, 1)
    memory2 = BattleMemoryBuffer(config.n_envs, 1)
    buffer = RolloutBuffer()

    collect_rollouts(
        cast(Any, vec_env),
        cast(Any, FakePolicy(action=7)),
        buffer,
        config,
        trajectories1,
        trajectories2,
        memory1,
        memory2,
        provider,
    )

    assert provider.completed == [0, 1, 2, 0, 1, 2]
    second_game = buffer.trajectories[6]
    assert second_game.series_id == "series-0"
    assert second_game.game_number == 2
    assert second_game.player == 0
    assert second_game.series_features is not None
    assert second_game.series_tokens is None
    assert second_game.series_mask is None
    assert all(not entries for entries in memory1.tokens)

