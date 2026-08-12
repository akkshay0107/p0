from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import PassBattleOrder

from p0.battle.actions import ACT_SIZE
from p0.evaluation.harness import (
    EvaluationHarness,
)
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import ActOutput
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.rl_player import RLPlayer
from p0.runtime.poke_env_action_adapter import action_to_single_order
from p0.teams.source import FixedTeamSource
from p0.training.config import TrainingConfig
from p0.training.rollout import (
    BattleMemoryBuffer,
    collect_rollouts,
)
from p0.training.trajectory import (
    TrajectoryBatch,
    TrajectoryStorage,
    compute_gae_batch,
    prepare_trajectory_batches,
)
from p0.training.vector_env import ThreadVecEnv


class FakePolicy:
    def __init__(self, action: int):
        self.action = action
        self.device = torch.device("cpu")
        self.d_model = 1
        self.batch_sizes: list[int] = []
        self.summarized_lengths: list[int] = []
        self.series = SimpleNamespace(resample_single_game=self._resample_single_game)

    def _resample_single_game(self, history: torch.Tensor) -> torch.Tensor:
        self.summarized_lengths.append(history.size(1))
        return torch.zeros((history.size(0), 4, self.d_model))

    def encode(self, obs: StructuredObservation, action_mask: torch.Tensor):
        del action_mask
        return obs

    def prepare(self, encoded, memory):
        return encoded, memory

    def act(
        self,
        prepared,
        action_mask: torch.Tensor,
        *,
        top_p: float = 1.0,
        deterministic: bool = False,
    ) -> ActOutput:
        del top_p, deterministic
        _, memory = prepared
        assert torch.count_nonzero(memory.series_tokens) == 0
        assert torch.count_nonzero(memory.series_mask) == 0
        batch_size = action_mask.size(0)
        self.batch_sizes.append(batch_size)
        actions = torch.full((batch_size, 2), self.action, dtype=torch.long)
        return ActOutput(
            actions=actions,
            log_probs=torch.full((batch_size,), -0.5),
            value=torch.full((batch_size,), 0.25),
            history_token=torch.ones((batch_size, 1)),
        )


class FakeVecEnv:
    def __init__(self, n_envs: int, done_status: int = 1):
        self.n_envs = n_envs
        self.done_status = done_status
        self.last_masks1 = np.ones((n_envs, 2, ACT_SIZE), dtype=np.bool_)
        self.last_masks2 = np.ones((n_envs, 2, ACT_SIZE), dtype=np.bool_)
        self.obs1_buffers = StructuredObservation.empty_batch(n_envs)
        self.obs2_buffers = StructuredObservation.empty_batch(n_envs)
        self.last_infos = [{"series_id": f"series-{i}"} for i in range(n_envs)]
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
        rewards1 = np.resize(np.array([0.0, 1.0, 1.0], dtype=np.float32), self.n_envs)
        rewards2 = np.resize(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.n_envs)
        done_status = np.full(self.n_envs, self.done_status, dtype=np.int64)
        infos: list[dict[str, Any]] = []
        for env_id in range(self.n_envs):
            info: dict[str, Any] = {"series_id": f"series-{env_id}"}
            if self.done_status == 2:
                info["terminal_observation1"] = StructuredObservation.empty_batch(1)[0]
                info["terminal_observation2"] = StructuredObservation.empty_batch(1)[0]
            infos.append(info)
        return (
            self.last_masks1,
            self.last_masks2,
            rewards1,
            rewards2,
            done_status,
            infos,
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


class StatusEnv(BufferBindingEnv):
    """Minimal SimEnv stand-in reporting a fixed terminated/truncated state."""

    def __init__(self, *, terminated: bool, truncated: bool):
        super().__init__()
        self.terminated = terminated
        self.truncated = truncated
        self.agent1 = SimpleNamespace(username="agent1")
        self.agent2 = SimpleNamespace(username="agent2")
        self.series_id = "series-0"
        self.series_scores = [0, 0]
        self.series_games_played = 1

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        mask = np.ones(2 * ACT_SIZE, dtype=np.int64)
        return {name: {"action_mask": mask} for name in ("agent1", "agent2")}

    def reset(self) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
        return self._observations(), {}

    def step(self, action: dict[str, np.ndarray]):
        del action
        flags = {"agent1": self.terminated, "agent2": self.terminated}
        return (
            self._observations(),
            {"agent1": 0.0, "agent2": 0.0},
            flags,
            {"agent1": self.truncated, "agent2": self.truncated},
            {},
        )


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected"),
    [(False, False, 0), (True, False, 1), (False, True, 2)],
)
def test_thread_vec_env_reports_a_tri_state_done_status(
    terminated: bool, truncated: bool, expected: int
) -> None:
    # A boolean array here would report truncation as termination, which
    # silently disables bootstrapping for every game that hits the step cap.
    vec_env = ThreadVecEnv(cast(Any, [StatusEnv(terminated=terminated, truncated=truncated)]))
    try:
        done_status = vec_env.step([{}])[4]
        assert done_status.dtype == np.int64
        assert done_status.tolist() == [expected]
    finally:
        vec_env.shutdown()


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


def test_thread_vec_env_rejects_an_action_count_mismatch():
    vec_env = ThreadVecEnv(cast(Any, [BufferBindingEnv(), BufferBindingEnv()]))
    try:
        with pytest.raises(ValueError, match="Number of actions"):
            vec_env.step([{}])
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


def test_collect_rollouts_records_both_self_play_streams():
    config = TrainingConfig(n_envs=3, rollout_steps=1)
    vec_env = FakeVecEnv(config.n_envs)
    policy = FakePolicy(action=7)
    buffer = []
    trajectories1 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    trajectories2 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    memory1 = BattleMemoryBuffer(config.n_envs, 1)
    memory2 = BattleMemoryBuffer(config.n_envs, 1)

    from p0.model.token_store import SeriesTokenStore

    series_store1 = SeriesTokenStore(1)
    series_store2 = SeriesTokenStore(1)

    collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        buffer,
        config,
        trajectories1,
        trajectories2,
        memory1,
        memory2,
        series_store1,
        series_store2,
    )

    assert len(buffer) == 2 * config.n_envs
    assert all(torch.all(episode.actions == 7) for episode in buffer)
    assert all(float(episode.dones[-1]) == 1.0 for episode in buffer)
    assert all(episode.bootstrap_value == 0.0 for episode in buffer)
    assert trajectories1.step_counts.tolist() == [0, 0, 0]
    assert trajectories2.step_counts.tolist() == [0, 0, 0]
    assert all(not entries for entries in memory1.tokens)
    assert all(not entries for entries in memory2.tokens)
    side_two_actions = [
        actions[f"agent2-{env_id}"] for env_id, actions in enumerate(vec_env.received_actions[0])
    ]
    assert all(action.tolist() == [7, 7] for action in side_two_actions)
    assert policy.batch_sizes == [2 * config.n_envs]


def test_truncated_rollout_bootstraps_instead_of_ending_the_game():
    config = TrainingConfig(n_envs=3, rollout_steps=1)
    vec_env = FakeVecEnv(config.n_envs, done_status=2)
    policy = FakePolicy(action=7)
    buffer: list[TrajectoryBatch] = []
    trajectories1 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    trajectories2 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)

    from p0.model.token_store import SeriesTokenStore

    collect_rollouts(
        cast(Any, vec_env),
        cast(Any, policy),
        buffer,
        config,
        trajectories1,
        trajectories2,
        BattleMemoryBuffer(config.n_envs, 1),
        BattleMemoryBuffer(config.n_envs, 1),
        SeriesTokenStore(1),
        SeriesTokenStore(1),
    )

    assert len(buffer) == 2 * config.n_envs
    # A truncated game is not a terminal state: it keeps a zero done flag so the
    # value target bootstraps off the terminal observation instead of assuming 0.
    assert all(float(episode.dones[-1]) == 0.0 for episode in buffer)
    assert all(episode.bootstrap_value == pytest.approx(0.25) for episode in buffer)


def test_battle_memory_keeps_the_whole_game_but_windows_the_reducer_inputs():
    memory = BattleMemoryBuffer(1, d_model=1)
    env_ids = torch.tensor([0])
    total = HISTORY_WINDOW + 5
    for step in range(total):
        memory.append(env_ids, torch.full((1, 1), float(step)))

    # The end-of-game series summary compresses every decision.
    assert len(memory.tokens[0]) == total

    history, mask, ages = memory.inputs(env_ids, torch.device("cpu"), torch.float32)
    assert history.shape == (1, HISTORY_WINDOW, 1)
    assert bool(mask.all())
    assert history[0, -1, 0].item() == float(total - 1)
    assert history[0, 0, 0].item() == float(total - HISTORY_WINDOW)
    assert ages[0, -1].item() == 0


def test_live_player_spills_history_in_fixed_windows():
    player = cast(Any, RLPlayer.__new__(RLPlayer))
    player.policy = SimpleNamespace(device=torch.device("cpu"), d_model=1)
    player._memory_model_id = id(player.policy)
    player._empty_history_tensor = torch.zeros((1, 0, 1))
    player._battle_histories = {}
    player._series_store = SeriesTokenStore(1)

    battle = SimpleNamespace(battle_tag="battle-1")
    capacity = 2 * HISTORY_WINDOW
    for step in range(capacity):
        player._append_history(battle, torch.tensor([float(step)]))

    history = player._battle_histories["battle-1"]
    assert history.decision_count == capacity
    assert not history.cpu_chunks
    assert len(history.device_tokens) == capacity

    player._append_history(battle, torch.tensor([float(capacity)]))

    total = capacity + 1
    assert history.decision_count == total
    assert [chunk.size(0) for chunk in history.cpu_chunks] == [HISTORY_WINDOW]
    assert len(history.device_tokens) == HISTORY_WINDOW + 1
    assert history.cpu_chunks[0][:, 0].tolist() == list(range(HISTORY_WINDOW))
    assert history.device_tokens[0].item() == HISTORY_WINDOW
    assert history.complete_values(torch.device("cpu"))[0, :, 0].tolist() == list(range(total))

    for step in range(total, 3 * HISTORY_WINDOW + 1):
        player._append_history(battle, torch.tensor([float(step)]))

    total = 3 * HISTORY_WINDOW + 1
    assert history.decision_count == total
    assert [chunk.size(0) for chunk in history.cpu_chunks] == [HISTORY_WINDOW] * 2
    assert len(history.device_tokens) == HISTORY_WINDOW + 1
    assert history.complete_values(torch.device("cpu"))[0, :, 0].tolist() == list(range(total))
    memory = player._memory_inputs(battle)
    assert memory.history_tokens[0, -1, 0].item() == float(total - 1)
    assert memory.history_tokens[0, 0, 0].item() == float(total - HISTORY_WINDOW)


def test_end_of_game_series_summary_sees_every_decision():
    from p0.model.token_store import SeriesTokenStore

    vec_env = FakeVecEnv(1, done_status=0)
    policy = FakePolicy(action=7)
    memory1 = BattleMemoryBuffer(1, 1)
    memory2 = BattleMemoryBuffer(1, 1)

    def collect(steps: int) -> None:
        collect_rollouts(
            cast(Any, vec_env),
            cast(Any, policy),
            [],
            TrainingConfig(n_envs=1, rollout_steps=steps),
            TrajectoryStorage.allocate(1, max_steps=200, d_model=1),
            TrajectoryStorage.allocate(1, max_steps=200, d_model=1),
            memory1,
            memory2,
            SeriesTokenStore(1),
            SeriesTokenStore(1),
        )

    # Run past the reducer window without finishing the game.
    collect(HISTORY_WINDOW + 5)
    assert not policy.summarized_lengths

    vec_env.done_status = 1
    collect(1)

    # The summary compresses the whole battle, not just the reducer window.
    assert policy.summarized_lengths == [HISTORY_WINDOW + 6, HISTORY_WINDOW + 6]


def test_storage_allocates_completes_and_resets_one_environment():
    storage = TrajectoryStorage.allocate(2, 3, d_model=1)
    storage.step_counts[1] = 2
    storage.actions[1, :2] = 7
    completed = storage.complete(1)
    assert completed is not None
    assert completed.length == 2
    assert torch.all(completed.actions == 7)
    assert storage.step_counts.tolist() == [0, 0]


def test_storage_reports_explicit_overflow():
    storage = TrajectoryStorage.allocate(1, 1, d_model=1)
    storage.step_counts[0] = 1
    with pytest.raises(OverflowError, match="exceeded"):
        storage.ensure_capacity(torch.tensor([0]))


def test_completed_batch_prepares_returns_advantages_and_chunks():
    batch = TrajectoryBatch(
        observations=StructuredObservation.empty_batch(3),
        action_masks=torch.ones((3, 2, 49), dtype=torch.bool),
        actions=torch.zeros((3, 2), dtype=torch.long),
        log_probs=torch.zeros(3),
        values=torch.tensor([0.2, 0.1, 0.0]),
        rewards=torch.tensor([0.0, 1.0, 0.5]),
        dones=torch.tensor([0.0, 1.0, 1.0]),
        length=3,
    )
    prepared = prepare_trajectory_batches([batch], torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    assert prepared[0].returns is not None
    assert prepared[0].advantages is not None


def test_completed_batch_only_moves_ppo_inputs_to_target_device():
    batch = TrajectoryBatch(
        observations=StructuredObservation.empty_batch(1),
        action_masks=torch.ones((1, 2, 49), dtype=torch.bool),
        actions=torch.zeros((1, 2), dtype=torch.long),
        log_probs=torch.zeros(1),
        values=torch.zeros(1),
        rewards=torch.ones(1),
        dones=torch.ones(1),
        length=1,
    )

    prepared = prepare_trajectory_batches(
        [batch], torch.device("meta"), gamma=0.99, gae_lambda=0.95
    )
    result = prepared[0]

    assert result.observations.categorical.device.type == "meta"
    assert result.returns is not None and result.returns.device.type == "meta"
    assert result.advantages is not None and result.advantages.device.type == "meta"
    assert result.values.device.type == "cpu"
    assert result.rewards.device.type == "cpu"
    assert result.dones.device.type == "cpu"


def test_action_validation_rejects_orders_outside_battle_order_space():
    valid_order = PassBattleOrder()
    battle = cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            valid_orders=([valid_order], []),
        ),
    )

    order = action_to_single_order(
        0,
        battle,
        fake=False,
        position=0,
    )
    assert str(order) == str(valid_order)

    battle.valid_orders[0].clear()
    with pytest.raises(ValueError, match="not in action space"):
        action_to_single_order(
            0,
            battle,
            fake=False,
            position=0,
        )


def test_evaluation_harness_falls_back_without_corpus(tmp_path: Path) -> None:
    harness = EvaluationHarness(
        corpus_path=tmp_path / "nonexistent_manifest.json",
        corpus_hash="nonexistent",
        episodes_per_matchup=5,
        seed=123,
        smoke_test=True,
    )
    sources = harness.build_team_sources()
    assert len(sources) == 3
    for key, source in sources.items():
        assert isinstance(source, FixedTeamSource)
        assert harness.category_metadata[key]["fallback"] is True
        # Sampled team should match the default test team
        team = source.sample(harness.rng)
        assert "Pikachu" in team.packed


def test_evaluation_confidence_intervals_and_matchup_serialization_are_deterministic() -> None:
    from p0.evaluation.harness import MatchupResult, hashlib_team, wilson_score_interval

    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    lower, upper = wilson_score_interval(3, 5)
    assert 0.0 < lower < 0.6 < upper < 1.0
    team_hash = hashlib_team("team-data")
    assert team_hash == hashlib_team("team-data")
    result = MatchupResult(
        policy_a="live",
        policy_b="random",
        team_category="seen",
        total_games=5,
        wins_a=3,
        wins_b=2,
        ties=0,
        win_rate_a=0.6,
        confidence_interval_a=(lower, upper),
        per_team_results={team_hash: {"wins": 3, "games": 5, "win_rate": 0.6}},
    )
    serialized = result.to_dict()
    assert serialized["confidence_interval_a"] == [lower, upper]
    assert serialized["per_team_results"][team_hash]["games"] == 5


def test_evaluation_team_source_fallback_is_repeatable_without_a_corpus(tmp_path) -> None:
    first = EvaluationHarness(
        corpus_path=tmp_path / "missing.json",
        corpus_hash="missing",
        episodes_per_matchup=5,
        seed=91,
        smoke_test=True,
    )
    second = EvaluationHarness(
        corpus_path=tmp_path / "missing.json",
        corpus_hash="missing",
        episodes_per_matchup=5,
        seed=91,
        smoke_test=True,
    )
    first_sources = first.build_team_sources()
    second_sources = second.build_team_sources()
    assert tuple(first_sources) == tuple(second_sources)
    for key in first_sources:
        assert (
            first_sources[key].sample(first.rng).packed
            == second_sources[key].sample(second.rng).packed
        )


_SPEC = importlib.util.spec_from_file_location(
    "benchmark_reducer_depth",
    Path(__file__).parents[2] / "bench" / "benchmark_reducer_depth.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
BenchmarkConfig = _MODULE.BenchmarkConfig
run_benchmark = _MODULE.run_benchmark


def _benchmark_config(**overrides):
    values = {
        "device": "cpu",
        "dtype": "float32",
        "seed": 7,
        "warmup": 1,
        "iterations": 1,
        "repeats": 2,
        "batch_size": 1,
        "time_steps": 1,
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 32,
        "deep_reducer_layers": 2,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def _small_policy(reducer_layers: int = 1):
    return build_policy(
        ModelConfig(8, 2, reducer_layers, 32),
        default_runtime_resources(),
    )
