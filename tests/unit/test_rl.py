from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle
from torch.amp import GradScaler

from p0.battle.actions import ACT_SIZE
from p0.evaluation.harness import (
    EvaluationHarness,
    MatchupResult,
    hashlib_team,
    wilson_score_interval,
)
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import ActOutput
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation, TokenType
from p0.model.token_store import SeriesTokenStore
from p0.rl_player import RLPlayer, _LiveBattleHistory
from p0.runtime.env import MegaEnv, SimEnv
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import FixedTeamSource
from p0.training import ppo as ppo_module
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import _run_batched_ppo, compute_ppo_objective
from p0.training.rollout import (
    BattleMemoryBuffer,
    collect_rollouts,
)
from p0.training.trainer import PPOTrainer
from p0.training.trajectory import (
    TrajectoryBatch,
    TrajectoryStorage,
    compute_gae_batch,
    prepare_trajectory_batches,
)
from p0.training.utils import amp_enabled
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
    """Verify ThreadVecEnv returns a tri-state done status: 0 (running), 1 (terminated), 2 (truncated for bootstrapping).
    
    Distinguishing truncation (2) from termination (1) is vital: truncated episodes bootstrap
    value targets off the next state rather than zeroing out subsequent returns.
    """
    vec_env = ThreadVecEnv(cast(Any, [StatusEnv(terminated=terminated, truncated=truncated)]))
    try:
        done_status = vec_env.step([{}])[4]
        assert done_status.dtype == np.int64
        assert done_status.tolist() == [expected]
    finally:
        vec_env.shutdown()


def test_thread_vec_env_binds_each_env_to_its_preallocated_rows():
    """Verify ThreadVecEnv sets memory pointers of worker sub-environments to pre-allocated batch buffer rows."""
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
    """Verify step() raises ValueError if supplied action list does not match n_envs."""
    vec_env = ThreadVecEnv(cast(Any, [BufferBindingEnv(), BufferBindingEnv()]))
    try:
        with pytest.raises(ValueError, match="Number of actions"):
            vec_env.step([{}])
    finally:
        vec_env.shutdown()


def test_compute_gae_batch_matches_single_episode_reference():
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


def test_collect_rollouts_records_both_self_play_streams():
    """Verify collect_rollouts simultaneously records trajectories for agent 1 and agent 2 across parallel environments."""
    config = TrainingConfig(n_envs=3, rollout_steps=1)
    vec_env = FakeVecEnv(config.n_envs)
    policy = FakePolicy(action=7)
    buffer = []
    trajectories1 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    trajectories2 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    memory1 = BattleMemoryBuffer(config.n_envs, 1)
    memory2 = BattleMemoryBuffer(config.n_envs, 1)

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

    # 3 environments x 2 agent perspectives = 6 recorded episodes
    assert len(buffer) == 2 * config.n_envs
    assert all(torch.all(episode.actions == 7) for episode in buffer)
    assert all(float(episode.dones[-1]) == 1.0 for episode in buffer)
    assert all(episode.bootstrap_value == 0.0 for episode in buffer)
    assert trajectories1.step_counts.tolist() == [0, 0, 0]
    assert trajectories2.step_counts.tolist() == [0, 0, 0]
    assert not memory1.step_counts.any()
    assert not memory2.step_counts.any()
    side_two_actions = [
        actions[f"agent2-{env_id}"] for env_id, actions in enumerate(vec_env.received_actions[0])
    ]
    assert all(action.tolist() == [7, 7] for action in side_two_actions)
    assert policy.batch_sizes == [2 * config.n_envs]


def test_truncated_rollout_bootstraps_instead_of_ending_the_game():
    """Verify truncated rollouts set terminal done flag to 0.0 and record bootstrap values for non-terminal returns."""
    config = TrainingConfig(n_envs=3, rollout_steps=1)
    vec_env = FakeVecEnv(config.n_envs, done_status=2)
    policy = FakePolicy(action=7)
    buffer: list[TrajectoryBatch] = []
    trajectories1 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)
    trajectories2 = TrajectoryStorage.allocate(config.n_envs, max_steps=4, d_model=1)

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
    """Verify BattleMemoryBuffer retains all game tokens for series summary while windowing reducer inputs to HISTORY_WINDOW."""
    memory = BattleMemoryBuffer(1, d_model=1)
    env_ids = torch.tensor([0])
    total = HISTORY_WINDOW + 5
    for step in range(total):
        memory.append(env_ids, torch.full((1, 1), float(step)))

    # The end-of-game series summary compresses every decision.
    assert memory.step_counts.tolist() == [total]

    history, mask, ages = memory.inputs(env_ids, torch.device("cpu"), torch.float32)
    # Reducer input shape is restricted to HISTORY_WINDOW
    assert history.shape == (1, HISTORY_WINDOW, 1)
    assert bool(mask.all())
    assert history[0, -1, 0].item() == float(total - 1)
    assert history[0, 0, 0].item() == float(total - HISTORY_WINDOW)
    assert ages[0, -1].item() == 0


def test_battle_memory_reports_explicit_overflow():
    """Verify BattleMemoryBuffer raises OverflowError if step count exceeds preallocated max_steps."""
    memory = BattleMemoryBuffer(1, d_model=1, max_steps=1)
    env_ids = torch.tensor([0])
    memory.append(env_ids, torch.ones((1, 1)))

    with pytest.raises(OverflowError, match="exceeded"):
        memory.append(env_ids, torch.ones((1, 1)))


def test_battle_memory_gathers_independent_environment_windows():
    """Verify BattleMemoryBuffer slices independent variable-length histories across distinct environment rows."""
    memory = BattleMemoryBuffer(2, d_model=1, max_steps=3)
    memory.append(torch.tensor([0, 1]), torch.tensor([[1.0], [10.0]]))
    memory.append(torch.tensor([1]), torch.tensor([[11.0]]))

    history, mask, ages = memory.inputs(
        torch.tensor([0, 1]),
        torch.device("cpu"),
        torch.float32,
    )

    assert mask.sum(dim=1).tolist() == [1, 2]
    assert history[0, -1, 0].item() == 1.0
    assert history[1, -2:, 0].tolist() == [10.0, 11.0]
    assert ages[0, -1].item() == 0
    assert ages[1, -2:].tolist() == [1, 0]

    memory.reset(0)
    assert memory.full_values(0) is None
    second_history = memory.full_values(1)
    assert second_history is not None
    assert second_history[0, :, 0].tolist() == [10.0, 11.0]


def test_live_player_keeps_cpu_history_in_one_list():
    """Verify RLPlayer on CPU devices retains full battle history in contiguous memory without splitting."""
    player = cast(Any, RLPlayer.__new__(RLPlayer))
    player.policy = SimpleNamespace(device=torch.device("cpu"), d_model=1)
    player._memory_model_id = id(player.policy)
    player._empty_history_tensor = torch.zeros((1, 0, 1))
    player._battle_histories = {}
    player._series_store = SeriesTokenStore(1)

    battle = SimpleNamespace(battle_tag="battle-1")
    total = 3 * HISTORY_WINDOW + 1
    for step in range(total):
        player._append_history(battle, torch.tensor([float(step)]))

    history = player._battle_histories["battle-1"]
    assert not history.spill_to_cpu
    assert history.decision_count == total
    assert not history.cpu_chunks
    assert len(history.resident_tokens) == total
    assert history.complete_values(torch.device("cpu"))[0, :, 0].tolist() == list(range(total))
    memory = player._memory_inputs(battle)
    assert memory.history_tokens[0, -1, 0].item() == float(total - 1)
    assert memory.history_tokens[0, 0, 0].item() == float(total - HISTORY_WINDOW)


def test_live_player_spills_device_history_in_fixed_windows():
    """Verify _LiveBattleHistory spills older GPU tokens to CPU chunks when history exceeds 2 * HISTORY_WINDOW."""
    history = _LiveBattleHistory(spill_to_cpu=True)
    device = torch.device("cpu")
    capacity = 2 * HISTORY_WINDOW
    for step in range(capacity):
        history.append(torch.tensor([float(step)]), device)

    assert history.decision_count == capacity
    assert not history.cpu_chunks
    assert len(history.resident_tokens) == capacity

    history.append(torch.tensor([float(capacity)]), device)

    total = capacity + 1
    assert history.decision_count == total
    assert [chunk.size(0) for chunk in history.cpu_chunks] == [HISTORY_WINDOW]
    assert len(history.resident_tokens) == HISTORY_WINDOW + 1
    assert history.cpu_chunks[0][:, 0].tolist() == list(range(HISTORY_WINDOW))
    assert history.resident_tokens[0].item() == HISTORY_WINDOW
    assert history.complete_values(torch.device("cpu"))[0, :, 0].tolist() == list(range(total))

    for step in range(total, 3 * HISTORY_WINDOW + 1):
        history.append(torch.tensor([float(step)]), device)

    total = 3 * HISTORY_WINDOW + 1
    assert history.decision_count == total
    assert [chunk.size(0) for chunk in history.cpu_chunks] == [HISTORY_WINDOW] * 2
    assert len(history.resident_tokens) == HISTORY_WINDOW + 1
    assert history.complete_values(torch.device("cpu"))[0, :, 0].tolist() == list(range(total))


def test_end_of_game_series_summary_sees_every_decision():
    """Verify series summary tokens compress the complete battle trajectory, not merely the sliding reducer window."""
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

    # The summary compresses the whole battle (HISTORY_WINDOW + 6), not just the reducer window.
    assert policy.summarized_lengths == [HISTORY_WINDOW + 6, HISTORY_WINDOW + 6]


def test_storage_allocates_completes_and_resets_one_environment():
    """Verify TrajectoryStorage allocation, completion slicing, and reset for a single environment index."""
    storage = TrajectoryStorage.allocate(2, 3, d_model=1)
    storage.step_counts[1] = 2
    storage.actions[1, :2] = 7
    completed = storage.complete(1)
    assert completed is not None
    assert completed.length == 2
    assert torch.all(completed.actions == 7)
    assert storage.step_counts.tolist() == [0, 0]


def test_storage_reports_explicit_overflow():
    """Verify TrajectoryStorage raises OverflowError when environment step count exceeds capacity."""
    storage = TrajectoryStorage.allocate(1, 1, d_model=1)
    storage.step_counts[0] = 1
    with pytest.raises(OverflowError, match="exceeded"):
        storage.ensure_capacity(torch.tensor([0]))


def test_completed_batch_prepares_returns_advantages_and_chunks():
    """Verify prepare_trajectory_batches calculates returns and GAE advantages on completed trajectory batches."""
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
    """Verify device transfer moves only policy gradient computation tensors to GPU while retaining tracking metrics on CPU."""
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


def test_evaluation_harness_falls_back_without_corpus_repeatably(tmp_path: Path) -> None:
    """Verify EvaluationHarness falls back to deterministic built-in team pools when corpus file is missing."""
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
    assert len(first_sources) == 3
    assert tuple(first_sources) == tuple(second_sources)
    for key, source in first_sources.items():
        assert isinstance(source, FixedTeamSource)
        assert first.category_metadata[key]["fallback"] is True
        first_team = source.sample(first.rng)
        second_team = second_sources[key].sample(second.rng)
        assert "Pikachu" in first_team.packed
        assert first_team.packed == second_team.packed


def test_evaluation_confidence_intervals_and_matchup_serialization_are_deterministic() -> None:
    """Verify Wilson score confidence interval calculation and matchup result serialization."""
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


_OBSERVATION_BUILDER = ObservationBuilder(default_runtime_resources())


def _make_test_double_battle() -> DoubleBattle:
    logger = logging.getLogger("test")
    logger.setLevel(logging.ERROR)
    battle = DoubleBattle("tag", "user", logger, 9)
    battle._player_role = "p1"
    battle._active_pokemon = {}
    battle._opponent_active_pokemon = {}
    battle._team = {}
    battle._opponent_team = {}
    battle._teampreview = False
    battle._available_switches = [[], []]
    battle._weather = {}
    battle._fields = {}
    battle._turn = 0
    battle._can_mega_evolve = [False, False]
    battle._side_conditions = {}
    battle._opponent_side_conditions = {}
    return battle


def test_sim_env_embed_and_mask_share_one_decision_view(monkeypatch):
    """Verify SimEnv reuses single decision view across both observation embedding and action mask generation."""
    battle = _make_test_double_battle()
    from p0.runtime import poke_env_battle_adapter

    original_decision_view = poke_env_battle_adapter.decision_view
    decision_builds = 0

    def counted_decision_view(current_battle):
        nonlocal decision_builds
        decision_builds += 1
        return original_decision_view(current_battle)

    monkeypatch.setattr(poke_env_battle_adapter, "decision_view", counted_decision_view)
    env = SimEnv.__new__(SimEnv)
    cast(Any, env).agent1 = SimpleNamespace(username=battle.player_username)
    cast(Any, env).agent2 = SimpleNamespace(username="other-player")
    env._observation_builder = _OBSERVATION_BUILDER
    env._battle_view_factory = battle_view
    out1 = StructuredObservation.empty_batch(1)[0]
    out2 = StructuredObservation.empty_batch(1)[0]
    env.set_observation_targets(out1, out2)

    result = env.embed_battle(battle)
    mask = env.get_action_mask(battle)

    assert result is out1
    assert result.token_type_ids[0] == TokenType.POKEMON
    assert len(mask) == FORMAT.action_size * 2
    assert decision_builds == 1


def test_calc_reward_scores_each_seat_without_touching_the_series():
    """Verify calc_reward assigns +1 for win and -1 for loss without mutating series state."""
    env = SimEnv.__new__(SimEnv)
    env._series_scores = [0, 0]
    won = SimpleNamespace(finished=True, won=True, lost=False)
    lost = SimpleNamespace(finished=True, won=False, lost=True)

    assert SimEnv.calc_reward(env, cast(Any, won)) == 1.0
    assert SimEnv.calc_reward(env, cast(Any, lost)) == -1.0
    assert SimEnv.calc_reward(env, cast(Any, SimpleNamespace(finished=False))) == 0.0
    assert env.series_scores == [0, 0]


def _stepping_env(monkeypatch, battle: Any, *, decision_steps: int = 0) -> SimEnv:
    agents = ("agent1", "agent2")
    monkeypatch.setattr(
        MegaEnv,
        "step",
        lambda self, actions: (
            {},
            dict.fromkeys(agents, 0.0),
            {agent: bool(battle.finished and battle.wiped) for agent in agents},
            {agent: bool(battle.finished and not battle.wiped) for agent in agents},
            {},
        ),
    )
    env = SimEnv.__new__(SimEnv)
    env._series_scores = [0, 0]
    env._decision_steps = decision_steps
    cast(Any, env).battle1 = battle
    return env


@pytest.mark.parametrize("wiped", [True, False])
def test_step_treats_every_finished_battle_as_terminal(monkeypatch, wiped: bool):
    """Verify SimEnv.step marks all completed battles as terminated (not truncated)."""
    battle = SimpleNamespace(finished=True, won=True, lost=False, wiped=wiped)
    env = _stepping_env(monkeypatch, battle)

    _, _, terminated, truncated, _ = env.step({})

    assert all(terminated.values())
    assert not any(truncated.values())
    assert env.series_scores == [1, 0]


def test_step_truncates_an_unfinished_game_at_the_decision_cap(monkeypatch):
    """Verify SimEnv.step flags truncation when decision step count reaches cap."""
    battle = SimpleNamespace(finished=False, won=False, lost=False, wiped=False)
    env = _stepping_env(monkeypatch, battle, decision_steps=197)

    _, rewards, terminated, truncated, _ = env.step({})

    assert not any(terminated.values())
    assert all(truncated.values())
    assert all(reward == 0.0 for reward in rewards.values())
    assert env.series_scores == [0, 0]


def test_a_best_of_three_series_resets_once_a_side_wins_twice(monkeypatch):
    """Verify Best of 3 series tracks game wins and rotates series identity + teams once a player achieves 2 wins."""
    monkeypatch.setattr(MegaEnv, "reset", lambda self, seed=None, options=None: "reset")
    env = SimEnv.__new__(SimEnv)
    env._series_scores = [0, 0]
    env._series_games_played = 1
    env._resume_reset_pending = False
    env._agent_rng = random.Random(1)
    env._opponent_rng = random.Random(2)
    env.series_id = "series-1"
    battle = SimpleNamespace(finished=True, won=True, lost=False)

    # Game 1 win -> score [1, 0], advance to game 2
    env._record_game_result(cast(Any, battle))
    env.reset()
    assert env.series_scores == [1, 0]
    assert env.series_games_played == 2

    # Game 2 win -> score [2, 0], series won
    env._record_game_result(cast(Any, battle))
    assert env.series_scores == [2, 0]

    sampled: list[str] = []
    cast(Any, env)._agent_team_source = SimpleNamespace(
        sample=lambda rng: SimpleNamespace(packed="agent-team")
    )
    cast(Any, env)._opponent_team_source = SimpleNamespace(
        sample=lambda rng: SimpleNamespace(packed="opponent-team")
    )
    cast(Any, env).agent1 = SimpleNamespace(update_team=sampled.append)
    cast(Any, env).agent2 = SimpleNamespace(update_team=sampled.append)

    # Reset after series win starts fresh series with new teams
    env.reset()
    assert env.series_scores == [0, 0]
    assert env.series_games_played == 1
    assert env.series_id != "series-1"
    assert sampled == ["agent-team", "opponent-team"]


def test_sim_env_training_state_restores_teams_and_preserves_game_boundary(monkeypatch):
    """Verify SimEnv serialization saves and restores team strings, series scores, and game counters."""
    class TeamBuilder:
        def __init__(self, packed: str):
            self.packed = packed

        def yield_team(self) -> str:
            return self.packed

        def update_team(self, packed: str) -> None:
            self.packed = packed

    class Player:
        def __init__(self, packed: str):
            self._team = TeamBuilder(packed)

        def update_team(self, packed: str) -> None:
            self._team = TeamBuilder(packed)

    env = SimEnv.__new__(SimEnv)
    env._agent_rng = random.Random(10)
    env._opponent_rng = random.Random(11)
    env._series_scores = [1, 0]
    env._series_games_played = 2
    env._decision_steps = 17
    env._resume_reset_pending = False
    env.series_id = "series-1"
    cast(Any, env).agent1 = Player("agent-team")
    cast(Any, env).agent2 = Player("opponent-team")

    state = env.training_state()
    cast(Any, env).agent1.update_team("wrong-agent-team")
    cast(Any, env).agent2.update_team("wrong-opponent-team")
    env._series_scores = [0, 0]
    env._series_games_played = 0

    env.restore_training_state(state)
    monkeypatch.setattr(MegaEnv, "reset", lambda self, seed=None, options=None: "reset")

    assert env.reset() == "reset"
    assert cast(Any, env).agent1._team.yield_team() == "agent-team"
    assert cast(Any, env).agent2._team.yield_team() == "opponent-team"
    assert env.series_scores == [1, 0]
    assert env.series_games_played == 2

    env.reset()
    assert env.series_games_played == 3


def test_pure_ppo_objective_clips_and_weights_team_preview() -> None:
    """Verify PPO loss calculations with probability ratio clipping, value loss, and team preview loss multiplier."""
    config = TrainingConfig(
        clip_low=0.2,
        clip_high=0.2,
        teampreview_loss_mult=2.0,
        teampreview_alpha_mult=3.0,
        residual_entropy_coef=0.0,
    )
    total, policy, value, ratio, log_ratio = compute_ppo_objective(
        torch.log(torch.tensor([2.0, 0.5])),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
        torch.tensor([True, False]),
        config,
        alpha=0.1,
    )
    assert total.shape == policy.shape == value.shape == ratio.shape == log_ratio.shape == (2,)
    assert ratio.tolist() == pytest.approx([2.0, 0.5])
    # For element 0: ratio=2.0, clipped to 1.2, adv=1.0 -> policy_loss = -1.2
    # value_loss = (0 - 1)^2 = 1.0 -> total = 0.5 * 1.0 - 1.2 = -0.7 * 2.0 (team preview) = -1.4
    # For element 1: ratio=0.5, clipped to 0.8, adv=1.0 -> policy_loss = -0.5 (min of unclipped=0.5, clipped=0.8)
    # value_loss = (1 - 1)^2 = 0.0 -> total = -0.5 (no team preview scaling)
    assert policy[0].item() == pytest.approx(-1.2)
    assert policy[1].item() == pytest.approx(-0.5)
    assert total[0].item() == pytest.approx((config.value_coef * 1.0 - 1.2) * 2.0)
    assert total[1].item() == pytest.approx(-0.5)


def test_ppo_amp_is_cuda_only() -> None:
    """Verify mixed precision AMP is active only when CUDA devices are present and enabled in TrainingConfig."""
    config = TrainingConfig(enable_optim=True)

    assert not amp_enabled(config, torch.device("cpu"))
    assert amp_enabled(config, torch.device("cuda"))
    assert not amp_enabled(TrainingConfig(enable_optim=False), torch.device("cuda"))


def test_trainer_cancellation_saves_once_before_collecting(tmp_path: Path) -> None:
    """Verify PPOTrainer saves checkpoint immediately upon cancellation request before collecting new rollouts."""
    saved = []

    class Store:
        def save_training_state(self, path, episode, policy, **kwargs):
            saved.append((path, episode, policy, kwargs))

    collector = SimpleNamespace(vector_env=SimpleNamespace(reset=lambda: None))
    updater = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
        scaler=object(),
    )
    trainer = PPOTrainer(
        policy=cast(Any, object()),
        policy_store=cast(Any, Store()),
        checkpoint_path=tmp_path / "checkpoint.pt",
        collector=cast(Any, collector),
        updater=cast(Any, updater),
        magnet=cast(Any, object()),
        scheduler=cast(Any, object()),
        training_config=TrainingConfig(
            num_episodes=9, magnet_refresh_interval=1, ramp_up_phase=0.5
        ),
        cancel_requested=lambda: True,
    )
    trainer.run()
    assert [(path, episode) for path, episode, _, _ in saved] == [(tmp_path / "checkpoint.pt", 0)]


def test_trainer_saves_final_completed_episode(tmp_path: Path) -> None:
    """Verify PPOTrainer saves final completed episode state upon reaching target num_episodes."""
    saved = []

    class Store:
        def save_training_state(self, path, episode, policy, **kwargs):
            saved.append((path, episode))

    collector = SimpleNamespace(vector_env=SimpleNamespace(reset=lambda: None))
    updater = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
        scaler=object(),
    )
    trainer = PPOTrainer(
        policy=cast(Any, object()),
        policy_store=cast(Any, Store()),
        checkpoint_path=tmp_path / "checkpoint.pt",
        collector=cast(Any, collector),
        updater=cast(Any, updater),
        magnet=cast(Any, object()),
        scheduler=cast(Any, object()),
        training_config=TrainingConfig(
            num_episodes=9, magnet_refresh_interval=1, ramp_up_phase=0.5
        ),
    )

    trainer.run(start_episode=9)

    assert saved == [(tmp_path / "checkpoint.pt", 9)]


def test_ppo_updates_all_policy_paths() -> None:
    """Verify backward pass on batched PPO loss computes non-zero gradients across encoder, actor, and critic parameter groups."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    obs = StructuredObservation.empty_batch(1).to(device)
    episode = TrajectoryBatch(
        observations=obs,
        actions=torch.tensor([[1, 2]], dtype=torch.long, device=device),
        log_probs=torch.zeros(1, device=device),
        advantages=torch.ones(1, device=device),
        returns=torch.ones(1, device=device),
        values=torch.zeros(1, device=device),
        rewards=torch.zeros(1, device=device),
        dones=torch.ones(1, device=device),
        action_masks=torch.ones((1, 2, ACT_SIZE), dtype=torch.bool, device=device),
        length=1,
    )
    config = TrainingConfig()
    magnet = Magnet(policy)

    loss, _, steps = _run_batched_ppo(
        [episode], policy, magnet, config, device, alpha=config.magnet_alpha
    )
    assert steps == 1

    policy.zero_grad(set_to_none=True)
    loss.backward()

    assert any(
        p.grad is not None and torch.abs(p.grad).sum() > 0 for p in policy.encoder.parameters()
    )
    assert any(
        p.grad is not None and torch.abs(p.grad).sum() > 0 for p in policy.actor.parameters()
    )
    assert any(
        p.grad is not None and not torch.all(p.grad == 0) for p in policy.critic.parameters()
    )


def test_ppo_caches_magnet_logits_for_repeated_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Magnet anchor network evaluation is cached across multiple PPO mini-epochs to eliminate redundant forward passes."""
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())
    magnet = Magnet(policy)
    episode = TrajectoryBatch(
        observations=StructuredObservation.empty_batch(1),
        actions=torch.tensor([[1, 2]], dtype=torch.long),
        log_probs=torch.zeros(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        action_masks=torch.ones((1, 2, ACT_SIZE), dtype=torch.bool),
        returns=torch.zeros(1),
        advantages=torch.ones(1),
        length=1,
    )

    calls = 0
    original_evaluate = magnet.policy.evaluate

    def counted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(magnet.policy, "evaluate", counted_evaluate)
    cache: dict[int, torch.Tensor] = {}
    config = TrainingConfig(enable_optim=False)

    _run_batched_ppo(
        [episode],
        policy,
        magnet,
        config,
        policy.device,
        alpha=config.magnet_alpha,
        magnet_cache=cache,
    )
    _run_batched_ppo(
        [episode],
        policy,
        magnet,
        config,
        policy.device,
        alpha=config.magnet_alpha,
        magnet_cache=cache,
    )
    assert calls == 1
    assert list(cache) == [id(episode)]


def test_full_precision_ppo_discards_non_finite_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PPO optimizer skips gradient step and leaves model weights unmodified when gradients contain inf/NaN values."""
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())
    magnet = Magnet(policy)
    parameter = next(policy.parameters())
    episode = TrajectoryBatch(
        observations=StructuredObservation.empty_batch(1),
        actions=torch.tensor([[1, 2]], dtype=torch.long),
        log_probs=torch.zeros(1),
        advantages=torch.ones(1),
        returns=torch.ones(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        action_masks=torch.ones((1, 2, ACT_SIZE), dtype=torch.bool),
        length=1,
        explained_variance=0.0,
    )

    def finite_chunk_with_bad_backward(*args, **kwargs):
        del args, kwargs
        zero = torch.zeros((), device=policy.device)
        metrics = {
            "policy_loss": zero,
            "value_loss": zero,
            "normalized_entropy": zero,
            "magnet_kl": zero,
            "kl_div": zero,
            "clip_frac": zero,
        }
        return parameter.sum() * 0.0, metrics, 1

    monkeypatch.setattr(ppo_module, "_run_batched_ppo", finite_chunk_with_bad_backward)
    gradient_hook = parameter.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}

    try:
        result = ppo_module.ppo_update(
            [episode],
            policy,
            magnet,
            optimizer,
            GradScaler(device="cpu", enabled=False),
            TrainingConfig(batch_size=1, minibatch_size=1, ppo_epochs=1),
            episode=0,
            alpha=0.0,
            cancel_requested=lambda: False,
        )
    finally:
        gradient_hook.remove()

    assert result["grad_norm"] == 0.0
    assert all(torch.equal(before[name], value) for name, value in policy.named_parameters())


def test_ppo_keeps_series_encoder_out_of_the_bo1_graph() -> None:
    """Verify series encoder parameters receive no gradients during single-game (Bo1) PPO rollouts."""
    policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
    observation = StructuredObservation.empty_batch(1)
    action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)
    episode = TrajectoryBatch(
        observations=observation,
        action_masks=action_mask,
        actions=torch.zeros((1, 2), dtype=torch.long),
        log_probs=torch.zeros(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        length=1,
        returns=torch.zeros(1),
        advantages=torch.ones(1),
    )
    loss, _, _ = _run_batched_ppo(
        [episode],
        policy,
        Magnet(policy),
        TrainingConfig(enable_optim=False),
        policy.device,
        alpha=0.0,
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.series.parameters())
