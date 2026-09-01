from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from p0.evaluation.harness import (
    EvaluationHarness,
    MatchupResult,
    hashlib_team,
    wilson_score_interval,
)
from p0.format_config import FORMAT, current_manifest
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.teams.corpus import CorpusEntry, CorpusSplit, TeamCorpusManifest, corpus_content_hash
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FixedTeamSource
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import compute_ppo_objective, ppo_update
from p0.training.rollout import BattleMemoryBuffer
from p0.training.trajectory import (
    CollectedTrajectory,
    TrajectoryStorage,
    compute_gae_batch,
    prepare_trajectory_batches,
)
from p0.training.utils import select_optimization_precision


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


def test_battle_memory_keeps_the_whole_game_but_windows_the_reducer_inputs():
    """Verify BattleMemoryBuffer retains all game tokens for series summary while windowing reducer inputs to HISTORY_WINDOW."""
    memory = BattleMemoryBuffer(1, d_model=1)
    env_ids = torch.tensor([0])
    total = HISTORY_WINDOW + 5
    for step in range(total):
        memory.append(env_ids, torch.full((1, 1), float(step)))

    # The end-of-game series summary compresses every decision.
    assert memory.step_counts.tolist() == [total]

    history, mask = memory.inputs(env_ids, torch.device("cpu"), torch.float32)
    # Reducer input shape is restricted to HISTORY_WINDOW
    assert history.shape == (1, HISTORY_WINDOW, 1)
    assert bool(mask.all())
    assert history[0, -1, 0].item() == float(total - 1)
    assert history[0, 0, 0].item() == float(total - HISTORY_WINDOW)


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

    history, mask = memory.inputs(
        torch.tensor([0, 1]),
        torch.device("cpu"),
        torch.float32,
    )

    assert mask.sum(dim=1).tolist() == [1, 2]
    assert history[0, -1, 0].item() == 1.0
    assert history[1, -2:, 0].tolist() == [10.0, 11.0]

    memory.reset(0)
    assert memory.full_values(0) is None
    second_history = memory.full_values(1)
    assert second_history is not None
    assert second_history[0, :, 0].tolist() == [10.0, 11.0]


def test_storage_allocates_completes_and_resets_one_environment():
    """Verify TrajectoryStorage allocation, completion slicing, and reset for a single environment index."""
    storage = TrajectoryStorage.allocate(2, 3)
    storage.step_counts[1] = 2
    storage.actions[1, :2] = 7
    completed = storage.complete(1, 0.0, ())
    assert completed.length == 2
    assert torch.all(completed.actions == 7)
    assert storage.step_counts.tolist() == [0, 0]


def test_storage_reports_explicit_overflow():
    """Verify TrajectoryStorage raises OverflowError when environment step count exceeds capacity."""
    storage = TrajectoryStorage.allocate(1, 1)
    storage.step_counts[0] = 1
    with pytest.raises(OverflowError, match="exceeded"):
        storage.ensure_capacity(torch.tensor([0]))


def test_completed_batch_prepares_returns_advantages_and_chunks():
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
    prepared = prepare_trajectory_batches([batch], torch.device("cpu"), gamma=0.99, gae_lambda=0.95)
    assert prepared[0].returns.shape == (3,)
    assert prepared[0].advantages.shape == (3,)


def test_completed_batch_only_moves_ppo_inputs_to_target_device():
    """Verify device transfer moves only policy gradient computation tensors to GPU while retaining tracking metrics on CPU."""
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
    assert result.values.device.type == "cpu"
    assert result.rewards.device.type == "cpu"
    assert result.dones.device.type == "cpu"


def test_preparing_no_trajectories_is_a_noop() -> None:
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


def test_ppo_recomputes_prior_game_series_context_with_live_gradients() -> None:
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


def test_evaluation_harness_falls_back_without_corpus_repeatably(tmp_path: Path) -> None:
    """Verify EvaluationHarness falls back to deterministic built-in team pools when corpus file is missing."""
    first = EvaluationHarness(
        teams_path=tmp_path / "missing",
        episodes_per_matchup=5,
        seed=91,
        smoke_test=True,
    )
    second = EvaluationHarness(
        teams_path=tmp_path / "missing",
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


def test_evaluation_harness_accepts_regular_manifest_for_bo3(tmp_path: Path) -> None:
    entries = tuple(
        CorpusEntry(
            canonical_hash=hashlib.sha256(f"canonical-{split}".encode()).hexdigest(),
            packed=f"packed-team-{split}",
            packed_sha256=hashlib.sha256(f"packed-team-{split}".encode()).hexdigest(),
            split=split,
            usage_count=1,
        )
        for split in (CorpusSplit.TRAIN, CorpusSplit.VALIDATION, CorpusSplit.TEST)
    )
    manifest = TeamCorpusManifest(
        global_contract_sha256=current_manifest().global_sha256,
        format_id=FORMAT.battle_format,
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-08-19T00:00:00Z",
        sampling_metadata={},
    )
    pool_dir = tmp_path / "all"
    pool_dir.mkdir()
    (pool_dir / "corpus_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    harness = EvaluationHarness(teams_path=pool_dir, format_id=FORMAT.bo3_format)
    sources = harness.build_team_sources()
    assert set(sources) == {"seen", "validation_unseen_canonical", "test_unseen_canonical"}
    assert all(isinstance(source, CorpusTeamSource) for source in sources.values())


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


def test_pure_ppo_objective_uses_symmetric_clipping() -> None:
    """Verify conventional PPO ratio clipping and value loss."""
    config = TrainingConfig(
        clip_range=0.2,
        entropy_coef=0.0,
    )
    total, policy, value, ratio, log_ratio = compute_ppo_objective(
        torch.log(torch.tensor([2.0, 0.5])),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2),
        torch.ones(2),
        config,
        alpha=0.1,
    )
    assert total.shape == policy.shape == value.shape == ratio.shape == log_ratio.shape == (2,)
    assert ratio.tolist() == pytest.approx([2.0, 0.5])
    # For element 0: ratio=2.0, clipped to 1.2, adv=1.0 -> policy_loss = -1.2
    # value_loss = (0 - 1)^2 = 1.0 -> total = 0.5 * 1.0 - 1.2 = -0.7
    # For element 1: ratio=0.5, clipped to 0.8, adv=1.0 -> policy_loss = -0.5 (min of unclipped=0.5, clipped=0.8)
    # value_loss = (1 - 1)^2 = 0.0 -> total = -0.5 (no team preview scaling)
    assert policy[0].item() == pytest.approx(-1.2)
    assert policy[1].item() == pytest.approx(-0.5)
    assert total[0].item() == pytest.approx(config.value_coef * 1.0 - 1.2)
    assert total[1].item() == pytest.approx(-0.5)


def test_ppo_objective_matches_simple_reference() -> None:
    """Verify symmetric PPO clipping, value loss, MMD penalty, and entropy bonus."""
    config = TrainingConfig(
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.2,
    )
    batch_size = 64
    generator = torch.Generator().manual_seed(20260807)
    current_log_probs = torch.randn(batch_size, generator=generator)
    old_log_probs = torch.randn(batch_size, generator=generator)
    advantages = torch.randn(batch_size, generator=generator)
    values = torch.randn(batch_size, generator=generator)
    returns = torch.randn(batch_size, generator=generator)
    entropy = torch.rand(batch_size, generator=generator)
    kl = torch.rand(batch_size, generator=generator)
    total, policy, value, ratio, log_ratio = compute_ppo_objective(
        current_log_probs,
        values,
        entropy,
        kl,
        old_log_probs,
        advantages,
        returns,
        config,
        alpha=0.3,
    )
    expected_ratio = torch.exp(current_log_probs - old_log_probs)
    expected_clipped = torch.clamp(
        expected_ratio,
        1.0 - config.clip_range,
        1.0 + config.clip_range,
    )
    expected_policy = -torch.minimum(
        expected_ratio * advantages,
        expected_clipped * advantages,
    )
    expected_value = (values - returns).square()
    expected_total = expected_policy + config.value_coef * expected_value + 0.3 * kl
    expected_total = expected_total - config.entropy_coef * entropy
    torch.testing.assert_close(ratio, expected_ratio)
    torch.testing.assert_close(log_ratio, current_log_probs - old_log_probs)
    torch.testing.assert_close(policy, expected_policy)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(total, expected_total)


def test_optimization_precision_waterfall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify BF16, FP16, and FP32 are selected in preference order."""
    assert select_optimization_precision(True, torch.device("cpu")) == (
        torch.float32,
        False,
        False,
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert select_optimization_precision(True, torch.device("cuda")) == (
        torch.bfloat16,
        True,
        False,
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert select_optimization_precision(True, torch.device("cuda")) == (
        torch.float16,
        True,
        True,
    )
    assert select_optimization_precision(False, torch.device("cuda")) == (
        torch.float32,
        False,
        False,
    )
