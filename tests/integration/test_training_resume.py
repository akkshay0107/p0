"""A bounded real PPO runner interruption and resume on CPU."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.paths import DEFAULT_PATHS
from p0.training.checkpoint import CheckpointStore
from p0.training.config import GlobalConfig, TeamsConfig, TrainingConfig
from p0.training.files import training_run
from p0.training.magnet import Magnet
from p0.training.ppo import ppo_update
from p0.training.ppo_runner import run_training
from p0.training.trajectory import (
    CollectedTrajectory,
    prepare_trajectory_batches,
)
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
class TestTrainingResume:
    @pytest.mark.integration
    def test_short_rollout_windows_reach_updates_and_resume(self, tmp_path: Path) -> None:
        team = tmp_path / "team.txt"
        team.write_text(DEFAULT_TEST_TEAM)
        store = CheckpointStore()
        initial = tmp_path / "initial.pt"
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        store.save_policy(
            initial,
            policy,
            metadata={"gamma": 0.99, "value_target_semantics": "discounted_terminal_outcome.v1"},
        )
        paths = replace(
            DEFAULT_PATHS,
            initial_policy_checkpoint=initial,
            checkpoint_path=tmp_path / "checkpoints" / "ppo.pt",
            runs_dir=tmp_path / "runs",
        )
        training = TrainingConfig(
            num_episodes=3,
            n_envs=1,
            rollout_steps=1,
            batch_size=256,
            minibatch_size=256,
            ppo_epochs=1,
            enable_optim=False,
            ramp_up_phase=0.34,
            magnet_refresh_interval=1,
        )
        config = GlobalConfig(
            training=training, paths=paths, teams=TeamsConfig(all=team, reduced=team)
        )

        run_training(config)

        saved = store.read(paths.checkpoint_path)
        assert store.load_episode(saved) == 3
        metrics_path = paths.runs_dir / "ppo_training" / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [record["step"] for record in metrics] == [1, 2, 3]
        assert all(record["trajectory_count"] > 0 for record in metrics)

        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed)
        assert store.load_episode(paths.checkpoint_path) == 3

    @pytest.mark.integration
    def test_ppo_resume_continues_training_and_metrics(self, tmp_path: Path) -> None:
        team = tmp_path / "team.txt"
        team.write_text(DEFAULT_TEST_TEAM)
        store = CheckpointStore()
        initial = tmp_path / "initial.pt"
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        store.save_policy(
            initial,
            policy,
            metadata={"gamma": 0.99, "value_target_semantics": "discounted_terminal_outcome.v1"},
        )
        paths = replace(
            DEFAULT_PATHS,
            initial_policy_checkpoint=initial,
            checkpoint_path=tmp_path / "checkpoints" / "ppo.pt",
            runs_dir=tmp_path / "runs",
        )
        training = TrainingConfig(
            num_episodes=3,
            n_envs=1,
            rollout_steps=220,
            batch_size=256,
            minibatch_size=256,
            ppo_epochs=1,
            enable_optim=False,
            ramp_up_phase=0.34,
            magnet_refresh_interval=1,
        )
        config = GlobalConfig(
            training=training, paths=paths, teams=TeamsConfig(all=team, reduced=team)
        )
        checks = 0

        def cancelled() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 280

        run_training(config, cancel_requested=cancelled)
        interrupted = store.read(paths.checkpoint_path)
        completed = store.load_episode(interrupted)
        assert 0 < completed < 3
        metrics_path = paths.runs_dir / "ppo_training" / "metrics.jsonl"
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert history and history[-1]["step"] == completed
        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed)
        final = store.read(paths.checkpoint_path)
        assert store.load_episode(final) == 3
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [record["step"] for record in metrics] == [1, 2, 3]
        assert metrics[: len(history)] == history
        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                interrupted.artifact["model_state_dict"].values(),
                final.artifact["model_state_dict"].values(),
                strict=True,
            )
        )

    @pytest.mark.integration
    def test_cancel_at_initial_team_preview_saves_a_resumable_checkpoint(
        self, tmp_path: Path
    ) -> None:
        team = tmp_path / "team.txt"
        team.write_text(DEFAULT_TEST_TEAM)
        store = CheckpointStore()
        initial = tmp_path / "initial.pt"
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        store.save_policy(
            initial,
            policy,
            metadata={"gamma": 0.99, "value_target_semantics": "discounted_terminal_outcome.v1"},
        )
        paths = replace(
            DEFAULT_PATHS,
            initial_policy_checkpoint=initial,
            checkpoint_path=tmp_path / "checkpoints" / "ppo.pt",
            runs_dir=tmp_path / "runs",
        )
        config = GlobalConfig(
            training=TrainingConfig(n_envs=1, enable_optim=False),
            paths=paths,
            teams=TeamsConfig(all=team, reduced=team),
        )
        run_training(config, cancel_requested=lambda: True)
        saved = store.read(paths.checkpoint_path)
        assert store.load_episode(saved) == 0
        state = saved.artifact["training_state"]
        assert "optimizer_state_dict" in state
        assert "magnet_state_dict" in state
        assert not (paths.runs_dir / "ppo_training" / "metrics.jsonl").exists()
        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed, cancel_requested=lambda: True)
        assert store.load_episode(paths.checkpoint_path) == 0

    @pytest.mark.integration
    def test_invalid_minibatch_does_not_poison_saved_success(self, tmp_path: Path) -> None:
        torch.manual_seed(23)
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        initial_weights = {name: value.clone() for name, value in policy.state_dict().items()}
        collected = [
            CollectedTrajectory(
                observations=StructuredObservation.empty_batch(1),
                action_masks=torch.ones((1, 2, 49), dtype=torch.bool),
                actions=torch.tensor([[7, 8]], dtype=torch.long),
                log_probs=torch.tensor([float("nan") if index == 0 else 0.0]),
                values=torch.zeros(1),
                rewards=torch.ones(1),
                dones=torch.ones(1),
                length=1,
                bootstrap_value=0.0,
                series_history=(),
            )
            for index in range(2)
        ]
        prepared = prepare_trajectory_batches(
            collected, torch.device("cpu"), gamma=0.99, gae_lambda=0.95
        )
        stats = ppo_update(
            prepared,
            policy,
            Magnet(policy),
            torch.optim.SGD(policy.parameters(), lr=1e-3),
            torch.amp.GradScaler("cpu", enabled=False),
            TrainingConfig(
                num_episodes=20,
                n_envs=1,
                rollout_steps=1,
                batch_size=1,
                minibatch_size=1,
                ppo_epochs=1,
                target_kl=1.0e9,
                enable_optim=False,
            ),
            episode=0,
            alpha=0.0,
        )
        assert stats["optimizer_updates"] == 1
        assert all(math.isfinite(value) for value in stats.values())
        assert any(
            not torch.equal(initial_weights[name], value)
            for name, value in policy.state_dict().items()
        )

        checkpoint = tmp_path / "checkpoint.pt"
        store = CheckpointStore()
        optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
        with training_run(store, checkpoint, tmp_path / "metrics", trainer_kind="ppo") as run:
            run.start(0)
            run.record(1, {}, {"train": {"policy_loss": stats["policy_loss"]}})
            run.save(
                1,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )
        with training_run(
            store,
            checkpoint,
            tmp_path / "metrics",
            trainer_kind="ppo",
            source_path=checkpoint,
            resume=True,
        ) as resumed:
            resumed.start(1)
        restored = store.load_policy(checkpoint, torch.device("cpu"))
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
