"""A bounded real PPO runner interruption and resume on CPU."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.paths import DEFAULT_PATHS
from p0.training.checkpoint import CheckpointStore
from p0.training.config import GlobalConfig, TeamsConfig, TrainingConfig
from p0.training.ppo_runner import run_training
from tests.team_fixtures import DEFAULT_TEST_TEAM


class TestTrainingResume:
    @pytest.mark.integration
    def test_ppo_resume_preserves_history_and_parent(self, tmp_path: Path) -> None:
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
        history = interrupted.artifact["training_state"]["run"]["metrics"]
        assert history and history[-1]["step"] == completed
        parent_hash = hashlib.sha256(paths.checkpoint_path.read_bytes()).hexdigest()
        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed)
        final = store.read(paths.checkpoint_path)
        assert store.load_episode(final) == 3
        assert final.artifact["provenance"]["run"]["parent"]["sha256"] == parent_hash
        metrics = json.loads((paths.runs_dir / "ppo_training" / "metrics.json").read_text())[
            "metrics"
        ]
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
        assert state["run"]["metrics"] == []
        assert saved.artifact["provenance"]["environment_state"][0]["series_games_played"] == 1
        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed, cancel_requested=lambda: True)
        assert store.load_episode(paths.checkpoint_path) == 0
