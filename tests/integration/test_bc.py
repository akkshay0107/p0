"""Integration tests for Behavior Cloning training and checkpoint orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset
from p0.training.bc import BCTrainer
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from tests.unit.replay_fixtures import sample_replay_payload


@pytest.mark.heavy
@pytest.mark.integration
class TestBCTrainingIntegration:
    def test_real_shards_close_all_worker_game_boundaries(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (
                sample_replay_payload("worker-a", parent="worker-series-a"),
                sample_replay_payload("worker-b", parent="worker-series-b"),
            )
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        dataset = LazyReplayDataset(built.manifest_path)
        policy = build_policy(
            ModelConfig(64, 4, 1, 128),
            default_runtime_resources(),
        )
        trainer = BCTrainer(
            policy,
            dataset,
            BCConfig(
                batch_decisions=3,
                max_chunk_size=3,
                num_workers=2,
                learning_rate=1e-3,
                enable_optim=False,
            ),
            device="cpu",
        )

        metrics = trainer.train_epoch()

        assert metrics["decisions"] == 8
        assert metrics["games"] == 4

    def test_replay_to_series_bc_checkpoint_smoke(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (sample_replay_payload("game-1"), sample_replay_payload("game-2"))
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        dataset = LazyReplayDataset(built.manifest_path)
        policy = build_policy(
            ModelConfig(
                d_model=64,
                nhead=4,
                reducer_layers=1,
                dim_feedforward=128,
            ),
            default_runtime_resources(),
        )
        trainer = BCTrainer(
            policy,
            dataset,
            BCConfig(
                batch_decisions=2,
                learning_rate=1e-3,
                epochs=1,
                enable_optim=False,
            ),
            device="cpu",
        )

        metrics = trainer.train()

        assert metrics["decisions"] == 8
        assert metrics["games"] == 4
        assert torch.isfinite(torch.tensor(metrics["overall_nll"]))
        checkpoint = tmp_path / "bc.pt"
        store = CheckpointStore()
        store.save_training(
            checkpoint,
            1,
            trainer.policy,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
            trainer_kind="bc",
        )

        restored = build_policy(trainer.policy.config, default_runtime_resources())
        restored_trainer = BCTrainer(
            restored,
            (),
            trainer.config,
            device="cpu",
        )
        assert (
            store.load_training(
                checkpoint,
                restored_trainer.policy,
                optimizer=restored_trainer.optimizer,
                scaler=restored_trainer.scaler,
                expected_trainer_kind="bc",
                require_training_state=True,
            )
            == 1
        )
        for name, parameter in trainer.policy.state_dict().items():
            torch.testing.assert_close(parameter, restored_trainer.policy.state_dict()[name])
