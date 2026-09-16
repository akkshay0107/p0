"""Recovery tests for the shared training files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.training.checkpoint import CheckpointStore
from p0.training.files import training_run


class TestTrainingFiles:
    def test_resume_recovers_history_after_display_write_failure(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "checkpoint.pt"
        metrics_dir = tmp_path / "metrics"
        with training_run(
            store, checkpoint, metrics_dir, trainer_kind="ppo", settings={"gamma": 0.9}
        ) as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(1, policy, metadata={"gamma": 0.9})
            (metrics_dir / "metrics.json").unlink()
            (metrics_dir / "metrics.json").mkdir()
            run.record(2, {"loss": 0.25}, {"train": {"loss": 0.25}})
            with pytest.raises(OSError):
                run.save(2, policy, metadata={"gamma": 0.9})
        parent_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (metrics_dir / "metrics.json").rmdir()
        with training_run(
            store,
            checkpoint,
            metrics_dir,
            trainer_kind="ppo",
            settings={"gamma": 0.9},
            source_path=checkpoint,
            resume=True,
        ) as run:
            assert run.source is not None
            assert store.load_episode(run.source) == 2
            run.start(2)
            history = json.loads((metrics_dir / "metrics.json").read_text())["metrics"]
            assert [record["loss"] for record in history] == [0.5, 0.25]
            run.record(3, {"loss": 0.125}, {"train": {"loss": 0.125}})
            run.save(3, policy, metadata={"gamma": 0.9})
        details = store.load_metadata(checkpoint)["run"]
        assert details["parent"]["sha256"] == parent_hash
        assert "parent" not in details["parent"]
        assert [
            record["step"]
            for record in json.loads((metrics_dir / "metrics.json").read_text())["metrics"]
        ] == [1, 2, 3]

    def test_output_lock_prevents_a_second_writer(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        checkpoint = tmp_path / "checkpoint.pt"
        with training_run(store, checkpoint, tmp_path, trainer_kind="bc", settings={}):
            with pytest.raises(ValueError, match="already in use"):
                with training_run(store, checkpoint, tmp_path, trainer_kind="bc", settings={}):
                    pytest.fail("A second writer acquired the output")

    def test_selected_policy_is_frozen_and_portable(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "first" / "checkpoint.pt"
        with training_run(
            store, checkpoint, checkpoint.parent, trainer_kind="bc", settings={}
        ) as run:
            run.start(0)
            run.state["best_policy"] = store.snapshot_policy(policy, {"selected_epoch": 1})
            before = next(iter(run.state["best_policy"]["model_state_dict"].values())).clone()
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.add_(1)
            run.record(1, {"loss": 1.0}, {"train": {"loss": 1.0}})
            run.save(1, policy, metadata={})
        (checkpoint.parent / "bc_best_policy.pt").unlink()
        moved = tmp_path / "moved.pt"
        moved.write_bytes(checkpoint.read_bytes())
        destination = tmp_path / "second" / "checkpoint.pt"
        with training_run(
            store,
            destination,
            destination.parent,
            trainer_kind="bc",
            settings={},
            source_path=moved,
            resume=True,
        ) as run:
            run.start(1)
        best = store.read(destination.parent / "bc_best_policy.pt")
        torch.testing.assert_close(next(iter(best.artifact["model_state_dict"].values())), before)
        assert (
            json.loads((destination.parent / "metrics.json").read_text())["metrics"][0]["loss"]
            == 1.0
        )

    def test_resume_cannot_overwrite_another_metrics_directory(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "checkpoint.pt"
        metrics_dir = tmp_path / "metrics"
        with training_run(store, checkpoint, metrics_dir, trainer_kind="ppo", settings={}) as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(1, policy, metadata={})
        other = tmp_path / "other"
        other.mkdir()
        sentinel = other / "metrics.json"
        sentinel.write_text('{"run_id":"another-run","metrics":[{"step":1,"loss":7}]}')
        original = sentinel.read_bytes()
        with pytest.raises(ValueError, match="fresh output directory"):
            with training_run(
                store,
                checkpoint,
                other,
                trainer_kind="ppo",
                settings={},
                source_path=checkpoint,
                resume=True,
            ):
                pytest.fail("A resume claimed another run's files")
        assert sentinel.read_bytes() == original

    def test_portable_resume_rebuilds_tensorboard_history(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "first" / "checkpoint.pt"
        with training_run(
            store, checkpoint, checkpoint.parent, trainer_kind="ppo", settings={}
        ) as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(1, policy, metadata={})
        destination = tmp_path / "second" / "checkpoint.pt"
        with training_run(
            store,
            destination,
            destination.parent,
            trainer_kind="ppo",
            settings={},
            source_path=checkpoint,
            resume=True,
        ) as run:
            run.start(1)
            run.record(2, {"loss": 0.25}, {"train": {"loss": 0.25}})
            run.save(2, policy, metadata={})
        events = EventAccumulator(str(destination.parent / "tensorboard")).Reload()
        assert [(event.step, event.value) for event in events.Scalars("train/loss")] == [
            (1, 0.5),
            (2, 0.25),
        ]

    def test_invalid_saved_board_does_not_purge_existing_events(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "checkpoint.pt"
        with training_run(store, checkpoint, tmp_path, trainer_kind="ppo", settings={}) as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(1, policy, metadata={})
        artifact = torch.load(checkpoint, weights_only=True)
        artifact["training_state"]["run"]["metrics"][0]["board"] = None
        torch.save(artifact, checkpoint)
        with training_run(
            store,
            checkpoint,
            tmp_path,
            trainer_kind="ppo",
            settings={},
            source_path=checkpoint,
            resume=True,
        ) as run:
            with pytest.raises(ValueError, match="board metrics"):
                run.start(1)
        events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
        assert [(event.step, event.value) for event in events.Scalars("train/loss")] == [(1, 0.5)]
