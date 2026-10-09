"""Tests for the shared training output files."""

from __future__ import annotations

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


class TestMetricsOnResume:
    def test_resume_in_place_drops_metrics_recorded_after_the_checkpoint(
        self, tmp_path: Path
    ) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        checkpoint = tmp_path / "checkpoint.pt"
        metrics_dir = tmp_path / "metrics"
        with training_run(store, checkpoint, metrics_dir, trainer_kind="ppo") as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.record(2, {"loss": 0.25}, {"train": {"loss": 0.25}})
            run.save(
                2,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )
            # The process stops after recording step 3 but before saving it.
            run.record(3, {"loss": 9.0}, {"train": {"loss": 9.0}})

        with training_run(
            store, checkpoint, metrics_dir, trainer_kind="ppo", source_path=checkpoint, resume=True
        ) as run:
            assert run.source is not None
            assert store.load_episode(run.source) == 2
            run.start(2)
            assert [
                (record["step"], record["loss"])
                for record in map(json.loads, run.metrics_path.read_text().splitlines())
            ] == [(1, 0.5), (2, 0.25)]
            run.record(3, {"loss": 0.125}, {"train": {"loss": 0.125}})
            run.save(
                3,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )

        assert [
            (record["step"], record["loss"])
            for record in map(json.loads, (metrics_dir / "metrics.jsonl").read_text().splitlines())
        ] == [(1, 0.5), (2, 0.25), (3, 0.125)]
        assert store.load_episode(checkpoint) == 3
        events = EventAccumulator(str(metrics_dir / "tensorboard")).Reload()
        assert [(event.step, event.value) for event in events.Scalars("train/loss")] == [
            (1, 0.5),
            (2, 0.25),
            (3, 0.125),
        ]

    def test_resume_ignores_a_partial_last_metrics_line(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        checkpoint = tmp_path / "checkpoint.pt"
        with training_run(store, checkpoint, tmp_path, trainer_kind="ppo") as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(
                1,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )
        with (tmp_path / "metrics.jsonl").open("ab") as stream:
            stream.write(b'{"loss": 0.2, "st')

        with training_run(
            store, checkpoint, tmp_path, trainer_kind="ppo", source_path=checkpoint, resume=True
        ) as run:
            run.start(1)
            run.record(2, {"loss": 0.25}, {"train": {"loss": 0.25}})

        assert [
            (record["step"], record["loss"])
            for record in map(json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines())
        ] == [(1, 0.5), (2, 0.25)]

    def test_resume_into_a_fresh_directory_starts_new_metrics(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        checkpoint = tmp_path / "first" / "checkpoint.pt"
        with training_run(store, checkpoint, checkpoint.parent, trainer_kind="ppo") as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(
                1,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )
        saved = checkpoint.read_bytes()

        destination = tmp_path / "second" / "checkpoint.pt"
        with training_run(
            store,
            destination,
            destination.parent,
            trainer_kind="ppo",
            source_path=checkpoint,
            resume=True,
        ) as run:
            run.start(1)
            run.record(2, {"loss": 0.25}, {"train": {"loss": 0.25}})
            run.save(
                2,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )

        assert [
            (record["step"], record["loss"])
            for record in map(
                json.loads, (destination.parent / "metrics.jsonl").read_text().splitlines()
            )
        ] == [(2, 0.25)]
        assert store.load_episode(destination) == 2
        assert checkpoint.read_bytes() == saved


class TestOutputReuse:
    def test_output_lock_prevents_a_second_writer(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        checkpoint = tmp_path / "checkpoint.pt"
        with training_run(store, checkpoint, tmp_path, trainer_kind="bc"):
            with pytest.raises(ValueError, match="already in use"):
                with training_run(store, checkpoint, tmp_path, trainer_kind="bc"):
                    pytest.fail("A second writer acquired the output")

    def test_only_an_in_place_resume_may_reuse_existing_output(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        checkpoint = tmp_path / "first" / "checkpoint.pt"
        with training_run(store, checkpoint, checkpoint.parent, trainer_kind="ppo") as run:
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(
                1,
                policy,
                optimizer=optimizer,
                metadata={},
                scaler=torch.amp.GradScaler("cpu", enabled=False),
            )
        saved = checkpoint.read_bytes()
        metrics = (checkpoint.parent / "metrics.jsonl").read_bytes()

        with pytest.raises(ValueError, match="fresh output directory"):
            with training_run(store, checkpoint, checkpoint.parent, trainer_kind="ppo"):
                pytest.fail("A new run claimed an existing checkpoint")

        elsewhere = tmp_path / "second" / "checkpoint.pt"
        with pytest.raises(ValueError, match="fresh output directory"):
            with training_run(
                store,
                elsewhere,
                checkpoint.parent,
                trainer_kind="ppo",
                source_path=checkpoint,
                resume=True,
            ):
                pytest.fail("A resume to another path claimed existing metrics")

        assert checkpoint.read_bytes() == saved
        assert (checkpoint.parent / "metrics.jsonl").read_bytes() == metrics
