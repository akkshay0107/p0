"""Portable checkpoint export through real files and archives."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from p0.cli.export_training import export_checkpoint
from p0.format_config import sha256_file
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest, write_split_manifest
from p0.teams.factory import build_team_source
from p0.training.checkpoint import CheckpointStore
from p0.training.config import TrainingConfig, load_config
from p0.training.files import training_run
from tests.team_fixtures import DEFAULT_TEST_TEAM
from tests.unit.replay_fixtures import sample_replay_payload


class TestExportTraining:
    def test_ppo_export_preserves_checkpoint_bytes_and_team_inputs(self, tmp_path: Path) -> None:
        team = tmp_path / "team.txt"
        team.write_text(DEFAULT_TEST_TEAM)
        settings = {
            "training": asdict(TrainingConfig()),
            "teams": [dict(build_team_source(team).describe())] * 2,
        }
        store = CheckpointStore()
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        checkpoint = tmp_path / "training" / "checkpoint.pt"
        with training_run(
            store, checkpoint, checkpoint.parent, trainer_kind="ppo", settings=settings
        ) as run:
            run.metadata["inputs"] = {"agent_teams": str(team), "opponent_teams": str(team)}
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(
                1,
                policy,
                metadata={"gamma": 0.99},
                optimizer=torch.optim.AdamW(policy.parameters()),
            )
        output = tmp_path / "export.tar.gz"
        export_checkpoint(checkpoint, output)
        target = tmp_path / "restored"
        with tarfile.open(output) as archive:
            archive.extractall(target, filter="data")
        assert (target / "run/checkpoint.pt").read_bytes() == checkpoint.read_bytes()
        hashes = json.loads((target / "run/checksums.json").read_text())
        for name, digest in hashes.items():
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
        config_path = target / "run/config.yaml"
        config_value = json.loads(config_path.read_text())
        config_value["paths"]["repository_root"] = str(target)
        config_path.write_text(json.dumps(config_value))
        config = load_config(config_path)
        assert dict(build_team_source(config.teams.reduced).describe()) == settings["teams"][0]
        assert config.paths.resume_checkpoint == target / "run/checkpoint.pt"
        saved_archive = output.read_bytes()
        team.write_text("malformed team")
        with pytest.raises(ValueError):
            export_checkpoint(checkpoint, output)
        assert output.read_bytes() == saved_archive

    def test_bc_export_restores_shards_and_split_without_original_paths(
        self, tmp_path: Path
    ) -> None:
        compiled = compile_payloads(
            (sample_replay_payload("export-train", parent="export-series"),)
        )
        built = write_tensor_shards(
            compiled,
            tmp_path / "shards",
            max_decisions_per_shard=8,
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = tmp_path / "splits.json"
        write_split_manifest(
            SeriesSplitManifest(
                built.manifest.global_contract_sha256,
                0,
                {next(iter(built.manifest.source_series)): "train"},
                dataset_hash=built.manifest.dataset_hash,
            ),
            split_path,
        )
        store = CheckpointStore()
        checkpoint = tmp_path / "training" / "checkpoint.pt"
        policy = build_policy(ModelConfig(32, 4, 1, 64), default_runtime_resources())
        with training_run(
            store,
            checkpoint,
            checkpoint.parent,
            trainer_kind="bc",
            settings={"gamma": 0.99, "value_coef": 0.5, "overfit": False},
        ) as run:
            run.metadata["inputs"] = {
                "shard_manifest": str(built.manifest_path),
                "split_manifest": str(split_path),
            }
            run.start(0)
            run.record(1, {"loss": 0.5}, {"train": {"loss": 0.5}})
            run.save(
                1,
                policy,
                metadata={
                    "dataset_hash": built.manifest.dataset_hash,
                    "split_manifest_sha256": sha256_file(split_path),
                    "epoch_budget": 2,
                },
            )
        output = tmp_path / "export.tar.gz"
        export_checkpoint(checkpoint, output)
        target = tmp_path / "restored"
        with tarfile.open(output) as archive:
            archive.extractall(target, filter="data")
        config_path = target / "run/config.yaml"
        value = json.loads(config_path.read_text())
        value["paths"] = {"repository_root": str(target)}
        config_path.write_text(json.dumps(value))
        config = load_config(config_path).bc
        restored = LazyReplayDataset(
            config.shard_manifest, split_manifest=config.split_manifest, verify_hashes=True
        )
        assert restored.manifest.dataset_hash == built.manifest.dataset_hash
        assert len(list(restored.for_split("train"))) == 2
        assert config.epochs == 2
