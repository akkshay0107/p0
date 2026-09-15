"""Tests for behavior-cloning runner and CLI orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import (
    SeriesSplitManifest,
    write_split_manifest,
)
from p0.replays.schema import LabelKind
from p0.training.bc_runner import evaluate_bc, train_bc
from p0.training.config import BCConfig
from tests.unit.replay_fixtures import sample_replay_payload


class TestBCRunner:
    def test_training_rejects_empty_validation_before_saving_a_policy(
        self,
        tmp_path: Path,
    ) -> None:
        result = compile_payloads(
            (sample_replay_payload("empty-validation", parent="only-series"),)
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            max_decisions_per_shard=8,
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = tmp_path / "splits.json"
        only_series = next(iter(built.manifest.source_series))
        write_split_manifest(
            SeriesSplitManifest(
                built.manifest.global_contract_sha256,
                0,
                {only_series: "train"},
                dataset_hash=built.manifest.dataset_hash,
            ),
            split_path,
        )
        config = BCConfig(
            batch_decisions=4,
            max_chunk_size=4,
            epochs=1,
            num_workers=0,
            enable_optim=False,
            shard_manifest=built.manifest_path,
            split_manifest=split_path,
            output_dir=tmp_path / "output",
        )

        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        original_shard = shard_path.read_bytes()
        tampered_shard = bytearray(original_shard)
        tampered_shard[-1] ^= 1
        shard_path.write_bytes(tampered_shard)
        with pytest.raises(ValueError, match="Shard hash mismatch"):
            train_bc(config, device="cpu")
        shard_path.write_bytes(original_shard)

        with pytest.raises(ValueError, match="validation split has no accepted series"):
            train_bc(config, device="cpu")

        assert not list((tmp_path / "output").rglob("bc_best_policy.pt"))

        write_split_manifest(
            SeriesSplitManifest(
                built.manifest.global_contract_sha256,
                0,
                {only_series: "validation"},
                dataset_hash=built.manifest.dataset_hash,
            ),
            split_path,
        )
        with pytest.raises(ValueError, match="train split has no accepted series"):
            train_bc(config, device="cpu")

    def test_overfit_runner_reaches_its_bounded_acceptance_target(self, tmp_path: Path) -> None:
        compiled = compile_payloads(
            (
                sample_replay_payload("overfit-train", parent="overfit-train-series"),
                sample_replay_payload(
                    "overfit-validation",
                    parent="overfit-validation-series",
                ),
            )
        )
        exact_compiled = replace(
            compiled,
            games=tuple(
                replace(
                    game,
                    perspectives=tuple(
                        replace(
                            perspective,
                            decisions=tuple(
                                replace(
                                    decision,
                                    evidence=replace(
                                        decision.evidence,
                                        label_kind=LabelKind.EXACT,
                                        candidates=(decision.evidence.candidates[0],),
                                    ),
                                )
                                for decision in perspective.decisions
                            ),
                        )
                        for perspective in game.perspectives
                    ),
                )
                for game in compiled.games
            ),
        )
        built = write_tensor_shards(
            exact_compiled,
            tmp_path / "shards",
            max_decisions_per_shard=8,
            created_at="2026-01-01T00:00:00Z",
        )
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        split_path = tmp_path / "splits.json"
        write_split_manifest(
            SeriesSplitManifest(
                built.manifest.global_contract_sha256,
                0,
                {
                    series_by_replay["overfit-train"]: "train",
                    series_by_replay["overfit-validation"]: "validation",
                },
                dataset_hash=built.manifest.dataset_hash,
            ),
            split_path,
        )
        config = BCConfig(
            batch_decisions=4,
            max_chunk_size=4,
            learning_rate=1e-3,
            epochs=30,
            num_workers=0,
            enable_optim=False,
            shard_manifest=built.manifest_path,
            split_manifest=split_path,
            output_dir=tmp_path / "output",
        )

        result = train_bc(config, overfit=True, device="cpu")

        assert result["overfit_passed"] is True
        assert 0 < result["completed_epoch"] <= config.epochs
        assert result["final_training"]["overall_nll"] <= (
            result["initial_training"]["overall_nll"] * 0.2
        )
        assert result["final_training"]["exact_joint_accuracy"] >= 0.9

    def test_noop_resume_in_another_output_reports_original_existing_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        result = compile_payloads(
            (
                sample_replay_payload("resume-train", parent="train-series"),
                sample_replay_payload("resume-validation", parent="validation-series"),
            )
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            max_decisions_per_shard=8,
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = tmp_path / "splits.json"
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        write_split_manifest(
            SeriesSplitManifest(
                built.manifest.global_contract_sha256,
                0,
                {
                    series_by_replay["resume-train"]: "train",
                    series_by_replay["resume-validation"]: "validation",
                },
                dataset_hash=built.manifest.dataset_hash,
            ),
            split_path,
        )
        first_config = BCConfig(
            batch_decisions=4,
            max_chunk_size=4,
            epochs=1,
            num_workers=0,
            enable_optim=False,
            shard_manifest=built.manifest_path,
            split_manifest=split_path,
            output_dir=tmp_path / "first-output",
        )
        first = train_bc(first_config, device="cpu")
        first_latest = Path(first["latest_training_checkpoint"])
        first_best = Path(first["best_policy_checkpoint"])
        with pytest.raises(ValueError, match="already contains an experiment"):
            train_bc(first_config, device="cpu")
        occupied_output = tmp_path / "occupied-output" / built.manifest.dataset_hash
        occupied_output.mkdir(parents=True)
        sentinel = occupied_output / "unrelated.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with pytest.raises(ValueError, match="already contains an experiment"):
            train_bc(
                replace(
                    first_config,
                    output_dir=tmp_path / "occupied-output",
                    resume_checkpoint=first_latest,
                ),
                device="cpu",
            )
        assert sentinel.read_text(encoding="utf-8") == "keep"

        resumed = train_bc(
            replace(
                first_config,
                output_dir=tmp_path / "second-output",
                resume_checkpoint=first_latest,
            ),
            device="cpu",
        )

        assert resumed["completed_epoch"] == 1
        assert resumed["latest_training_checkpoint"] == str(first_latest)
        assert resumed["best_policy_checkpoint"] == str(first_best)
        assert resumed["metrics_path"] is None
        assert first_latest.is_file()
        assert first_best.is_file()
        evaluation = evaluate_bc(first_config, first_best, split="validation", device="cpu")
        assert evaluation["objective"] == {
            "gamma": first_config.gamma,
            "value_target_semantics": "discounted_terminal_outcome.v1",
        }
        cli = subprocess.run(
            (
                sys.executable,
                "-m",
                "p0.cli.bc",
                "evaluate",
                "--config",
                str(Path(__file__).parents[2] / "config.example.yaml"),
                "--shard-manifest",
                str(built.manifest_path),
                "--split-manifest",
                str(split_path),
                "--checkpoint",
                str(first_best),
                "--split",
                "validation",
                "--device",
                "cpu",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        cli_evaluation = json.loads(cli.stdout)
        assert cli_evaluation["split"] == "validation"
        assert cli_evaluation["checkpoint"] == str(first_best)
        assert cli_evaluation["metrics"] == evaluation["metrics"]
        with pytest.raises(ValueError, match="provenance field 'gamma' is incompatible"):
            evaluate_bc(
                replace(first_config, gamma=0.5),
                first_best,
                split="validation",
                device="cpu",
            )
        with pytest.raises(ValueError, match="test split has no accepted series"):
            evaluate_bc(first_config, first_best, split="test", device="cpu")

        cancelled = train_bc(
            replace(first_config, output_dir=tmp_path / "cancelled-output"),
            device="cpu",
            cancel_requested=lambda: True,
        )
        assert cancelled["completed_epoch"] == 0
        assert cancelled["cancelled"] is True
        assert cancelled["latest_training_checkpoint"] is None
        assert cancelled["best_policy_checkpoint"] is None
        assert cancelled["metrics_path"] is None

        with pytest.raises(RuntimeError, match="requires at least one exact policy label"):
            train_bc(
                replace(first_config, output_dir=tmp_path / "overfit-output"),
                overfit=True,
                device="cpu",
            )

        continued = train_bc(
            replace(
                first_config,
                epochs=2,
                resume_checkpoint=first_latest,
            ),
            device="cpu",
        )
        assert continued["completed_epoch"] == 2
        control = train_bc(
            replace(
                first_config,
                epochs=2,
                output_dir=tmp_path / "control-output",
            ),
            device="cpu",
        )
        continued_artifact = torch.load(
            Path(continued["latest_training_checkpoint"]),
            weights_only=True,
        )
        control_artifact = torch.load(
            Path(control["latest_training_checkpoint"]),
            weights_only=True,
        )
        torch.testing.assert_close(
            continued_artifact["model_state_dict"],
            control_artifact["model_state_dict"],
        )
        torch.testing.assert_close(
            continued_artifact["training_state"]["optimizer_state_dict"],
            control_artifact["training_state"]["optimizer_state_dict"],
        )

        stale_checkpoint = tmp_path / "stale-selection.pt"
        stale_artifact = torch.load(first_latest, weights_only=True)
        stale_artifact["training_state"]["episode"] = 1
        stale_artifact["provenance"]["selection_state"]["selected_epoch"] = 2
        torch.save(stale_artifact, stale_checkpoint)
        with pytest.raises(ValueError, match="stale selected-policy state"):
            train_bc(
                replace(
                    first_config,
                    output_dir=tmp_path / "stale-output",
                    resume_checkpoint=stale_checkpoint,
                ),
                device="cpu",
            )

        best_bytes = first_best.read_bytes()
        first_best.write_bytes(best_bytes + b"interrupted replacement")
        with pytest.raises(ValueError, match="does not match its saved digest"):
            train_bc(
                replace(
                    first_config,
                    output_dir=tmp_path / "digest-output",
                    resume_checkpoint=first_latest,
                ),
                device="cpu",
            )
        first_best.write_bytes(best_bytes)

        first_best.unlink()
        with pytest.raises(ValueError, match="selected policy artifact is missing"):
            train_bc(
                replace(
                    first_config,
                    output_dir=tmp_path / "third-output",
                    resume_checkpoint=first_latest,
                ),
                device="cpu",
            )
