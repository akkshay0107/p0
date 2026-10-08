"""Integration tests for behavior-cloning runner and CLI orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import (
    SeriesSplitManifest,
    write_split_manifest,
)
from p0.replays.schema import LabelKind
from p0.training.bc_runner import evaluate_bc, train_bc
from p0.training.checkpoint import CheckpointStore, value_objective_metadata
from p0.training.config import BCConfig
from tests.unit.replay_fixtures import sample_replay_payload


@pytest.mark.heavy
@pytest.mark.integration
class TestBCRunner:
    @pytest.mark.parametrize(
        ("split_assignment", "match_error"),
        [
            ("train", "validation split has no accepted series"),
            ("validation", "train split has no accepted series"),
        ],
    )
    def test_training_rejects_empty_splits(
        self,
        tmp_path: Path,
        split_assignment: str,
        match_error: str,
    ) -> None:
        result = compile_payloads((sample_replay_payload("empty-split", parent="only-series"),))
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = built.manifest_path.parent / "splits.json"
        only_series = next(iter(built.manifest.source_series))
        write_split_manifest(
            SeriesSplitManifest(
                0,
                {only_series: split_assignment},
                dataset_id=built.manifest.dataset_id,
                split_id="split-test",
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
            output_dir=tmp_path / "output",
        )

        with pytest.raises(ValueError, match=match_error):
            train_bc(config, device="cpu")

        assert not list((tmp_path / "output").rglob("bc_best_policy.pt"))

    def test_runner_learns_on_toy_dataset(self, tmp_path: Path) -> None:
        compiled = compile_payloads(
            (
                sample_replay_payload("train-replay", parent="train-series"),
                sample_replay_payload(
                    "validation-replay",
                    parent="validation-series",
                ),
            )
        )
        exact_series = []
        for series in compiled.accepted_series:
            game = series.games[0]
            perspectives = tuple(
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
            )
            exact_series.append(replace(series, games=(replace(game, perspectives=perspectives),)))
        exact_compiled = replace(compiled, accepted_series=tuple(exact_series))
        built = write_tensor_shards(
            exact_compiled,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        split_path = built.manifest_path.parent / "splits.json"
        write_split_manifest(
            SeriesSplitManifest(
                0,
                {
                    series_by_replay["train-replay"]: "train",
                    series_by_replay["validation-replay"]: "validation",
                },
                dataset_id=built.manifest.dataset_id,
                split_id="split-test",
            ),
            split_path,
        )
        config = BCConfig(
            batch_decisions=4,
            max_chunk_size=4,
            learning_rate=1e-3,
            epochs=20,
            num_workers=0,
            enable_optim=False,
            shard_manifest=built.manifest_path,
            output_dir=tmp_path / "output",
        )

        result = train_bc(config, device="cpu")

        assert result["completed_epoch"] == config.epochs
        assert result["latest_training_checkpoint"] is not None
        assert result["best_policy_checkpoint"] is not None
        assert result["final_validation"] is not None
        assert result["final_training"]["overall_nll"] <= (
            result["initial_training"]["overall_nll"] * 0.5
        )

    def test_resume_restores_selected_policy_and_metrics_without_external_files(
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
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = built.manifest_path.parent / "splits.json"
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        write_split_manifest(
            SeriesSplitManifest(
                0,
                {
                    series_by_replay["resume-train"]: "train",
                    series_by_replay["resume-validation"]: "validation",
                },
                dataset_id=built.manifest.dataset_id,
                split_id="split-test",
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
            output_dir=tmp_path / "first-output",
        )
        first = train_bc(first_config, device="cpu")
        first_latest = Path(first["latest_training_checkpoint"])
        first_best = Path(first["best_policy_checkpoint"])
        with pytest.raises(ValueError, match="already contains an experiment"):
            train_bc(first_config, device="cpu")
        occupied_output = tmp_path / "occupied-output"
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
        assert Path(resumed["best_policy_checkpoint"]).is_file()
        assert json.loads(Path(resumed["metrics_path"]).read_text())["completed_step"] == 1
        assert first_latest.is_file()
        assert first_best.is_file()

        overfit_checkpoint_path = tmp_path / "legacy_overfit.pt"
        overfit_artifact = dict(torch.load(first_latest, weights_only=True))
        overfit_artifact["provenance"] = dict(overfit_artifact["provenance"])
        overfit_artifact["provenance"]["trainer_config"] = {
            **overfit_artifact["provenance"]["trainer_config"],
            "overfit": True,
        }
        torch.save(overfit_artifact, overfit_checkpoint_path)
        with pytest.raises(ValueError, match="Cannot resume an overfit training run"):
            train_bc(
                replace(
                    first_config,
                    output_dir=tmp_path / "overfit-output",
                    resume_checkpoint=overfit_checkpoint_path,
                ),
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

        # The inference file is replaceable; recovery comes from the training checkpoint.
        first_best.write_bytes(b"interrupted replacement")
        recovered = train_bc(
            replace(first_config, epochs=2, resume_checkpoint=first_latest),
            device="cpu",
        )
        assert recovered["completed_epoch"] == 2
        best = torch.load(first_best, weights_only=True)
        torch.testing.assert_close(
            best["model_state_dict"],
            continued_artifact["training_state"]["run"]["best_policy"]["model_state_dict"],
        )
        first_best.unlink()
        portable = tmp_path / "moved.pt"
        portable.write_bytes(first_latest.read_bytes())
        restored = train_bc(
            replace(
                first_config,
                epochs=2,
                output_dir=tmp_path / "third-output",
                resume_checkpoint=portable,
            ),
            device="cpu",
        )
        assert Path(restored["best_policy_checkpoint"]).is_file()
        history = json.loads(Path(restored["metrics_path"]).read_text())["metrics"]
        assert [record["step"] for record in history] == [1, 2]

    def test_bc_evaluation_and_cli_parity(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (
                sample_replay_payload("eval-train", parent="train-series"),
                sample_replay_payload("eval-validation", parent="validation-series"),
            )
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = built.manifest_path.parent / "splits.json"
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        write_split_manifest(
            SeriesSplitManifest(
                0,
                {
                    series_by_replay["eval-train"]: "train",
                    series_by_replay["eval-validation"]: "validation",
                },
                dataset_id=built.manifest.dataset_id,
                split_id="split-test",
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
            output_dir=tmp_path / "output",
        )
        policy_checkpoint = tmp_path / "best_policy.pt"
        store = CheckpointStore()
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        store.save_policy(
            policy_checkpoint,
            policy,
            metadata={
                **value_objective_metadata(config.gamma),
                "trainer_kind": "bc",
                "selected_epoch": 1,
            },
        )

        evaluation = evaluate_bc(config, policy_checkpoint, split="validation", device="cpu")
        assert evaluation["objective"] == {
            "gamma": config.gamma,
            "value_target_semantics": "discounted_terminal_outcome.v1",
        }
        cli_config = tmp_path / "bc-config.yaml"
        cli_config.write_text(
            json.dumps(
                {
                    "bc": {
                        "shard_manifest": str(built.manifest_path),
                    }
                }
            ),
            encoding="utf-8",
        )
        cli = subprocess.run(
            (
                sys.executable,
                "-m",
                "p0.cli.bc",
                "evaluate",
                "--config",
                str(cli_config),
                "--checkpoint",
                str(policy_checkpoint),
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
        assert cli_evaluation["checkpoint"] == str(policy_checkpoint)
        assert cli_evaluation["metrics"] == evaluation["metrics"]
        with pytest.raises(ValueError, match="provenance field 'gamma' is incompatible"):
            evaluate_bc(
                replace(config, gamma=0.5),
                policy_checkpoint,
                split="validation",
                device="cpu",
            )
        with pytest.raises(ValueError, match="test split has no accepted series"):
            evaluate_bc(config, policy_checkpoint, split="test", device="cpu")

    def test_training_cancellation_halts_cleanly(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (
                sample_replay_payload("cancel-train", parent="train-series"),
                sample_replay_payload("cancel-validation", parent="validation-series"),
            )
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            created_at="2026-01-01T00:00:00Z",
        )
        split_path = built.manifest_path.parent / "splits.json"
        series_by_replay = {
            replay_ids[0]: series_id
            for series_id, replay_ids in built.manifest.source_series.items()
        }
        write_split_manifest(
            SeriesSplitManifest(
                0,
                {
                    series_by_replay["cancel-train"]: "train",
                    series_by_replay["cancel-validation"]: "validation",
                },
                dataset_id=built.manifest.dataset_id,
                split_id="split-test",
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
            output_dir=tmp_path / "cancelled-output",
        )

        cancelled = train_bc(config, device="cpu", cancel_requested=lambda: True)
        assert cancelled["completed_epoch"] == 0
        assert cancelled["cancelled"] is True
        assert cancelled["latest_training_checkpoint"] is None
        assert cancelled["best_policy_checkpoint"] is None
        assert cancelled["metrics_path"] is None
        assert not list((tmp_path / "cancelled-output").rglob("*.pt"))
