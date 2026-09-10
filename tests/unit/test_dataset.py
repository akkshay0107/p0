"""Tests for replay dataset loading, splitting, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    load_active_runtime_manifest,
)
from p0.replays.dataset import (
    LazyReplayDataset,
    SeriesSplitManifest,
    assign_series_splits,
    load_split_manifest,
    write_split_manifest,
)
from tests.unit.replay_fixtures import (
    build_dataset,
    build_dataset_from_payloads,
    golden_replay_payload,
    sample_replay_payload,
    torch_summaries,
    write_dataset_replay_dataset,
)


class TestReplayDatasets:
    def test_split_assignment_is_order_independent_and_round_trips(self, tmp_path: Path) -> None:
        """Verify series split hashing produces deterministic train/val/test splits regardless of input series ID ordering."""
        global_hash = load_active_runtime_manifest(DEFAULT_RUNTIME_MANIFEST).global_sha256
        first = assign_series_splits(
            ("series-b", "series-a"),
            seed=17,
            validation_fraction=0.2,
            test_fraction=0.2,
            global_contract_sha256=global_hash,
            dataset_hash="a" * 64,
        )
        second = assign_series_splits(
            ("series-a", "series-b"),
            seed=17,
            validation_fraction=0.2,
            test_fraction=0.2,
            global_contract_sha256=global_hash,
            dataset_hash="a" * 64,
        )
        assert first.to_dict() == second.to_dict()
        path = tmp_path / "splits.json"
        write_split_manifest(first, path)
        assert load_split_manifest(path).to_dict() == first.to_dict()

    def test_split_dataset_keeps_series_together(self, tmp_path: Path) -> None:
        """Verify series split partitioning assigns all games of a series to the same split, avoiding data leakage."""
        built = write_dataset_replay_dataset(
            tmp_path,
            (
                sample_replay_payload("game-1", "series-1"),
                sample_replay_payload("game-2", "series-2"),
            ),
        )
        series_ids = sorted({str(summary["series_id"]) for summary in torch_summaries(built)})
        global_hash = built.manifest.global_contract_sha256
        split = SeriesSplitManifest(
            global_hash,
            0,
            {series_ids[0]: "train", series_ids[1]: "test"},
            dataset_hash=built.manifest.dataset_hash,
        )
        split_path = tmp_path / "splits.json"
        write_split_manifest(split, split_path)
        train = list(
            LazyReplayDataset(built.manifest_path, split="train", split_manifest=split_path)
        )
        test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split_path))
        assert {chunk.series_id for chunk in train} == {series_ids[0]}
        assert {chunk.series_id for chunk in test} == {series_ids[1]}

    def test_lazy_dataset_yields_canonical_bo3_game_perspectives(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify LazyReplayDataset yields 2 perspectives per game with canonical player tracking and terminal series flags."""
        built = write_dataset_replay_dataset(
            tmp_path, (sample_replay_payload("game-1"), sample_replay_payload("game-2"))
        )
        chunks = list(LazyReplayDataset(built.manifest_path))

        # 2 games * 2 perspectives = 4 chunks in sequential order
        assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        assert [chunk.canonical_player for chunk in chunks] == [0, 1, 0, 1]
        assert [chunk.is_series_end for chunk in chunks] == [False, False, True, True]
        assert all(chunk.length == 2 for chunk in chunks)
        assert chunks[2].candidate_offsets.tolist() == [0, 12, 15]

    def test_dataset_rejects_tampered_golden_shard(self, tmp_path: Path) -> None:
        """Verify LazyReplayDataset raises ValueError on tampered golden replay shards when verify_hashes=True."""
        built = build_dataset(tmp_path, 1)
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))

    def test_dataset_rejects_missing_golden_shard(self, tmp_path: Path) -> None:
        """Verify LazyReplayDataset detects missing physical shard files on disk and raises ValueError."""
        built = build_dataset(tmp_path, 1)
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.unlink()
        with pytest.raises(ValueError, match="Shard file is missing"):
            next(iter(LazyReplayDataset(built.manifest_path)))

    def test_dataset_rejects_missing_and_duplicate_source_records(self, tmp_path: Path) -> None:
        """Verify ShardManifest validation checks consistency between raw_replays and source_series records."""
        missing = build_dataset_from_payloads(
            tmp_path / "missing", (golden_replay_payload("only", series_id="series"),)
        )
        missing_manifest = missing.manifest.to_dict()
        missing_manifest["raw_replays"] = {}
        altered = missing.manifest_path.parent / "missing.json"
        altered.write_text(json.dumps(missing_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="raw_replays|source_games|partition"):
            LazyReplayDataset(altered)

        duplicate_payloads = (
            golden_replay_payload("duplicate", series_id="duplicate-series"),
            golden_replay_payload("duplicate", series_id="duplicate-series"),
        )
        with pytest.raises((ValueError, KeyError), match="duplicate|already|unique|invalid"):
            build_dataset_from_payloads(tmp_path / "duplicate", duplicate_payloads)
