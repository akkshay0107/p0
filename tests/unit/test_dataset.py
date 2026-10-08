"""Tests for replay dataset loading, splitting, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p0.model.structured_observation import (
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    TOKEN_IDX_ALLY_SIDE,
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
    sample_replay_payload,
    torch_summaries,
    write_dataset_replay_dataset,
)


class TestReplayDatasets:
    def test_split_assignment_is_order_independent_and_round_trips(self, tmp_path: Path) -> None:
        """Verify seeded series assignment produces deterministic train/val/test splits regardless of input series ID ordering."""

        first = assign_series_splits(
            ("series-b", "series-a"),
            seed=17,
            validation_fraction=0.2,
            test_fraction=0.2,
            dataset_id="a" * 64,
        )
        second = assign_series_splits(
            ("series-a", "series-b"),
            seed=17,
            validation_fraction=0.2,
            test_fraction=0.2,
            dataset_id="a" * 64,
        )
        assert first.assignments == second.assignments
        path = tmp_path / "splits.json"
        write_split_manifest(first, path)
        assert load_split_manifest(path).to_dict() == first.to_dict()

    @pytest.mark.parametrize(
        ("series_ids", "val_fraction", "test_fraction", "expected_splits"),
        [
            (("a", "b", "c", "d", "e"), 0.1, 0.1, {"train", "validation", "test"}),
            (("a", "b", "c"), 0.1, 0.1, {"train", "validation", "test"}),
            (("a", "b"), 0.1, 0.1, {"train"}),
            (("a",), 0.1, 0.1, {"train"}),
            (("a", "b", "c", "d", "e"), 0.0, 0.0, {"train"}),
        ],
    )
    def test_split_assignment_populates_all_requested_splits_when_possible(
        self,
        series_ids: tuple[str, ...],
        val_fraction: float,
        test_fraction: float,
        expected_splits: set[str],
    ) -> None:
        """Verify assign_series_splits allocates items to requested partitions under boundary conditions."""
        manifest = assign_series_splits(
            series_ids,
            validation_fraction=val_fraction,
            test_fraction=test_fraction,
            dataset_id="b" * 64,
        )
        assert set(manifest.assignments.values()) == expected_splits

    def test_split_dataset_keeps_series_together(self, tmp_path: Path) -> None:
        """Verify series split partitioning assigns all games of a multi-game series to the same split."""
        built = write_dataset_replay_dataset(
            tmp_path,
            (
                sample_replay_payload("game-1", "series-1", game_number=1),
                sample_replay_payload("game-2", "series-1", game_number=2),
                sample_replay_payload("game-3", "series-2", game_number=1),
                sample_replay_payload("game-4", "series-2", game_number=2),
            ),
        )
        series_ids = sorted({str(summary["series_id"]) for summary in torch_summaries(built)})

        split = SeriesSplitManifest(
            0,
            {series_ids[0]: "train", series_ids[1]: "test"},
            dataset_id=built.manifest.dataset_id,
            split_id="split-test",
        )
        split_path = tmp_path / "splits.json"
        write_split_manifest(split, split_path)
        train = list(
            LazyReplayDataset(built.manifest_path, split="train", split_manifest=split_path)
        )
        test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split_path))
        assert [(chunk.series_id, chunk.game_number, chunk.player) for chunk in train] == [
            (series_ids[0], 1, 0),
            (series_ids[0], 1, 1),
            (series_ids[0], 2, 0),
            (series_ids[0], 2, 1),
        ]
        assert [(chunk.series_id, chunk.game_number, chunk.player) for chunk in test] == [
            (series_ids[1], 1, 0),
            (series_ids[1], 1, 1),
            (series_ids[1], 2, 0),
            (series_ids[1], 2, 1),
        ]

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

    def test_each_game_perspective_carries_one_final_board_with_unknown_legality(
        self,
        tmp_path: Path,
    ) -> None:
        built = write_dataset_replay_dataset(
            tmp_path, (sample_replay_payload("game-1"), sample_replay_payload("game-2"))
        )
        chunks = list(LazyReplayDataset(built.manifest_path))

        for chunk in chunks:
            final = chunk.final_observation
            assert final.categorical.shape[0] == 1
            slot_unknown = final.numerical[
                0,
                TOKEN_IDX_ALLY_SIDE,
                NUM_IDX_SLOT_LEGALITY_UNKNOWN : NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2,
            ]
            assert slot_unknown.tolist() == [1.0, 1.0]
            assert not torch.equal(final.numerical[0], chunk.observations.numerical[-1])

    def test_dataset_rejects_missing_golden_shard(self, tmp_path: Path) -> None:
        """Verify LazyReplayDataset detects missing physical shard files on disk and raises ValueError."""
        built = build_dataset(tmp_path, 1)
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.unlink()
        with pytest.raises(ValueError, match="Shard file is missing"):
            next(iter(LazyReplayDataset(built.manifest_path)))

    def test_dataset_rejects_duplicate_persisted_summaries(self, tmp_path: Path) -> None:
        """Verify LazyReplayDataset detects and rejects duplicate persisted summaries at the reader boundary."""
        built = write_dataset_replay_dataset(
            tmp_path,
            (sample_replay_payload("game-1"), sample_replay_payload("game-2")),
        )
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        payload = torch.load(shard_path, weights_only=True, map_location="cpu")
        summaries = payload["series_summaries"]
        payload["series_summaries"] = [*summaries, summaries[0]]
        torch.save(payload, shard_path)

        manifest = built.manifest.to_dict()

        altered = built.manifest_path.parent / "altered.json"
        altered.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="Shard summary count does not match game offsets|Shard summaries do not cover",
        ):
            list(LazyReplayDataset(altered))

    def test_dataset_rejects_nonchronological_game_summaries(self, tmp_path: Path) -> None:
        built = write_dataset_replay_dataset(
            tmp_path,
            (sample_replay_payload("game-1"), sample_replay_payload("game-2")),
        )
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        payload = torch.load(shard_path, weights_only=True, map_location="cpu")
        summaries = payload["series_summaries"]
        payload["series_summaries"] = [*summaries[2:4], *summaries[0:2]]
        torch.save(payload, shard_path)

        manifest = built.manifest.to_dict()

        altered = built.manifest_path.parent / "altered.json"
        altered.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="not chronological"):
            list(LazyReplayDataset(altered))
