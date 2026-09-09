"""Tests for replay compilation and shard production."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    load_active_runtime_manifest,
    load_runtime_manifest,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.replays.compile import (
    CompilationResult,
    ShardBuildResult,
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.dataset import (
    LazyReplayDataset,
    SeriesSplitManifest,
    assign_series_splits,
    load_split_manifest,
    write_split_manifest,
)
from p0.replays.schema import (
    ActionEvidence,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    SeriesRecord,
)
from p0.replays.shards import (
    ShardIndexEntry,
    ShardManifest,
)
from tests.unit.replay_fixtures import golden_replay_payload, sample_replay_payload


def _write_dataset_replay_dataset(
    tmp_path: Path, payloads: tuple[dict[str, object], ...]
) -> ShardBuildResult:
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def _build_dataset_from_payloads(tmp_path, payloads):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path, count: int):
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    return _build_dataset_from_payloads(tmp_path, payloads)


def _payload_with_ots_natures(replay_id: str) -> dict[str, object]:
    """A pipeline payload whose open team sheets declare natures, as real replays do."""
    natures = {"Pikachu": "Jolly", "Eevee": "Adamant", "Bulbasaur": "Bold", "Charmander": "Timid"}
    payload = sample_replay_payload(replay_id)
    lines = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|"):
            head, _, body = line.rpartition("|")
            roster = json.loads(body)
            for mon in roster:
                mon["nature"] = natures.get(mon["species"], "Serious")
            line = f"{head}|{json.dumps(roster, separators=(',', ':'))}"
        lines.append(line)
    payload["log"] = "\n".join(lines)
    return payload


def _numerical_rows(result: CompilationResult) -> list[tuple[float, ...]]:
    builder = ObservationBuilder(default_runtime_resources())
    rows: list[tuple[float, ...]] = []
    for game in result.games:
        for perspective in game.perspectives:
            for snapshot in perspective.snapshots:
                observation = builder.build(snapshot.view)
                rows.extend(
                    tuple(float(value) for value in token) for token in observation.numerical
                )
    return rows


def _evidence(kind: LabelKind) -> ActionEvidence:
    candidates = {
        LabelKind.EXACT: ((7, 1),),
        LabelKind.PARTIAL: ((7, 1), (8, 1)),
        LabelKind.UNKNOWN: (),
    }[kind]
    return ActionEvidence(
        label_kind=kind,
        candidates=candidates,
        confidence=0.5 if kind is not LabelKind.UNKNOWN else 0.0,
        mask_provenance=MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tags=("fixture",),
    )


def _series_record() -> SeriesRecord:
    return SeriesRecord(
        series_id="s1",
        format_id="gen9championsvgc2026regmbbo3",
        players=("alice", "bob"),
        game_replay_ids=("r1", "r2"),
        game_player_roles=((0, 1), (1, 0)),
        team_hashes=("a" * 64, "b" * 64),
        is_complete=True,
        score=(2, 0),
        grouping_method=GroupingMethod.PARENT_ROOM,
        grouping_confidence=1.0,
    )


def _shard_manifest_fixture_unit() -> ShardManifest:
    active_contract = load_runtime_manifest().global_sha256
    entry = ShardIndexEntry(
        filename="shard-000.pt", sha256="c" * 64, decisions=10, games=2, series=1, byte_size=1024
    )
    return ShardManifest(
        global_contract_sha256=active_contract,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"max_candidates": 256},
        raw_replays={"game-1": "f" * 64},
        source_series={"series-1": ("game-1",)},
        source_games=1,
        accepted_games=1,
        rejected_games=0,
        artifact_hashes={
            "shard-000.pt": "c" * 64,
        },
    )


class TestReplayCompiler:
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

    def test_lazy_dataset_yields_canonical_bo3_game_perspectives(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify LazyReplayDataset yields 2 perspectives per game with canonical player tracking and terminal series flags."""
        built = _write_dataset_replay_dataset(
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

    def test_canonical_player_identity_survives_replay_side_swap(self, tmp_path: Path) -> None:
        """Verify canonical player IDs track original human players even when Showdown swaps p1/p2 sides between games."""
        first = sample_replay_payload("game-1")
        first["game_number"] = 1
        second = sample_replay_payload("game-2")
        second["game_number"] = 2
        second["p1"] = "Bob"
        second["p2"] = "Alice"

        built = _write_dataset_replay_dataset(tmp_path, (first, second))
        chunks = list(LazyReplayDataset(built.manifest_path))

        # Game 2 swaps sides (p1=Bob, p2=Alice), so canonical_player maps (p1->1, p2->0)
        assert [(chunk.game_number, chunk.player, chunk.canonical_player) for chunk in chunks] == [
            (1, 0, 0),
            (1, 1, 1),
            (2, 0, 1),
            (2, 1, 0),
        ]

    def test_downstream_shards_reject_noncontiguous_source_game_numbers(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject shard publication when a series has a missing first game."""
        second = sample_replay_payload("game-2")
        second["game_number"] = 2
        third = sample_replay_payload("game-3")
        third["game_number"] = 3

        with pytest.raises(ValueError, match="incomplete chronological games"):
            _write_dataset_replay_dataset(tmp_path, (third, second))

    def test_split_dataset_keeps_series_together(self, tmp_path: Path) -> None:
        """Verify series split partitioning assigns all games of a series to the same split, avoiding data leakage."""
        built = _write_dataset_replay_dataset(
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

    def test_dataset_rejects_tampered_shard(self, tmp_path: Path) -> None:
        """Verify LazyReplayDataset raises ValueError when shard bytes do not match the manifest SHA-256 hash."""
        built = _write_dataset_replay_dataset(tmp_path, (sample_replay_payload("game-1"),))
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))
