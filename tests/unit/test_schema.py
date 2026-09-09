"""Tests for replay schema and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p0.format_config import (
    current_manifest,
    load_runtime_manifest,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    StructuredObservation,
)
from p0.replays.compile import (
    CompilationResult,
    ShardBuildResult,
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.schema import (
    ActionEvidence,
    FetchIndexEntry,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    SeriesRecord,
)
from p0.replays.shards import (
    SHARD_TENSOR_SPECS,
    ShardIndexEntry,
    ShardManifest,
    load_shard_manifest,
    observation_field_specs,
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


class TestReplaySchemas:
    def test_evidence_shapes(self) -> None:
        """Verify ActionEvidence enforces candidate counts and ranges according to EXACT/PARTIAL/UNKNOWN taxonomy."""
        assert _evidence(LabelKind.EXACT).exact_action == (7, 1)
        with pytest.raises(ValueError, match="only defined for EXACT"):
            _evidence(LabelKind.PARTIAL).exact_action
        with pytest.raises(ValueError, match="exactly one candidate"):
            ActionEvidence(LabelKind.EXACT, (), 1.0, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
        with pytest.raises(ValueError, match="two or more"):
            ActionEvidence(
                LabelKind.PARTIAL,
                ((7, 1),),
                0.5,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="no candidates"):
            ActionEvidence(
                LabelKind.UNKNOWN,
                ((7, 1),),
                0.0,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="outside"):
            ActionEvidence(
                LabelKind.EXACT,
                ((49, 0),),
                1.0,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="Duplicate"):
            ActionEvidence(
                LabelKind.PARTIAL,
                ((7, 1), (7, 1)),
                0.5,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )

    def test_ir_round_trips(self) -> None:
        """Verify SeriesRecord and FetchIndexEntry serialize and deserialize cleanly."""
        series = _series_record()
        assert SeriesRecord.from_dict(series.to_dict()) == series
        fetch = FetchIndexEntry(
            replay_id="r1",
            format_id="gen9championsvgc2026regmbbo3",
            source_url="https://replay.pokemonshowdown.com/r1",
            fetched_at="2026-07-17T00:00:00Z",
            http_status=200,
            content_sha256="d" * 64,
            byte_size=100,
        )
        assert FetchIndexEntry.from_dict(fetch.to_dict()) == fetch

    def test_ir_rejects_bad_serializations(self) -> None:
        """Verify IR deserialization raises ValueError on schema version mismatch or missing/unknown fields."""
        payload = _series_record().to_dict()
        del payload["score"]
        payload["bogus"] = 1
        with pytest.raises(ValueError, match=r"missing=\['score'\], unknown=\['bogus'\]"):
            SeriesRecord.from_dict(payload)

    def test_ir_validates_construction(self) -> None:
        """Verify SeriesRecord validates structural invariants on deserialization."""
        with pytest.raises(ValueError, match="two wins"):
            SeriesRecord.from_dict({**_series_record().to_dict(), "score": [1, 0]})

    def test_observation_specs_are_derived(self) -> None:
        """Verify public observation and shard specs expose the runtime tensor contract."""
        specs = observation_field_specs()
        assert [spec[0] for spec in specs] == [
            "token_type_ids",
            "side_ids",
            "slot_ids",
            "categorical",
            "numerical",
            "spatial_cat",
            "spatial_num",
        ]
        observation = StructuredObservation.empty_batch(1)
        assert all(shape[0] == -1 for _, shape, _ in specs)
        assert all(
            dtype is tensor.dtype
            for (_, _, dtype), tensor in zip(specs, observation.tensors(), strict=True)
        )
        assert [spec[0] for spec in SHARD_TENSOR_SPECS] == [
            "action_mask",
            "mask_provenance",
            "label_kind",
            "label_confidence",
            "loss_mask",
            "decision_type",
            "exact_action",
            "candidate_values",
            "candidate_offsets",
            "game_offsets",
            "series_offsets",
            "outcome",
        ]

    def test_shard_manifest_contract(self) -> None:
        """Verify ShardManifest contract checks global SHA-256 validity and roundtrips through dictionary representation."""
        manifest = _shard_manifest_fixture_unit()
        assert ShardManifest.from_dict(manifest.to_dict()) == manifest
        assert manifest.decisions == 10 and manifest.games == 2 and manifest.series == 1
        assert load_shard_manifest(manifest.to_dict()) == manifest
        with pytest.raises(ValueError, match="incompatible"):
            load_shard_manifest({**manifest.to_dict(), "global_contract_sha256": "0" * 64})
        with pytest.raises(ValueError, match="unknown"):
            load_shard_manifest({**manifest.to_dict(), "runtime_manifest_sha256": "0" * 64})

    def test_shard_manifest_round_trip_and_tamper_detection(self, tmp_path: Path) -> None:
        """Verify ShardManifest verifies global runtime contract and detects tampered or mismatched source series."""
        vocab = tmp_path / "vocab.json"
        dex = tmp_path / "champions_dex.json"
        vocab.write_text(
            json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
        )
        dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
        runtime = current_manifest(vocab_path=vocab, dex_path=dex)

        entry = ShardIndexEntry("shard-000.pt", "c" * 64, 10, 2, 1, 100)
        manifest = ShardManifest(
            global_contract_sha256=runtime.global_sha256,
            shards=(entry,),
            diagnostics={"oov_ids": 0},
            created_at="2026-07-17T00:00:00Z",
            dataset_hash="d" * 64,
            source_format_id="gen9championsvgc2026regmbbo3",
            build_config={"seed": 3},
            raw_replays={"game-1": "f" * 64, "game-2": "e" * 64},
            source_series={"series-1": ("game-1", "game-2")},
            source_games=2,
            accepted_games=2,
            rejected_games=0,
            artifact_hashes={"shard-000.pt": "c" * 64},
        )
        assert ShardManifest.from_dict(manifest.to_dict()) == manifest
        manifest_path = tmp_path / "runtime_manifest.json"
        manifest_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")
        with pytest.raises(ValueError, match="default global manifest"):
            load_shard_manifest(
                {**manifest.to_dict(), "global_contract_sha256": "b" * 64}, manifest_path
            )
        with pytest.raises(ValueError, match="source_series"):
            ShardManifest.from_dict(
                {**manifest.to_dict(), "source_series": {"series-1": ("game-1",)}}
            )

    def test_schema_modules_stay_pure(self) -> None:
        """Verify intermediate representation modules stay pure without importing torch or runtime."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import p0.replays.schema, p0.battle.series\n"
            "assert 'torch' not in sys.modules, 'IR layer must stay torch-free'\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
            "import p0.replays.shards\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
