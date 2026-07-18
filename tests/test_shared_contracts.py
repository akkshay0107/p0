"""Pins for the shared cross-workstream schema contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest

from p0.battle.series import (
    MAX_PRIOR_GAMES,
    GameSummary,
    SideGameSummary,
)
from p0.format_config import load_runtime_manifest
from p0.model.config import ModelConfig
from p0.replays.schema import (
    ActionEvidence,
    DecisionRecord,
    DecisionType,
    FetchIndexEntry,
    GameEndReason,
    GameRecord,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    ReplayDiagnostics,
    SeriesRecord,
)
from p0.replays.shards import (
    SHARD_TENSOR_SPECS,
    ShardIndexEntry,
    ShardManifest,
    load_shard_manifest,
    observation_field_specs,
)
from p0.teams.corpus import (
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    SamplingPolicy,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)
from p0.training.config import load_config

ACTIVE_CONTRACT = load_runtime_manifest().runtime_contract_sha256


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


def _game_record() -> GameRecord:
    decision = DecisionRecord(
        decision_index=0,
        player=0,
        decision_type=DecisionType.TURN,
        pre_line_index=1,
        post_line_index=3,
        evidence=_evidence(LabelKind.PARTIAL),
    )
    return GameRecord(
        game_id="g1",
        series_id="s1",
        game_number=1,
        protocol_lines=("|start", "|turn|1", "|move|p1a: A|Protect|p1a: A", "|win|alice"),
        ots_payloads=("p1 sheet", "p2 sheet"),
        winner=0,
        end_reason=GameEndReason.NORMAL,
        turns=1,
        decisions=(decision,),
        diagnostics=ReplayDiagnostics(counters={"oov_ids": 0}, parse_errors=()),
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


def _side_summary() -> SideGameSummary:
    return SideGameSummary(
        leads=("koraidon", "fluttermane"),
        brought=("koraidon", "fluttermane", "amoonguss"),
        mega_species="",
        moves_used={"koraidon": ("collisioncourse",)},
        revealed_items={"koraidon": "clearamulet"},
        revealed_abilities={"koraidon": "orichalcumpulse"},
        revealed_formes=(),
        switch_count=2,
        pivot_count=0,
        plan_tags=("protect",),
    )


def _game_summary() -> GameSummary:
    return GameSummary(
        game_number=1,
        winner=0,
        series_score=(1, 0),
        turns=9,
        sides=(_side_summary(), _side_summary()),
        speed_observations=("koraidon>fluttermane",),
    )


def _shard_manifest() -> ShardManifest:
    entry = ShardIndexEntry(
        filename="shard-000.pt", sha256="c" * 64, decisions=10, games=2, series=1, byte_size=1024
    )
    return ShardManifest(
        runtime_contract_sha256=ACTIVE_CONTRACT,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
    )


def _corpus_entry(packed: str = "packed-team") -> CorpusEntry:
    import hashlib

    return CorpusEntry(
        canonical_hash=hashlib.sha256(packed.encode()).hexdigest(),
        packed=packed,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        split=CorpusSplit.TRAIN,
        usage_count=3,
        archetype_tags=("rain",),
    )


def _corpus_manifest(entries: tuple[CorpusEntry, ...]) -> TeamCorpusManifest:
    return TeamCorpusManifest(
        runtime_contract_sha256=ACTIVE_CONTRACT,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"policy": "usage_weighted"},
    )


def test_evidence_shapes() -> None:
    assert _evidence(LabelKind.EXACT).exact_action == (7, 1)
    with pytest.raises(ValueError, match="only defined for EXACT"):
        _evidence(LabelKind.PARTIAL).exact_action
    with pytest.raises(ValueError, match="exactly one candidate"):
        ActionEvidence(LabelKind.EXACT, (), 1.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="two or more"):
        ActionEvidence(LabelKind.PARTIAL, ((7, 1),), 0.5, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="no candidates"):
        ActionEvidence(LabelKind.UNKNOWN, ((7, 1),), 0.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="outside"):
        ActionEvidence(LabelKind.EXACT, ((49, 0),), 1.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="Duplicate"):
        ActionEvidence(LabelKind.PARTIAL, ((7, 1), (7, 1)), 0.5, MaskProvenance.ORACLE_REQUEST)


def test_ir_round_trips() -> None:
    game = _game_record()
    assert GameRecord.from_dict(game.to_dict()) == game
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


def test_ir_rejects_bad_serializations() -> None:
    payload = _game_record().to_dict()
    payload["ir_schema"] = 2
    with pytest.raises(ValueError, match="ir_schema"):
        GameRecord.from_dict(payload)
    payload = _series_record().to_dict()
    del payload["score"]
    payload["bogus"] = 1
    with pytest.raises(ValueError, match=r"missing=\['score'\], unknown=\['bogus'\]"):
        SeriesRecord.from_dict(payload)


def test_ir_validates_construction() -> None:
    with pytest.raises(ValueError, match="ascending"):
        game = _game_record()
        GameRecord.from_dict({**game.to_dict(), "decisions": [game.decisions[0].to_dict()] * 2})
    with pytest.raises(ValueError, match="two wins"):
        SeriesRecord.from_dict({**_series_record().to_dict(), "score": [1, 0]})


def test_game_summary_round_trip() -> None:
    summary = _game_summary()
    assert GameSummary.from_dict(summary.to_dict()) == summary
    assert MAX_PRIOR_GAMES == 2
    with pytest.raises(ValueError, match="schema"):
        GameSummary.from_dict({**summary.to_dict(), "summary_schema": 2})
    with pytest.raises(ValueError, match="normalized"):
        SideGameSummary.from_dict({**_side_summary().to_dict(), "leads": ["Koraidon", "x"]})
    with pytest.raises(ValueError, match="credit the winner"):
        GameSummary.from_dict({**summary.to_dict(), "series_score": [0, 1]})


def test_observation_specs_are_derived() -> None:
    from p0.model.structured_observation import StructuredObservation

    specs = observation_field_specs()
    assert [spec[0] for spec in specs] == [spec[0] for spec in StructuredObservation._FIELD_SPECS]
    for (name, shape, dtype), (_, base_shape, base_dtype) in zip(
        specs, StructuredObservation._FIELD_SPECS, strict=True
    ):
        assert shape == (-1, *base_shape) and dtype is base_dtype, name
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


def test_shard_manifest_contract() -> None:
    manifest = _shard_manifest()
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.decisions == 10 and manifest.games == 2 and manifest.series == 1
    assert load_shard_manifest(manifest.to_dict()) == manifest
    with pytest.raises(ValueError, match="incompatible"):
        load_shard_manifest({**manifest.to_dict(), "runtime_contract_sha256": "0" * 64})
    with pytest.raises(ValueError, match="legacy"):
        load_shard_manifest({**manifest.to_dict(), "runtime_manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact schema"):
        ShardManifest.from_dict({**manifest.to_dict(), "artifact_schema": "p0.replay_shard.v0"})
    with pytest.raises(ValueError, match="observation_schema_version"):
        ShardManifest.from_dict({**manifest.to_dict(), "observation_schema_version": 2})


def test_corpus_manifest_contract() -> None:
    entries = (_corpus_entry("team-a"), _corpus_entry("team-b"))
    manifest = _corpus_manifest(entries)
    assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
    assert load_corpus_manifest(manifest.to_dict()) == manifest
    assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
    with pytest.raises(ValueError, match="does not match the packed team"):
        CorpusEntry(
            canonical_hash="a" * 64,
            packed="team",
            packed_sha256="b" * 64,
            split=CorpusSplit.TRAIN,
            usage_count=1,
        )
    with pytest.raises(ValueError, match="does not match the entries"):
        TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "0" * 64})
    with pytest.raises(ValueError, match="Duplicate corpus entry"):
        _corpus_manifest((entries[0], entries[0]))


def test_corpus_source_spec_validates() -> None:
    spec = CorpusSourceSpec(
        corpus_path="teams/corpus_manifest.json",
        corpus_hash="a" * 64,
        format_id="gen9championsvgc2026regmb",
        split=CorpusSplit.TRAIN,
        seed=0,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    assert spec.allow_mirror and spec.curriculum_stage == ""
    with pytest.raises(ValueError, match="split"):
        CorpusSourceSpec(
            corpus_path="x",
            corpus_hash="a" * 64,
            format_id="f",
            split=CorpusSplit.UNSPECIFIED,
            seed=0,
            sampling_policy=SamplingPolicy.UNIFORM_CANONICAL,
        )


def test_model_config_series_fields() -> None:
    config = ModelConfig.baseline()
    assert config.series_tokens == 4 and not config.series_context_enabled
    assert ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="disabled during the refactor baseline"):
        ModelConfig(
            d_model=64,
            nhead=4,
            reducer_layers=1,
            history_tokens=2,
            dim_feedforward=128,
            series_context_enabled=True,
        )
    stale = config.to_dict()
    del stale["series_tokens"]
    with pytest.raises(ValueError, match=r"missing=\['series_tokens'\]"):
        ModelConfig.from_dict(stale)


def test_reserved_config_sections(tmp_path) -> None:
    config = load_config("config.yaml.example")
    assert config.bc.chunk_length == 16
    assert config.corpus.agent_split == "train"
    assert config.evaluation.episodes_per_matchup == 20
    bad = tmp_path / "config.yaml"
    bad.write_text("bc:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown BCConfig field"):
        load_config(bad)


def test_schema_modules_stay_pure() -> None:
    code = (
        "import sys\n"
        "import p0.replays.schema, p0.battle.series\n"
        "assert 'torch' not in sys.modules, 'IR layer must stay torch-free'\n"
        "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
        "import p0.replays.shards\n"
        "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
