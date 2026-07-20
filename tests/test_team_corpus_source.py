"""Tests for the corpus-backed TeamSource implementation and sampling policies."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from p0.format_config import FORMAT, current_manifest
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    SamplingPolicy,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import ValidatedTeam


def _make_entry(
    index: int,
    canonical_index: int | None = None,
    split: CorpusSplit = CorpusSplit.TRAIN,
    usage_count: int = 10,
    tags: tuple[str, ...] = ("balance",),
) -> CorpusEntry:
    if canonical_index is None:
        canonical_index = index
    canonical = f"canonical_{canonical_index:04d}"
    canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    packed = f"Nickname|Species{index}|item|ability|move1,move2|nature"
    packed_sha256 = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    return CorpusEntry(
        canonical_hash=canonical_hash,
        packed=packed,
        packed_sha256=packed_sha256,
        split=split,
        usage_count=usage_count,
        archetype_tags=tags,
        spread_provenance="imputed",
    )


def _write_manifest(
    tmp_path: Path, entries: tuple[CorpusEntry, ...]
) -> tuple[Path, TeamCorpusManifest]:
    manifest = TeamCorpusManifest(
        artifact_schema=CORPUS_MANIFEST_SCHEMA,
        runtime_contract_sha256=current_manifest().runtime_contract_sha256,
        format_id=FORMAT.battle_format,
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-19T12:00:00Z",
        sampling_metadata={"pool_size": len(entries)},
    )
    path = tmp_path / "corpus_manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path, manifest


def test_corpus_source_implements_protocol_and_describes(tmp_path: Path) -> None:
    entries = tuple(_make_entry(i) for i in range(5))
    path, manifest = _write_manifest(tmp_path, entries)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=42,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    source = CorpusTeamSource(spec)
    assert hasattr(source, "sample") and callable(source.sample)
    assert hasattr(source, "describe") and callable(source.describe)
    rng = random.Random(spec.seed)
    sampled = source.sample(rng)
    assert isinstance(sampled, ValidatedTeam)
    assert sampled.packed in [e.packed for e in entries]

    desc = source.describe()
    assert desc["kind"] == "corpus"
    assert desc["corpus_hash"] == manifest.corpus_hash
    assert desc["pool_size"] == 5
    hashes = desc["team_hashes"]
    assert isinstance(hashes, tuple)
    assert len(hashes) == 5


def test_corpus_source_validates_spec(tmp_path: Path) -> None:
    entries = tuple(_make_entry(i) for i in range(3))
    path, manifest = _write_manifest(tmp_path, entries)

    # Wrong corpus_hash raises ValueError
    bad_spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash="0" * 64,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=1,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    with pytest.raises(ValueError, match="does not match"):
        CorpusTeamSource(bad_spec)

    # Wrong format_id raises ValueError
    bad_format = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id="wrong-format",
        split=CorpusSplit.TRAIN,
        seed=1,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    with pytest.raises(ValueError, match="format"):
        CorpusTeamSource(bad_format)


def test_corpus_source_rejects_empty_filtered_pool(tmp_path: Path) -> None:
    entries = tuple(_make_entry(i, split=CorpusSplit.TRAIN) for i in range(3))
    path, manifest = _write_manifest(tmp_path, entries)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TEST,
        seed=1,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    with pytest.raises(ValueError, match="No corpus entries match"):
        CorpusTeamSource(spec)


def test_sampling_policy_usage_weighted(tmp_path: Path) -> None:
    e_common = _make_entry(1, usage_count=10000)
    e_rare = _make_entry(2, usage_count=1)
    path, manifest = _write_manifest(tmp_path, (e_common, e_rare))
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=100,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(100)
    counts = {e_common.packed_sha256: 0, e_rare.packed_sha256: 0}
    for _ in range(500):
        t = source.sample(rng)
        counts[t.team_hash] += 1
    assert counts[e_common.packed_sha256] > 490


def test_sampling_policy_uniform_canonical(tmp_path: Path) -> None:
    # 90 entries for canonical 1, 10 entries for canonical 2
    entries_1 = tuple(_make_entry(i, canonical_index=1, usage_count=100) for i in range(1, 91))
    entries_2 = tuple(_make_entry(i, canonical_index=2, usage_count=100) for i in range(91, 101))
    path, manifest = _write_manifest(tmp_path, entries_1 + entries_2)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=200,
        sampling_policy=SamplingPolicy.UNIFORM_CANONICAL,
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(200)
    canonical_counts: dict[str, int] = {}
    for _ in range(600):
        t = source.sample(rng)
        # Find which canonical_index t belongs to
        e = next(entry for entry in entries_1 + entries_2 if entry.packed_sha256 == t.team_hash)
        canonical_counts[e.canonical_hash] = canonical_counts.get(e.canonical_hash, 0) + 1
    # Should be close to 50/50 across the two canonical teams, not 90/10
    assert len(canonical_counts) == 2
    for count in canonical_counts.values():
        assert 220 <= count <= 380


def test_sampling_policy_uniform_archetype(tmp_path: Path) -> None:
    # 50 balance entries, 2 hyperoffense entries
    entries_bal = tuple(_make_entry(i, tags=("balance",)) for i in range(1, 51))
    entries_ho = tuple(_make_entry(i, tags=("hyperoffense",)) for i in range(51, 53))
    path, manifest = _write_manifest(tmp_path, entries_bal + entries_ho)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=300,
        sampling_policy=SamplingPolicy.UNIFORM_ARCHETYPE,
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(300)
    tag_counts: dict[str, int] = {"balance": 0, "hyperoffense": 0}
    for _ in range(600):
        t = source.sample(rng)
        e = next(entry for entry in entries_bal + entries_ho if entry.packed_sha256 == t.team_hash)
        tag_counts[e.archetype_tags[0]] += 1
    # Should be close to 50/50 across the two archetypes
    assert 220 <= tag_counts["balance"] <= 380
    assert 220 <= tag_counts["hyperoffense"] <= 380


def test_sampling_policy_rare_coverage(tmp_path: Path) -> None:
    e_common = _make_entry(1, usage_count=10000)
    e_rare = _make_entry(2, usage_count=1)
    path, manifest = _write_manifest(tmp_path, (e_common, e_rare))
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=400,
        sampling_policy=SamplingPolicy.RARE_COVERAGE,
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(400)
    counts = {e_common.packed_sha256: 0, e_rare.packed_sha256: 0}
    for _ in range(500):
        t = source.sample(rng)
        counts[t.team_hash] += 1
    # Rare should be sampled overwhelmingly more often when inversely weighted
    assert counts[e_rare.packed_sha256] > 490


def test_curriculum_stage_filtering(tmp_path: Path) -> None:
    e1 = _make_entry(1, tags=("balance",))
    e2 = _make_entry(2, tags=("hyperoffense",))
    e3 = _make_entry(3, tags=("trickroom",))
    path, manifest = _write_manifest(tmp_path, (e1, e2, e3))
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=500,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
        curriculum_stage="trickroom",
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(500)
    for _ in range(20):
        t = source.sample(rng)
        assert t.team_hash == e3.packed_sha256


def test_mirroring_control_sample_pair(tmp_path: Path) -> None:
    e1 = _make_entry(1, canonical_index=1)
    e2 = _make_entry(2, canonical_index=2)
    path, manifest = _write_manifest(tmp_path, (e1, e2))

    # With allow_mirror=False, sample_pair must always return two distinct canonical teams
    spec_no_mirror = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=600,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
        allow_mirror=False,
    )
    source_no_mirror = CorpusTeamSource(spec_no_mirror)
    rng = random.Random(600)
    for _ in range(50):
        t1, t2 = source_no_mirror.sample_pair(rng)
        # Check canonical hashes differ
        e1_obj = next(entry for entry in (e1, e2) if entry.packed_sha256 == t1.team_hash)
        e2_obj = next(entry for entry in (e1, e2) if entry.packed_sha256 == t2.team_hash)
        assert e1_obj.canonical_hash != e2_obj.canonical_hash

    # Single canonical team pool with allow_mirror=False raises ValueError on sample_pair
    path_single, manifest_single = _write_manifest(tmp_path, (e1,))
    spec_single = CorpusSourceSpec(
        corpus_path=str(path_single),
        corpus_hash=manifest_single.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=601,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
        allow_mirror=False,
    )
    source_single = CorpusTeamSource(spec_single)
    with pytest.raises(ValueError, match="Cannot sample non-mirror pair"):
        source_single.sample_pair(rng)


def test_sampling_policy_matchup_balanced_and_lazy_caching(tmp_path: Path) -> None:
    # Under USAGE_WEIGHTED policy, canonical and archetype indexes should remain uninitialized (None) until explicitly requested
    e1 = _make_entry(1, canonical_index=1, usage_count=100)
    e2 = _make_entry(2, canonical_index=2, usage_count=100)
    path, manifest = _write_manifest(tmp_path, (e1, e2))
    spec_usage = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=700,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    source_usage = CorpusTeamSource(spec_usage)
    assert source_usage._by_canonical is None
    assert source_usage._by_archetype is None
    # Sampling should succeed without building unneeded indexes
    rng = random.Random(700)
    assert source_usage.sample(rng) is not None

    # Under MATCHUP_BALANCED policy, canonical index should be built during prepare_sampling
    spec_matchup = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        seed=701,
        sampling_policy=SamplingPolicy.MATCHUP_BALANCED,
        allow_mirror=False,
    )
    source_matchup = CorpusTeamSource(spec_matchup)
    assert source_matchup._by_canonical is not None
    assert source_matchup._canonical_keys is not None
    assert len(source_matchup._canonical_keys) == 2
    # Verify non-mirror sampling with exclusion works cleanly under MATCHUP_BALANCED
    t1, t2 = source_matchup.sample_pair(rng)
    assert t1.team_hash != t2.team_hash
