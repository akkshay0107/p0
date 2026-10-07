"""Tests for team sources, sampling, and pool resolution."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from p0.format_config import FORMAT, active_global_contract
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.source import (
    CorpusTeamSource,
    FileTeamSource,
    FixedTeamSource,
    ValidatedTeam,
    build_team_source,
)
from tests.team_fixtures import DEFAULT_TEST_TEAM


def _make_entry(
    index: int,
    canonical_index: int | None = None,
    usage_count: int = 10,
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
        usage_count=usage_count,
        spread_provenance="imputed",
    )


def _write_manifest(
    tmp_path: Path, entries: tuple[CorpusEntry, ...]
) -> tuple[Path, TeamCorpusManifest]:
    manifest = TeamCorpusManifest(
        artifact_schema=CORPUS_MANIFEST_SCHEMA,
        global_contract_sha256=active_global_contract().global_sha256,
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


class TestTeamSources:
    def test_corpus_source_implements_protocol_and_describes(self, tmp_path: Path) -> None:
        """Verify CorpusTeamSource samples legal teams and provides accurate metadata descriptions."""
        entries = tuple(_make_entry(i) for i in range(5))
        path, manifest = _write_manifest(tmp_path, entries)
        source = CorpusTeamSource.from_path(path)
        rng = random.Random(42)
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

    def test_corpus_source_validates_path(self, tmp_path: Path) -> None:
        """Verify CorpusTeamSource rejects nonexistent paths and empty manifests."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            CorpusTeamSource.from_path(nonexistent)

        path, _ = _write_manifest(tmp_path, ())
        with pytest.raises(ValueError, match="no entries"):
            CorpusTeamSource.from_path(path)

    def test_uniform_canonical_sampling(self, tmp_path: Path) -> None:
        """Verify uniform canonical sampling equalizes archetype probabilities regardless of variant counts per archetype."""
        # 90 entries for canonical 1, 10 entries for canonical 2
        entries_1 = tuple(_make_entry(i, canonical_index=1, usage_count=100) for i in range(1, 91))
        entries_2 = tuple(
            _make_entry(i, canonical_index=2, usage_count=100) for i in range(91, 101)
        )
        path, manifest = _write_manifest(tmp_path, entries_1 + entries_2)
        source = CorpusTeamSource.from_path(path)
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

    def test_validated_team_requires_64_character_sha256(self) -> None:
        """Verify ValidatedTeam rejects a packed-team hash shorter than 64 characters."""
        with pytest.raises(ValueError, match="SHA-256"):
            ValidatedTeam("packed", "short")

    def test_fixed_team_source(self) -> None:
        team = ValidatedTeam.from_showdown(DEFAULT_TEST_TEAM)
        source = FixedTeamSource(team)
        rng = random.Random(0)
        assert source.sample(rng) == team
        assert source.describe()["kind"] == "fixed"


class TestBuildTeamSource:
    def test_resolves_corpus_manifest(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        manifest_path = tmp_path / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

        source = build_team_source(manifest_path)
        assert isinstance(source, CorpusTeamSource)
        assert source.describe()["kind"] == "corpus"

    def test_accepts_regular_manifest_for_bo3(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        source = build_team_source(path, expected_format_id=FORMAT.bo3_format)
        assert isinstance(source, CorpusTeamSource)
        assert source.describe()["format_id"] == manifest.format_id

    def test_rejects_incompatible_manifest_format(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        incompatible = replace(manifest, format_id="unsupported-format")
        path.write_text(json.dumps(incompatible.to_dict()), encoding="utf-8")

        with pytest.raises(ValueError, match="Corpus format mismatch"):
            build_team_source(path, expected_format_id=FORMAT.bo3_format)

    def test_falls_back_to_file_source(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        team_text = "\n\n".join(
            f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
            for i in range(1, 7)
        )
        (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
        source = build_team_source(pool_dir)
        assert isinstance(source, FileTeamSource)
        expected_team = ValidatedTeam.from_showdown(team_text)
        rng = random.Random(0)
        sampled = source.sample(rng)
        assert sampled.packed == expected_team.packed
        assert sampled.team_hash == expected_team.team_hash
        assert source.describe() == {
            "kind": "file_pool",
            "format": "showdown-export",
            "pool_id": hashlib.sha256(expected_team.team_hash.encode()).hexdigest(),
            "team_hashes": (expected_team.team_hash,),
        }

    def test_rejects_invalid_manifest(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        (pool_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid corpus manifest"):
            build_team_source(pool_dir)
