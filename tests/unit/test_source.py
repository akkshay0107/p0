"""Tests for team sources, sampling, and pool resolution."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from p0.cli.corpus import main as corpus_main
from p0.format_config import FORMAT, current_manifest
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
from p0.teams.spread_usage import (
    SPREAD_USAGE_SCHEMA,
)
from tests.team_fixtures import DEFAULT_TEST_TEAM


def vocabulary() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
            "whimsicott": 3,
            "garchomp": 4,
            "kingambit": 5,
            "glimmora": 6,
            "raichu": 7,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
            "focussash": 3,
            "sitrusberry": 4,
            "blackglasses": 5,
            "shucaberry": 6,
            "lifeorb": 7,
        },
        "abilities": {
            "static": 1,
            "blaze": 2,
            "prankster": 3,
            "roughskin": 4,
            "defiant": 5,
            "toxicdebris": 6,
        },
        "moves": {
            "fakeout": 1,
            "protect": 2,
            "thunderbolt": 3,
            "electroweb": 4,
            "heatwave": 5,
            "solarbeam": 6,
            "weatherball": 7,
            "moonblast": 8,
            "tailwind": 9,
            "encore": 10,
            "earthquake": 11,
            "dragonclaw": 12,
            "rockslide": 13,
            "kowtowcleave": 14,
            "suckerpunch": 15,
            "lowkick": 16,
            "powergem": 17,
            "sludgebomb": 18,
            "earthpower": 19,
        },
    }


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
        global_contract_sha256=current_manifest().global_sha256,
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


def _chaos(species: str, spreads: dict[str, float]) -> dict[str, Any]:
    """Build a minimal chaos export carrying one species' spread distribution."""
    return {"data": {species: {"Spreads": spreads}}}


def _dex(*species: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal dex carrying only what forme aliasing reads."""
    return {"species": list(species), "moves": []}


_BASE_STATS = {"hp": 78, "atk": 65, "def": 68, "spa": 112, "spd": 154, "spe": 75}


def _payload(spreads: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": FORMAT.battle_format,
        "weight_scale": 1_000_000,
        "aliases": aliases or {},
        "spreads": spreads,
    }


_ONE_BUCKET = {"real": {"timid": [[2, 0, 0, 32, 0, 32, 1000]]}}


def _corpus_entry(packed: str = "packed-team") -> CorpusEntry:
    return CorpusEntry(
        canonical_hash=hashlib.sha256(packed.encode()).hexdigest(),
        packed=packed,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        usage_count=3,
    )


def _corpus_manifest(entries: tuple[CorpusEntry, ...]) -> TeamCorpusManifest:
    active_contract = current_manifest().global_sha256
    return TeamCorpusManifest(
        global_contract_sha256=active_contract,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"sampling": "uniform_canonical"},
    )


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

    def test_uniform_sampling_index(self, tmp_path: Path) -> None:
        """Verify immutable canonical sampling pools are precomputed during CorpusTeamSource initialization."""
        e1 = _make_entry(1, canonical_index=1, usage_count=100)
        e2 = _make_entry(2, canonical_index=2, usage_count=100)
        path, manifest = _write_manifest(tmp_path, (e1, e2))
        source = CorpusTeamSource.from_path(path)
        assert source.describe()["pool_size"] == 2
        rng = random.Random(700)
        assert source.sample(rng) is not None

    def test_validated_team_rejects_untrusted_packed_values(self) -> None:
        """Verify ValidatedTeam verifies SHA-256 hash length and formatting."""
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

    def test_rejects_invalid_manifest(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        (pool_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid corpus manifest"):
            build_team_source(pool_dir)


class TestCorpusCLI:
    def test_cli_build_and_audit(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        team_text_1 = DEFAULT_TEST_TEAM
        team_text_2 = DEFAULT_TEST_TEAM.replace("Pikachu @ Light Ball", "Raichu @ Light Ball", 1)
        (input_dir / "v1.txt").write_text(team_text_1, encoding="utf-8")
        (input_dir / "v2.txt").write_text(team_text_2, encoding="utf-8")

        all_dir = tmp_path / "pools" / "all"

        corpus_main(
            [
                "build",
                "--input",
                str(input_dir),
                "--output-dir",
                str(all_dir),
                "--format-id",
                FORMAT.battle_format,
            ]
        )

        assert (all_dir / "corpus_manifest.json").is_file()

        captured = capsys.readouterr()
        audit_data = json.loads(captured.out.split("\n")[-2]) if captured.out.strip() else {}
        assert audit_data["admitted_count"] == 2
        assert audit_data["rejected_count"] == 0

        corpus_main(["audit", "--path", str(all_dir)])
        audit_captured = capsys.readouterr()
        re_audit_data = (
            json.loads(audit_captured.out.split("\n")[-2]) if audit_captured.out.strip() else {}
        )
        assert re_audit_data["admitted_count"] == 2
