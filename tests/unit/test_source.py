"""Tests for team sources and sampling."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from p0.format_config import FORMAT, current_manifest
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import ValidatedTeam
from p0.teams.spread_usage import (
    SPREAD_USAGE_SCHEMA,
)


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
    split: CorpusSplit = CorpusSplit.TRAIN,
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
        split=split,
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
        split=CorpusSplit.TRAIN,
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
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=FORMAT.battle_format,
            split=CorpusSplit.TRAIN,
        )
        source = CorpusTeamSource(spec)
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

    def test_corpus_source_validates_spec(self, tmp_path: Path) -> None:
        """Verify CorpusTeamSource rejects mismatched corpus hashes and format IDs."""
        entries = tuple(_make_entry(i) for i in range(3))
        path, manifest = _write_manifest(tmp_path, entries)

        # Wrong corpus_hash raises ValueError
        bad_spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash="0" * 64,
            format_id=FORMAT.battle_format,
            split=CorpusSplit.TRAIN,
        )
        with pytest.raises(ValueError, match="does not match"):
            CorpusTeamSource(bad_spec)

        # Wrong format_id raises ValueError
        bad_format = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id="wrong-format",
            split=CorpusSplit.TRAIN,
        )
        with pytest.raises(ValueError, match="format"):
            CorpusTeamSource(bad_format)

    def test_corpus_source_rejects_empty_filtered_pool(self, tmp_path: Path) -> None:
        """Verify CorpusTeamSource raises ValueError when split filter produces 0 available entries."""
        entries = tuple(_make_entry(i, split=CorpusSplit.TRAIN) for i in range(3))
        path, manifest = _write_manifest(tmp_path, entries)
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=FORMAT.battle_format,
            split=CorpusSplit.TEST,
        )
        with pytest.raises(ValueError, match="No corpus entries match"):
            CorpusTeamSource(spec)

    def test_uniform_canonical_sampling(self, tmp_path: Path) -> None:
        """Verify uniform canonical sampling equalizes archetype probabilities regardless of variant counts per archetype."""
        # 90 entries for canonical 1, 10 entries for canonical 2
        entries_1 = tuple(_make_entry(i, canonical_index=1, usage_count=100) for i in range(1, 91))
        entries_2 = tuple(
            _make_entry(i, canonical_index=2, usage_count=100) for i in range(91, 101)
        )
        path, manifest = _write_manifest(tmp_path, entries_1 + entries_2)
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=FORMAT.battle_format,
            split=CorpusSplit.TRAIN,
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

    def test_uniform_sampling_index(self, tmp_path: Path) -> None:
        """Verify immutable canonical sampling pools are precomputed during CorpusTeamSource initialization."""
        e1 = _make_entry(1, canonical_index=1, usage_count=100)
        e2 = _make_entry(2, canonical_index=2, usage_count=100)
        path, manifest = _write_manifest(tmp_path, (e1, e2))
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=FORMAT.battle_format,
            split=CorpusSplit.TRAIN,
        )
        source = CorpusTeamSource(spec)
        assert source.describe()["pool_size"] == 2
        rng = random.Random(700)
        assert source.sample(rng) is not None

    def test_validated_team_rejects_untrusted_packed_values(self) -> None:
        """Verify ValidatedTeam verifies SHA-256 hash length and formatting."""
        with pytest.raises(ValueError, match="SHA-256"):
            ValidatedTeam("packed", "short")
