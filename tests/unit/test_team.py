"""Tests for canonical team records and corpus construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.corpus_build import (
    audit_corpus,
    build_corpus,
    write_corpus_manifest,
)
from p0.teams.spread_usage import (
    SPREAD_USAGE_SCHEMA,
)
from p0.teams.stat_points import StatPoints
from p0.teams.team import (
    TeamMetadata,
    TeamRecord,
    deduplicate_variants,
)
from p0.teams.validation import validate_many
from tests.team_fixtures import metadata, team_variant


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


class TestTeam:
    def test_team_hash_ignores_display_and_member_order(self) -> None:
        """Verify CanonicalTeam team_hash computation is invariant to member permutation or nickname ordering."""
        first = team_variant()
        reversed_members = tuple(reversed(first.team.members))
        second = team_variant(members=reversed_members)
        assert first.team.team_hash == second.team.team_hash

    def test_deduplication_merges_metadata_but_preserves_spread_variants(self) -> None:
        """Verify deduplicate_variants merges replay/series metadata for identical teams while preserving distinct EV spread variants."""
        first = team_variant()
        duplicate = replace(first, metadata=metadata("series-2", 2))
        alternate = replace(
            first,
            spreads=tuple(StatPoints(hp=32, defense=17, spd=17) for _ in first.spreads),
        )
        result = deduplicate_variants((duplicate, alternate, first))
        assert len(result) == 2
        merged = next(item for item in result if item.spreads == first.spreads)
        assert merged.metadata.usage_count == 3
        assert merged.metadata.source_series == ("series-1", "series-2")

    def test_team_record_serialization_round_trip_is_strict(self) -> None:
        """Verify TeamRecord and TeamMetadata serialize and deserialize strictly, rejecting unknown fields."""
        variant = team_variant()
        assert TeamRecord.from_dict(variant.to_dict()) == replace(
            variant, team=variant.team.canonical()
        )
        assert TeamMetadata.from_dict(metadata().to_dict()) == metadata()
        with pytest.raises(ValueError, match="fields"):
            TeamRecord.from_dict({**variant.to_dict(), "unexpected": True})

    def test_corpus_builder_admits_valid_variants(self) -> None:
        """Verify build_corpus validates legal variants and generates coverage audit reports."""
        tokenizer = PokemonTokenizer(vocabulary())
        v1 = team_variant("Pikachu", usage_count=5)
        v2 = team_variant("Raichu", source_series=("series-2",), usage_count=3)
        manifest, audit = build_corpus(
            (v1, v2),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="a" * 64,
            format_id=FORMAT.battle_format,
        )
        assert len(manifest.entries) == 2
        assert manifest.format_id == FORMAT.battle_format
        assert manifest.global_contract_sha256 == "a" * 64
        assert audit["admitted_count"] == 2
        assert audit["rejected_count"] == 0
        assert set(audit["species_coverage"]) >= {"pikachu", "raichu"}

    def test_corpus_builder_rejects_oov_content(self) -> None:
        """Verify build_corpus filters valid team variants containing out-of-vocabulary content."""
        tokenizer = PokemonTokenizer(vocabulary())
        v_valid = team_variant("Pikachu")
        v_oov = team_variant("Pikachu", item="Leftovers")
        manifest, audit = build_corpus(
            (v_valid, v_oov),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="a" * 64,
        )
        assert len(manifest.entries) == 1
        assert audit["admitted_count"] == 1
        assert audit["rejected_count"] == 1
        assert "oov_item: Leftovers" in audit["rejections_by_reason"]

    def test_split_assignment_prevents_series_leakage(self) -> None:
        """Verify team variants originating from the same series ID are assigned to the same split (train/val/test)."""
        tokenizer = PokemonTokenizer(vocabulary())
        v1 = team_variant("Pikachu", source_series=("shared-series",))
        v2 = team_variant("Raichu", source_series=("shared-series",))
        v3 = team_variant("Pikachu", item="Life Orb", source_series=("other-series",))
        manifest, _ = build_corpus(
            (v1, v2, v3),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="a" * 64,
            ratio_train=0.5,
            ratio_val=0.5,
            ratio_test=0.0,
        )
        assert len(manifest.entries) == 3
        by_species = {entry.canonical_hash: entry.split for entry in manifest.entries}
        # Variants from shared-series must occupy identical split partition
        assert by_species[v1.team.team_hash] == by_species[v2.team.team_hash]
        assert by_species[v3.team.team_hash] in {CorpusSplit.TRAIN, CorpusSplit.VALIDATION}

    def test_audit_corpus_and_coverage(self) -> None:
        """Verify audit_corpus accurately aggregates admitted counts, species coverage sets, and split distribution."""
        tokenizer = PokemonTokenizer(vocabulary())
        v1 = team_variant("Pikachu", usage_count=10)
        v2 = team_variant("Raichu", source_series=("s2",), usage_count=5)
        manifest, audit = build_corpus(
            (v1, v2),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="b" * 64,
        )
        re_audit = audit_corpus(manifest)
        assert re_audit["admitted_count"] == 2
        assert "pikachu" in re_audit["species_coverage"]
        assert sum(re_audit["split_counts"].values()) == 2

    def test_write_corpus_manifest(self, tmp_path: Path) -> None:
        """Verify each pool gets only the manifest built from its own input directory."""
        tokenizer = PokemonTokenizer(vocabulary())
        # Each variant needs a unique species/item combination so canonical_hash is distinct.
        unique_variants = tuple(
            team_variant(
                species,
                item=item,
                source_series=(f"series-{i}",),
                usage_count=i * 10,
            )
            for i, (species, item) in enumerate(
                (
                    ("Pikachu", "Light Ball"),
                    ("Raichu", "Light Ball"),
                    ("Pikachu", "Life Orb"),
                    ("Raichu", "Life Orb"),
                ),
                start=1,
            )
        )
        manifest, _ = build_corpus(
            unique_variants,
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="c" * 64,
        )
        all_dir = tmp_path / "all"
        manifest_path = write_corpus_manifest(manifest, all_dir)

        assert manifest_path == all_dir / "corpus_manifest.json"
        assert manifest_path.is_file()

        manifest_all = TeamCorpusManifest.from_dict(json.loads(manifest_path.read_text()))

        assert len(manifest_all.entries) == len(manifest.entries)
