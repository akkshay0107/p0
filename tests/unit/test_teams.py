from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson
import pytest

from p0.cli.build_spreads import DEFAULT_CUTOFF, DEFAULT_MONTH, USAGE_URL
from p0.cli.corpus import main as corpus_main
from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)
from p0.teams.corpus_build import (
    audit_corpus,
    build_corpus,
    write_corpus_manifest,
)
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.factory import build_team_source
from p0.teams.source import FileTeamSource, ValidatedTeam
from p0.teams.spread_usage import (
    BO3_BLEND_WEIGHT,
    DEFAULT_SPREAD_TABLE_PATH,
    IMPUTED_FROM_FALLBACK,
    IMPUTED_FROM_USAGE,
    SPREAD_USAGE_SCHEMA,
    build_spread_table,
    cosmetic_forme_aliases,
    load_spread_table,
    load_spread_table_file,
    parse_spread_key,
)
from p0.teams.stat_points import BaseStats, StatPoints, calculate_stats, fallback_points
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


def test_team_hash_ignores_display_and_member_order() -> None:
    """Verify CanonicalTeam team_hash computation is invariant to member permutation or nickname ordering."""
    first = team_variant()
    reversed_members = tuple(reversed(first.team.members))
    second = team_variant(members=reversed_members)
    assert first.team.team_hash == second.team.team_hash


def test_deduplication_merges_metadata_but_preserves_spread_variants() -> None:
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


def test_team_record_serialization_round_trip_is_strict() -> None:
    """Verify TeamRecord and TeamMetadata serialize and deserialize strictly, rejecting unknown fields."""
    variant = team_variant()
    assert TeamRecord.from_dict(variant.to_dict()) == replace(
        variant, team=variant.team.canonical()
    )
    assert TeamMetadata.from_dict(metadata().to_dict()) == metadata()
    with pytest.raises(ValueError, match="fields"):
        TeamRecord.from_dict({**variant.to_dict(), "unexpected": True})


def test_corpus_builder_admits_valid_variants() -> None:
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


def test_corpus_builder_rejects_oov_content() -> None:
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


def test_split_assignment_prevents_series_leakage() -> None:
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


def test_audit_corpus_and_coverage() -> None:
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


def test_write_corpus_manifest(tmp_path: Path) -> None:
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


def test_corpus_source_implements_protocol_and_describes(tmp_path: Path) -> None:
    """Verify CorpusTeamSource satisfies the TeamSource protocol and provides accurate metadata descriptions."""
    entries = tuple(_make_entry(i) for i in range(5))
    path, manifest = _write_manifest(tmp_path, entries)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
    )
    source = CorpusTeamSource(spec)
    assert hasattr(source, "sample") and callable(source.sample)
    assert hasattr(source, "describe") and callable(source.describe)
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


def test_corpus_source_validates_spec(tmp_path: Path) -> None:
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


def test_corpus_source_rejects_empty_filtered_pool(tmp_path: Path) -> None:
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


def test_uniform_canonical_sampling(tmp_path: Path) -> None:
    """Verify uniform canonical sampling equalizes archetype probabilities regardless of variant counts per archetype."""
    # 90 entries for canonical 1, 10 entries for canonical 2
    entries_1 = tuple(_make_entry(i, canonical_index=1, usage_count=100) for i in range(1, 91))
    entries_2 = tuple(_make_entry(i, canonical_index=2, usage_count=100) for i in range(91, 101))
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


def test_uniform_sampling_index(tmp_path: Path) -> None:
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


def test_validated_team_rejects_untrusted_packed_values():
    """Verify ValidatedTeam verifies SHA-256 hash length and formatting."""
    with pytest.raises(ValueError, match="SHA-256"):
        ValidatedTeam("packed", "short")


def _chaos(species: str, spreads: dict[str, float]) -> dict[str, Any]:
    """Build a minimal chaos export carrying one species' spread distribution."""
    return {"data": {species: {"Spreads": spreads}}}


def _dex(*species: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal dex carrying only what forme aliasing reads."""
    return {"species": list(species), "moves": []}


_BASE_STATS = {"hp": 78, "atk": 65, "def": 68, "spa": 112, "spd": 154, "spe": 75}


def test_parse_spread_key_rejects_illegal_and_malformed() -> None:
    """Verify parse_spread_key enforces stat cap of 32, total budget of 66, and 6 stat components."""
    parsed = parse_spread_key("Impish:32/0/21/0/11/2")
    assert parsed is not None
    nature, points = parsed
    assert nature == "impish"
    assert points == StatPoints(hp=32, defense=21, spd=11, spe=2)

    assert parse_spread_key("Impish:33/0/0/0/0/0") is None, "per-stat cap of 32"
    assert parse_spread_key("Impish:32/32/32/0/0/0") is None, "66-point budget"
    assert parse_spread_key("Impish:32/0/0/0/0") is None, "needs six stats"
    assert parse_spread_key("Impish:a/0/0/0/0/0") is None
    assert parse_spread_key("32/0/0/0/0/0") is None, "needs a nature"


def test_spread_table_blends_shared_buckets_toward_bo3() -> None:
    """Verify spread table blends Bo3 and Bo1 distributions according to BO3_BLEND_WEIGHT (0.8/0.2)."""
    only_bo3 = "Timid:2/0/0/32/0/32"
    only_bo1 = "Timid:32/0/0/32/0/2"
    payload = build_spread_table(
        _chaos("Whimsicott", {only_bo1: 10.0}),
        _chaos("Whimsicott", {only_bo3: 1.0}),
        format_id=FORMAT.battle_format,
        dex=_dex(),
    )
    bucket = load_spread_table(payload).lookup("Whimsicott", "timid")

    # Each source normalizes to 1.0 first, so the raw 10:1 count gap is irrelevant
    # and the blend lands on the configured 0.8/0.2 split.
    assert [prior.points for prior in bucket] == [
        StatPoints(hp=2, spa=32, spe=32),
        StatPoints(hp=32, spa=32, spe=2),
    ]
    assert bucket[0].weight == pytest.approx(BO3_BLEND_WEIGHT, abs=1e-4)
    assert bucket[1].weight == pytest.approx(1.0 - BO3_BLEND_WEIGHT, abs=1e-4)


def test_spread_table_uses_single_source_when_bucket_is_absent() -> None:
    """Verify spread table adopts 100% weight when a spread appears in only one format export."""
    payload = build_spread_table(
        _chaos("Sylveon", {"Calm:32/0/0/0/32/2": 5.0}),
        _chaos("Incineroar", {"Impish:32/0/32/0/2/0": 5.0}),
        format_id=FORMAT.battle_format,
        dex=_dex(),
    )
    table = load_spread_table(payload)

    # A bucket only one export carries is taken whole rather than down-weighted.
    assert table.best("Sylveon", "calm") == StatPoints(hp=32, spd=32, spe=2)
    assert table.best("Incineroar", "impish") == StatPoints(hp=32, defense=32, spd=2)
    assert table.lookup("Sylveon", "calm")[0].weight == pytest.approx(1.0)


def test_spread_table_prunes_rare_natures() -> None:
    """Verify build_spread_table prunes natures below min_nature_share threshold (e.g. 5%)."""
    common = "Timid:2/0/0/32/0/32"
    rare = "Hardy:32/0/32/0/2/0"
    export = _chaos("Gholdengo", {common: 99.0, rare: 1.0})
    table = load_spread_table(
        build_spread_table(
            export, export, format_id=FORMAT.battle_format, dex=_dex(), min_nature_share=0.05
        )
    )

    assert table.best("Gholdengo", "timid") is not None
    assert table.best("Gholdengo", "hardy") is None, "1% share is below the 5% floor"

    kept = load_spread_table(
        build_spread_table(
            export, export, format_id=FORMAT.battle_format, dex=_dex(), min_nature_share=0.0
        )
    )
    assert kept.best("Gholdengo", "hardy") is not None


def test_spread_table_truncates_and_renormalizes() -> None:
    """Verify spread tables truncate entries to max_spreads per bucket and re-normalize probabilities to 1.0."""
    # Each allocation sums to 64, so all twenty are legal and distinct.
    spreads = {f"Jolly:{n}/{32 - n}/0/0/0/32": float(20 - n) for n in range(20)}
    export = _chaos("Garchomp", spreads)
    bucket = load_spread_table(
        build_spread_table(
            export, export, format_id=FORMAT.battle_format, dex=_dex(), max_spreads=5
        )
    ).lookup("Garchomp", "jolly")

    assert len(bucket) == 5
    assert sum(prior.weight for prior in bucket) == pytest.approx(1.0)
    # Stored weight-descending, so the argmax is entry zero.
    assert [prior.weight for prior in bucket] == sorted(
        (prior.weight for prior in bucket), reverse=True
    )


def test_spread_table_rejects_unsupported_schema() -> None:
    """Verify load_spread_table rejects invalid or unsupported schema versions."""
    payload = build_spread_table(
        _chaos("Garchomp", {"Jolly:2/32/0/0/0/32": 1.0}),
        _chaos("Garchomp", {"Jolly:2/32/0/0/0/32": 1.0}),
        format_id=FORMAT.battle_format,
        dex=_dex(),
    )
    payload["schema"] = SPREAD_USAGE_SCHEMA + 1
    with pytest.raises(ValueError, match="schema"):
        load_spread_table(payload)


def test_spread_table_lookup_normalizes_species_and_nature() -> None:
    """Verify spread table lookups normalize species names and natures case-insensitively."""
    payload = build_spread_table(
        _chaos("Charizard-Mega-Y", {"Timid:2/0/0/32/0/32": 1.0}),
        _chaos("Charizard-Mega-Y", {"Timid:2/0/0/32/0/32": 1.0}),
        format_id=FORMAT.battle_format,
        dex=_dex(),
    )
    table = load_spread_table(payload)

    expected = StatPoints(hp=2, spa=32, spe=32)
    assert table.best("Charizard-Mega-Y", "Timid") == expected
    assert table.best("charizardmegay", "timid") == expected
    assert table.best("Missingno", "timid") is None
    assert table.lookup("Missingno", "timid") == ()


def test_cosmetic_formes_alias_onto_their_base_species() -> None:
    """Verify cosmetic forme variations (e.g. Florges colours) automatically alias to base species."""
    dex = _dex(
        {
            "id": "florges",
            "name": "Florges",
            "baseSpecies": "Florges",
            "baseStats": _BASE_STATS,
            "formeOrder": ["Florges", "Florges-Blue", "Florges-White"],
        }
    )
    aliases = cosmetic_forme_aliases(dex)

    # Colour variants carry no dex record of their own, so they inherit the base.
    assert aliases["florgesblue"] == "florges"
    assert aliases["florgeswhite"] == "florges"
    assert "florges" not in aliases


def test_formes_with_distinct_base_stats_are_never_aliased() -> None:
    """Verify forme variations with distinct base stats (e.g. Floette-Eternal) are excluded from cosmetic aliasing."""
    eternal_stats = {"hp": 74, "atk": 65, "def": 67, "spa": 125, "spd": 128, "spe": 92}
    dex = _dex(
        {
            "id": "floette",
            "name": "Floette",
            "baseSpecies": "Floette",
            "baseStats": {"hp": 54, "atk": 45, "def": 47, "spa": 75, "spd": 98, "spe": 52},
            "formeOrder": ["Floette", "Floette-Blue", "Floette-Eternal"],
            "otherFormes": ["Floette-Eternal"],
        },
        {
            "id": "floetteeternal",
            "name": "Floette-Eternal",
            "baseSpecies": "Floette",
            "baseStats": eternal_stats,
        },
    )
    aliases = cosmetic_forme_aliases(dex)

    assert aliases["floetteblue"] == "floette", "cosmetic colour still folds"
    assert "floetteeternal" not in aliases, "different base stats must stay separate"


def test_aliases_resolve_in_one_hop_without_cycles() -> None:
    """Verify forme aliases resolve cleanly in a single lookup step without alias chaining or cycles."""
    shared = dict(_BASE_STATS)
    dex = _dex(
        *(
            {
                "id": f"alcremie{flavour}",
                "name": f"Alcremie-{flavour}",
                "baseSpecies": "Alcremie",
                "baseStats": shared,
                "formeOrder": ["Alcremie-ruby", "Alcremie-matcha"],
            }
            for flavour in ("ruby", "matcha")
        ),
        {"id": "alcremie", "name": "Alcremie", "baseSpecies": "Alcremie", "baseStats": shared},
    )
    aliases = cosmetic_forme_aliases(dex)

    assert aliases["alcremieruby"] == "alcremie"
    assert aliases["alcremiematcha"] == "alcremie"
    assert not [key for key, value in aliases.items() if value in aliases], "no alias chains"
    assert not [key for key, value in aliases.items() if key == value], "no self references"


def test_aliased_forme_shares_the_base_species_bucket() -> None:
    """Verify aliased cosmetic formes return the identical memory object as their base species."""
    dex = _dex(
        {
            "id": "florges",
            "name": "Florges",
            "baseSpecies": "Florges",
            "baseStats": _BASE_STATS,
            "formeOrder": ["Florges", "Florges-Blue"],
        }
    )
    export = _chaos("Florges", {"Modest:32/0/20/12/2/0": 1.0})
    table = load_spread_table(
        build_spread_table(export, export, format_id=FORMAT.battle_format, dex=dex)
    )

    expected = StatPoints(hp=32, defense=20, spa=12, spd=2)
    assert table.best("Florges", "modest") == expected
    assert table.best("Florges-Blue", "modest") == expected

    # The alias shares the base tuple rather than duplicating rows in the artifact.
    assert table.lookup("Florges-Blue", "modest") is table.lookup("Florges", "modest")


def _payload(spreads: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": FORMAT.battle_format,
        "weight_scale": 1_000_000,
        "aliases": aliases or {},
        "spreads": spreads,
    }


_ONE_BUCKET = {"real": {"timid": [[2, 0, 0, 32, 0, 32, 1000]]}}


def test_load_rejects_bucket_that_is_not_weight_descending() -> None:
    """Verify load_spread_table rejects spread buckets that are not strictly sorted in descending weight order."""
    unsorted_rows = {"real": {"timid": [[32, 0, 0, 32, 0, 2, 100], [2, 0, 0, 32, 0, 32, 900]]}}
    with pytest.raises(ValueError, match="weight-descending"):
        load_spread_table(_payload(unsorted_rows))


def test_load_rejects_unresolvable_aliases() -> None:
    """Verify load_spread_table rejects missing targets or alias chains."""
    with pytest.raises(ValueError, match="unknown species"):
        load_spread_table(_payload(_ONE_BUCKET, {"ghost": "missing"}))
    with pytest.raises(ValueError, match="unknown species|through another alias"):
        load_spread_table(_payload(_ONE_BUCKET, {"first": "second", "second": "real"}))


def test_load_accepts_a_single_hop_alias() -> None:
    """Verify load_spread_table accepts single-hop valid cosmetic aliases."""
    table = load_spread_table(_payload(_ONE_BUCKET, {"cosmetic": "real"}))
    assert table.best("cosmetic", "timid") == table.best("real", "timid")


def test_usage_export_url_matches_the_published_layout() -> None:
    """Verify Smogon stats URL formatter generates valid external chaos data paths."""
    url = USAGE_URL.format(month="2026-07", format_id="gen9championsvgc2026regmb", cutoff=1760)
    assert url == (
        "https://www.smogon.com/stats/2026-07/chaos/gen9championsvgc2026regmb-1760.json.gz"
    )


def test_shipped_spread_table_records_the_exports_it_came_from() -> None:
    """Verify committed spread_usage.json records source month, format filenames, and SHA-256 digests."""
    payload = orjson.loads(DEFAULT_SPREAD_TABLE_PATH.read_bytes())
    source = payload["source"]

    assert source["month"] == DEFAULT_MONTH
    assert set(source["exports"]) == {
        f"{FORMAT.battle_format}-{DEFAULT_CUTOFF}.json",
        f"{FORMAT.bo3_format}-{DEFAULT_CUTOFF}.json",
    }
    assert all(len(digest) == 64 for digest in source["exports"].values())


def test_first_point_and_nature_truncation_match_showdown():
    """Verify calculate_stats stat formulas match Showdown Gen 9 stat calculations at level 50."""
    base = BaseStats(100, 100, 100, 100, 100, 100)
    zero = calculate_stats(base, StatPoints(), "adamant")
    one = calculate_stats(base, StatPoints(atk=1), "adamant")
    assert zero == (175, 132, 120, 108, 120, 120)
    assert one[1] == zero[1] + 1


@pytest.mark.parametrize(
    "points, error",
    [
        (dict(hp=33), r"\[0, 32\]"),
        (dict(hp=32, atk=32, defense=3), "at most 66"),
    ],
)
def test_stat_point_validation(points, error):
    """Verify StatPoints enforces 0..32 per-stat bounds and 66 total point budget."""
    with pytest.raises(ValueError, match=error):
        StatPoints(**points)


def test_fallback_follows_move_category_priority() -> None:
    """Verify fallback_points allocates 66 points using category priority: physical > special > status."""
    physical = StatPoints(hp=32, atk=32, spe=2)
    special = StatPoints(hp=32, spa=32, spe=2)
    status = StatPoints(hp=32, defense=17, spd=17)

    assert fallback_points(("physical", "physical", "status", "status")) == physical
    assert fallback_points(("special", "special", "status", "status")) == special
    assert fallback_points(("status", "status", "special", "physical")) == status

    # Physical wins a tie with special regardless of which stat the species favours.
    assert fallback_points(("physical", "physical", "special", "special")) == physical

    # Every shape spends the full budget and stays legal.
    for categories in (("physical",) * 2, ("special",) * 2, ("status",) * 2):
        points = fallback_points(categories)
        assert points is not None
        assert sum(points.as_tuple()) == 66


def test_fallback_is_unknown_when_no_category_reaches_two() -> None:
    """Verify fallback_points returns None when known move categories are insufficient to establish archetype."""
    assert fallback_points(("physical", "special", "status")) is None
    assert fallback_points(()) is None
    assert fallback_points(("physical",)) is None


def test_usage_prior_beats_fallback_and_reports_confidence() -> None:
    """Verify resolve() prioritizes empirical usage table over generic category fallback."""
    table = load_spread_table_file()
    categories = ("physical", "status", "physical", "status")

    covered = table.resolve("Incineroar", "Impish", categories)
    assert covered is not None
    assert covered.origin == IMPUTED_FROM_USAGE
    assert 0.0 < covered.confidence <= 1.0
    assert sum(covered.points.as_tuple()) <= 66

    # The prior is consulted before the fallback, so a covered species must not
    # collapse onto the generic physical shape.
    assert covered.points != StatPoints(hp=32, atk=32, spe=2)

    uncovered = table.resolve("Missingno", "Impish", categories)
    assert uncovered is not None
    assert uncovered.origin == IMPUTED_FROM_FALLBACK
    assert uncovered.confidence == 0.0
    assert uncovered.points == StatPoints(hp=32, atk=32, spe=2)


def test_usage_prior_resolution_is_deterministic() -> None:
    """Verify resolve() produces identical StatPoints across multiple invocations."""
    table = load_spread_table_file()
    categories = ("special", "status", "special", "status")
    first = table.resolve("Charizard-Mega-Y", "Timid", categories)
    second = table.resolve("Charizard-Mega-Y", "Timid", categories)
    assert first == second


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


def test_corpus_manifest_contract() -> None:
    """Verify TeamCorpusManifest serializes losslessly and enforces canonical hash contracts."""
    entries = (_corpus_entry("team-a"), _corpus_entry("team-b"))
    assert entries[0].spread_provenance == "imputed"
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
    with pytest.raises(ValueError, match="unknown"):
        CorpusEntry.from_dict({**entries[0].to_dict(), "archetype_tags": []})


def test_corpus_source_spec_validates() -> None:
    """Verify CorpusSourceSpec validates split parameter."""
    spec = CorpusSourceSpec(
        corpus_path="teams/corpus_manifest.json",
        corpus_hash="a" * 64,
        format_id="gen9championsvgc2026regmb",
        split=CorpusSplit.TRAIN,
    )
    assert spec.split is CorpusSplit.TRAIN
    with pytest.raises(ValueError, match="split"):
        CorpusSourceSpec(
            corpus_path="x",
            corpus_hash="a" * 64,
            format_id="f",
            split=CorpusSplit.UNSPECIFIED,
        )


def test_build_team_source_resolves_corpus_manifest(tmp_path: Path) -> None:
    """Verify the team-source factory instantiates CorpusTeamSource for a manifest path."""
    tokenizer = PokemonTokenizer(vocabulary())
    contract_hash = current_manifest().global_sha256
    v1 = team_variant("Pikachu")
    manifest, _ = build_corpus(
        (v1,),
        tokenizer=tokenizer,
        validator=validate_many,
        global_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    source = build_team_source(manifest_path)
    assert isinstance(source, CorpusTeamSource)
    desc = source.describe()
    assert desc["kind"] == "corpus"
    assert desc["corpus_hash"] == manifest.corpus_hash
    assert desc["split"] == "TRAIN"
    assert desc["sampling"] == "uniform_canonical"


def test_build_team_source_accepts_regular_manifest_for_bo3(tmp_path: Path) -> None:
    """Verify a regular-format corpus can feed a Bo3 model."""
    path, manifest = _write_manifest(tmp_path, (_make_entry(1),))

    source = build_team_source(path, expected_format_id=FORMAT.bo3_format)
    assert isinstance(source, CorpusTeamSource)
    assert source.describe()["format_id"] == manifest.format_id


def test_build_team_source_rejects_incompatible_manifest_format(tmp_path: Path) -> None:
    """Verify unsupported corpus and model format pairs fail loudly."""
    path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
    incompatible = replace(manifest, format_id="unsupported-format")
    path.write_text(json.dumps(incompatible.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="Corpus format mismatch"):
        build_team_source(path, expected_format_id=FORMAT.bo3_format)


def test_build_team_source_falls_back_to_file_source(tmp_path: Path) -> None:
    """Verify the team-source factory falls back for a raw team directory."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    team_text = "\n\n".join(
        f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
        for i in range(1, 7)
    )
    (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
    source = build_team_source(pool_dir)
    assert isinstance(source, FileTeamSource)


def test_build_team_source_rejects_invalid_manifest(tmp_path: Path) -> None:
    """Verify a present manifest is authoritative and malformed content raises."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid corpus manifest"):
        build_team_source(pool_dir)


def test_corpus_cli_build_and_audit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify corpus CLI subcommands ('build' and 'audit') execute and output valid JSON audit results."""
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


def test_build_team_source_resolves_directory_manifest(tmp_path: Path) -> None:
    """Verify the team-source factory resolves a manifest inside a pool directory."""
    tokenizer = PokemonTokenizer(vocabulary())
    contract_hash = current_manifest().global_sha256
    v1 = team_variant("Pikachu")
    manifest, _ = build_corpus(
        (v1,),
        tokenizer=tokenizer,
        validator=validate_many,
        global_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    pool_dir = tmp_path / "pool_all"
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "corpus_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    (pool_dir / "invalid-team.txt").write_text("not a team", encoding="utf-8")

    source = build_team_source(pool_dir)
    assert isinstance(source, CorpusTeamSource)


def test_corpus_manifest_hash_is_order_independent_but_packed_content_bound() -> None:
    """Verify corpus content hash calculation is invariant to entry permutation but sensitive to packed team changes."""
    entries = tuple(
        CorpusEntry(
            canonical_hash=hashlib.sha256(f"canonical-{letter}".encode()).hexdigest(),
            packed=f"team-{letter}",
            packed_sha256=hashlib.sha256(f"team-{letter}".encode()).hexdigest(),
            split=CorpusSplit.TRAIN,
            usage_count=index + 1,
        )
        for index, letter in enumerate(("a", "b", "c"))
    )
    manifest = TeamCorpusManifest(
        global_contract_sha256="d" * 64,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"seed": 3},
    )
    assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
    assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
    with pytest.raises(ValueError, match="does not match"):
        TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "e" * 64})
