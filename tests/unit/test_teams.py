from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    SamplingPolicy,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.corpus_build import (
    _component_splits,
    audit_corpus,
    build_corpus,
    populate_pool_directories,
)
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FileTeamSource, FixedTeamSource, ValidatedTeam
from p0.teams.stat_points import StatPoints
from p0.teams.team import (
    CanonicalTeam,
    TeamMember,
    TeamMetadata,
    TeamRecord,
    deduplicate_variants,
    validate_evidence_cutoff,
)
from p0.teams.validation import (
    AdmissionResult,
    PersistentShowdownValidator,
    validate_many,
    validate_many_batched,
    validate_variant,
)


def _members():
    return (
        TeamMember(
            "Pikachu",
            "Light Ball",
            "Static",
            ("Fake Out", "Protect", "Thunderbolt", "Electroweb"),
            "Jolly",
        ),
        TeamMember(
            "Charizard",
            "Charizardite Y",
            "Blaze",
            ("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
            "Modest",
        ),
        TeamMember(
            "Whimsicott",
            "Focus Sash",
            "Prankster",
            ("Moonblast", "Tailwind", "Encore", "Protect"),
            "Timid",
        ),
        TeamMember(
            "Garchomp",
            "Sitrus Berry",
            "Rough Skin",
            ("Earthquake", "Dragon Claw", "Rock Slide", "Protect"),
            "Jolly",
        ),
        TeamMember(
            "Kingambit",
            "Black Glasses",
            "Defiant",
            ("Kowtow Cleave", "Sucker Punch", "Protect", "Low Kick"),
            "Adamant",
        ),
        TeamMember(
            "Glimmora",
            "Shuca Berry",
            "Toxic Debris",
            ("Power Gem", "Sludge Bomb", "Earth Power", "Protect"),
            "Modest",
        ),
    )


def _metadata(source="series-1", usage=1):
    return TeamMetadata(
        source_series=(source,),
        source_replays=(f"{source}-game-1",),
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-02T00:00:00Z",
        usage_count=usage,
        archetype_tags=("balance",),
    )


def _variant_team_corpus(members=None, metadata=None, spreads=None):
    members = members or _members()
    return TeamRecord(
        CanonicalTeam(tuple(members)),
        spreads or tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata or _metadata(),
    )


def test_team_hash_ignores_display_and_member_order():
    first = _variant_team_corpus()
    reversed_members = tuple(reversed(_members()))
    second = _variant_team_corpus(reversed_members)
    assert first.team.team_hash == second.team.team_hash


def test_deduplication_merges_metadata_but_preserves_spread_variants():
    first = _variant_team_corpus()
    duplicate = replace(first, metadata=_metadata("series-2", 2))
    alternate = replace(
        first,
        spreads=tuple(StatPoints(hp=32, defense=17, spd=17) for _ in first.spreads),
    )
    result = deduplicate_variants((duplicate, alternate, first))
    assert len(result) == 2
    merged = next(item for item in result if item.spreads == first.spreads)
    assert merged.metadata.usage_count == 3
    assert merged.metadata.source_series == ("series-1", "series-2")


def test_team_record_serialization_round_trip_is_strict():
    variant = _variant_team_corpus()
    assert TeamRecord.from_dict(variant.to_dict()) == replace(
        variant, team=variant.team.canonical()
    )
    with pytest.raises(ValueError, match="fields"):
        TeamRecord.from_dict({**variant.to_dict(), "unexpected": True})


def test_opponent_evidence_rejects_future_games():
    validate_evidence_cutoff(
        own_team=False, game_number=2, event_index=20, evidence_game=2, evidence_event=20
    )
    validate_evidence_cutoff(
        own_team=True, game_number=1, event_index=0, evidence_game=3, evidence_event=99
    )
    with pytest.raises(ValueError, match="future"):
        validate_evidence_cutoff(
            own_team=False, game_number=1, event_index=10, evidence_game=1, evidence_event=11
        )


def _mock_vocab_team_corpus_build() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
            "whimsicott": 3,
            "garchomp": 4,
            "kingambit": 5,
            "glimmora": 6,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
            "focussash": 3,
            "sitrusberry": 4,
            "blackglasses": 5,
            "shucaberry": 6,
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


def _variant_team_corpus_build(
    species: str = "Pikachu",
    source_series: tuple[str, ...] = ("series-1",),
    usage_count: int = 1,
    archetypes: tuple[str, ...] = ("balance",),
    move: str = "Fake Out",
) -> TeamRecord:
    members = (
        TeamMember(
            species=species,
            item="Light Ball",
            ability="Static",
            moves=(move, "Protect", "Thunderbolt", "Electroweb"),
            nature="Jolly",
        ),
        TeamMember(
            species="Charizard",
            item="Charizardite Y",
            ability="Blaze",
            moves=("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
            nature="Modest",
        ),
        TeamMember(
            species="Whimsicott",
            item="Focus Sash",
            ability="Prankster",
            moves=("Moonblast", "Tailwind", "Encore", "Protect"),
            nature="Timid",
        ),
        TeamMember(
            species="Garchomp",
            item="Sitrus Berry",
            ability="Rough Skin",
            moves=("Earthquake", "Dragon Claw", "Rock Slide", "Protect"),
            nature="Jolly",
        ),
        TeamMember(
            species="Kingambit",
            item="Black Glasses",
            ability="Defiant",
            moves=("Kowtow Cleave", "Sucker Punch", "Protect", "Low Kick"),
            nature="Adamant",
        ),
        TeamMember(
            species="Glimmora",
            item="Shuca Berry",
            ability="Toxic Debris",
            moves=("Power Gem", "Sludge Bomb", "Earth Power", "Protect"),
            nature="Modest",
        ),
    )
    return TeamRecord(
        team=CanonicalTeam(members),
        spreads=tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=TeamMetadata(
            source_series=source_series,
            source_replays=("series-1-game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=usage_count,
            archetype_tags=archetypes,
        ),
    )


def _mock_validator(variants: Sequence[TeamRecord], **kwargs: Any) -> tuple[AdmissionResult, ...]:
    return tuple(
        AdmissionResult(
            team_hash=variant.team.team_hash,
            valid=True,
            packed_team="]".join(
                f"{m.species}|{m.species}|{m.item}|{m.ability}|{','.join(m.moves)}|{m.nature}"
                for m in variant.team.members
            ),
            problems=(),
        )
        for variant in variants
    )


def test_corpus_builder_admits_valid_variants() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())
    v1 = _variant_team_corpus_build("Pikachu", usage_count=5)
    v2 = _variant_team_corpus_build("Charizard", source_series=("series-2",), usage_count=3)
    manifest, audit = build_corpus(
        (v1, v2),
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256="a" * 64,
        format_id=FORMAT.battle_format,
    )
    assert len(manifest.entries) == 2
    assert manifest.format_id == FORMAT.battle_format
    assert manifest.global_contract_sha256 == "a" * 64
    assert audit["admitted_count"] == 2
    assert audit["rejected_count"] == 0
    assert set(audit["species_coverage"]) >= {"pikachu", "charizard"}


def test_corpus_builder_rejects_oov_species() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())
    v_valid = _variant_team_corpus_build("Pikachu")
    v_oov = _variant_team_corpus_build("Missingno")
    manifest, audit = build_corpus(
        (v_valid, v_oov),
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256="a" * 64,
    )
    assert len(manifest.entries) == 1
    assert audit["admitted_count"] == 1
    assert audit["rejected_count"] == 1
    assert any("oov" in reason for reason in audit["rejections_by_reason"])


def test_corpus_builder_rejects_showdown_invalid() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())

    def failing_validator(
        variants: Sequence[TeamRecord], **kwargs: Any
    ) -> tuple[AdmissionResult, ...]:
        results = []
        for index, variant in enumerate(variants):
            if index == 1:
                results.append(
                    AdmissionResult(
                        variant.team.team_hash,
                        valid=False,
                        packed_team=None,
                        problems=("Illegal ability",),
                    )
                )
            else:
                packed = "]".join(
                    f"{m.species}|{m.species}|{m.item}|{m.ability}|{','.join(m.moves)}|{m.nature}"
                    for m in variant.team.members
                )
                results.append(
                    AdmissionResult(
                        variant.team.team_hash, valid=True, packed_team=packed, problems=()
                    )
                )
        return tuple(results)

    v1 = _variant_team_corpus_build("Pikachu", source_series=("s1",))
    v2 = _variant_team_corpus_build("Charizard", source_series=("s2",))
    manifest, audit = build_corpus(
        (v1, v2), tokenizer=tokenizer, validator=failing_validator, global_contract_sha256="a" * 64
    )
    assert len(manifest.entries) == 1
    assert audit["rejected_count"] == 1
    assert any("showdown_invalid" in reason for reason in audit["rejections_by_reason"])


def test_split_assignment_prevents_series_leakage() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())
    v1 = _variant_team_corpus_build("Pikachu", source_series=("shared-series",))
    v2 = _variant_team_corpus_build("Charizard", source_series=("shared-series",))
    v3 = _variant_team_corpus_build(
        "Whimsicott", source_series=("other-series",), archetypes=("held_out",)
    )
    manifest, _ = build_corpus(
        (v1, v2, v3),
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256="a" * 64,
        ratio_train=0.5,
        ratio_val=0.5,
        ratio_test=0.0,
        held_out_tags=("held_out",),
    )
    assert len(manifest.entries) == 3
    by_species = {entry.canonical_hash: entry.split for entry in manifest.entries}
    assert by_species[v1.team.team_hash] == by_species[v2.team.team_hash]
    assert by_species[v3.team.team_hash] == CorpusSplit.HELD_OUT_ARCHETYPE


def test_audit_corpus_and_coverage() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())
    v1 = _variant_team_corpus_build("Pikachu", usage_count=10, archetypes=("hyperoffense",))
    v2 = _variant_team_corpus_build(
        "Charizard", source_series=("s2",), usage_count=5, archetypes=("balance",)
    )
    manifest, audit = build_corpus(
        (v1, v2), tokenizer=tokenizer, validator=_mock_validator, global_contract_sha256="b" * 64
    )
    re_audit = audit_corpus(manifest)
    assert re_audit["admitted_count"] == 2
    assert "pikachu" in re_audit["species_coverage"]
    assert "hyperoffense" in re_audit["archetype_counts"]
    assert re_audit["archetype_counts"]["hyperoffense"] == 1


def test_populate_pool_directories(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab_team_corpus_build())
    # Each variant needs a unique species or move so canonical_hash is distinct
    unique_variants = tuple(
        _variant_team_corpus_build(
            "Pikachu" if i % 2 == 0 else "Charizard",
            source_series=(f"series-{i}",),
            usage_count=i * 10,
            move="Fake Out" if i < 3 else "Protect",
        )
        for i in range(1, 5)
    )
    manifest, _ = build_corpus(
        unique_variants,
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256="c" * 64,
    )
    populate_pool_directories(manifest, output_root=tmp_path, reduced_limit=2)

    all_path = tmp_path / "all" / "corpus_manifest.json"
    reduced_path = tmp_path / "reduced" / "corpus_manifest.json"
    assert all_path.is_file()
    assert reduced_path.is_file()

    manifest_all = TeamCorpusManifest.from_dict(json.loads(all_path.read_text()))
    manifest_reduced = TeamCorpusManifest.from_dict(json.loads(reduced_path.read_text()))

    assert len(manifest_all.entries) == len(manifest.entries)
    assert len(manifest_reduced.entries) <= 2
    assert all(entry in manifest_all.entries for entry in manifest_reduced.entries)
    # Ensure reduced manifest entries are sorted top usage
    assert sorted((e.usage_count for e in manifest_reduced.entries), reverse=True) == [
        e.usage_count for e in manifest_reduced.entries
    ]


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
    entries = tuple(_make_entry(i) for i in range(5))
    path, manifest = _write_manifest(tmp_path, entries)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
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
    entries = tuple(_make_entry(i) for i in range(3))
    path, manifest = _write_manifest(tmp_path, entries)

    # Wrong corpus_hash raises ValueError
    bad_spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash="0" * 64,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
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


def test_uniform_archetype_rejects_untagged_pool(tmp_path: Path) -> None:
    """An untagged pool must fail loudly, not collapse into uniform-over-entries."""
    entries = tuple(_make_entry(i, tags=()) for i in range(3))
    path, manifest = _write_manifest(tmp_path, entries)
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        sampling_policy=SamplingPolicy.UNIFORM_ARCHETYPE,
    )
    with pytest.raises(ValueError, match="requires archetype tags"):
        CorpusTeamSource(spec)


def test_held_out_tags_reject_untagged_corpus() -> None:
    """Requesting a hold-out with no tags anywhere would silently populate no split."""
    variants = (_variant_team_corpus(metadata=replace(_metadata(), archetype_tags=())),)
    with pytest.raises(ValueError, match="no team record carries an archetype tag"):
        _component_splits(
            variants,
            ratio_train=0.8,
            ratio_val=0.1,
            ratio_test=0.1,
            held_out_tags=("trick-room",),
        )

    # The same corpus must still split cleanly when no hold-out is requested.
    splits = _component_splits(
        variants, ratio_train=0.8, ratio_val=0.1, ratio_test=0.1, held_out_tags=()
    )
    assert splits[0] is not CorpusSplit.HELD_OUT_ARCHETYPE


def test_sampling_policy_rare_coverage(tmp_path: Path) -> None:
    e_common = _make_entry(1, usage_count=10000)
    e_rare = _make_entry(2, usage_count=1)
    path, manifest = _write_manifest(tmp_path, (e_common, e_rare))
    spec = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
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
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
        curriculum_stage="trickroom",
    )
    source = CorpusTeamSource(spec)
    rng = random.Random(500)
    for _ in range(20):
        t = source.sample(rng)
        assert t.team_hash == e3.packed_sha256


def test_sampling_policy_indexes_and_lazy_caching(tmp_path: Path) -> None:
    # Under USAGE_WEIGHTED policy, canonical and archetype indexes should remain uninitialized (None) until explicitly requested
    e1 = _make_entry(1, canonical_index=1, usage_count=100)
    e2 = _make_entry(2, canonical_index=2, usage_count=100)
    path, manifest = _write_manifest(tmp_path, (e1, e2))
    spec_usage = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        sampling_policy=SamplingPolicy.USAGE_WEIGHTED,
    )
    source_usage = CorpusTeamSource(spec_usage)
    assert source_usage._by_canonical is None
    assert source_usage._by_archetype is None
    # Sampling should succeed without building unneeded indexes
    rng = random.Random(700)
    assert source_usage.sample(rng) is not None

    spec_canonical = CorpusSourceSpec(
        corpus_path=str(path),
        corpus_hash=manifest.corpus_hash,
        format_id=FORMAT.battle_format,
        split=CorpusSplit.TRAIN,
        sampling_policy=SamplingPolicy.UNIFORM_CANONICAL,
    )
    source_canonical = CorpusTeamSource(spec_canonical)
    assert source_canonical._by_canonical is not None
    assert source_canonical._canonical_keys is not None
    assert len(source_canonical._canonical_keys) == 2


TEAM = """
Pikachu @ Light Ball
Ability: Static
Jolly Nature
- Fake Out

Charizard @ Charizardite Y
Ability: Blaze
Modest Nature
- Heat Wave

Whimsicott @ Focus Sash
Ability: Prankster
Timid Nature
- Tailwind

Garchomp @ Sitrus Berry
Ability: Rough Skin
Jolly Nature
- Earthquake

Kingambit @ Black Glasses
Ability: Defiant
Adamant Nature
- Sucker Punch

Glimmora @ Shuca Berry
Ability: Toxic Debris
Modest Nature
- Power Gem
"""


def _variant_team_sources():
    members = tuple(
        TeamMember(species, "item", "ability", ("Protect",), "Serious")
        for species in ("Pikachu", "Charizard", "Whimsicott", "Garchomp", "Kingambit", "Glimmora")
    )
    return TeamRecord(
        CanonicalTeam(members),
        tuple(StatPoints() for _ in members),
        TeamMetadata((), (), "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    )


def test_file_source_prepares_stable_pool_and_uses_caller_rng(tmp_path):
    (tmp_path / ".ignored").write_text("bad", encoding="utf-8")
    (tmp_path / "b.txt").write_text(TEAM.replace("Light Ball", "Sitrus Berry"), encoding="utf-8")
    (tmp_path / "a.txt").write_text(TEAM, encoding="utf-8")
    source = FileTeamSource(tmp_path)
    first_rng, second_rng = random.Random(9), random.Random(9)
    first = [source.sample(first_rng).team_hash for _ in range(4)]
    second = [source.sample(second_rng).team_hash for _ in range(4)]
    assert first == second
    hashes = source.describe()["team_hashes"]
    assert isinstance(hashes, tuple)
    assert len(hashes) == 2


def test_fixed_source_reuses_prepared_team():
    source = FixedTeamSource(TEAM)
    assert source.sample(random.Random(1)) is source.sample(random.Random(2))


def test_sources_reject_empty_and_malformed_pools(tmp_path):
    with pytest.raises(FileNotFoundError, match="No team files"):
        FileTeamSource(tmp_path)
    (tmp_path / "bad.txt").write_text("not a team", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed team file"):
        FileTeamSource(tmp_path)


def test_validate_many_preserves_order_and_parses_diagnostics():
    calls = []

    def runner(*args, **kwargs):
        calls.append(kwargs["input"])
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout='[{"valid": true, "packedTeam": "packed", "problems": []}, {"valid": true, "packedTeam": "packed", "problems": []}]',
            stderr="",
        )

    variants = (_variant_team_sources(), _variant_team_sources())
    results = validate_many(variants, runner=runner)
    assert [result.team_hash for result in results] == [item.team.team_hash for item in variants]
    assert len(calls) == 1


def test_validated_team_rejects_untrusted_packed_values():
    with pytest.raises(ValueError, match="SHA-256"):
        ValidatedTeam("packed", "short")


def _variant_team_validation_batch(
    species: str = "Pikachu", item: str = "Light Ball"
) -> TeamRecord:
    members = (
        TeamMember(
            species=species,
            item=item,
            ability="Static",
            moves=("Fake Out", "Protect", "Thunderbolt", "Electroweb"),
            nature="Jolly",
        ),
        TeamMember(
            species="Charizard",
            item="Charizardite Y",
            ability="Blaze",
            moves=("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
            nature="Modest",
        ),
        TeamMember(
            species="Whimsicott",
            item="Focus Sash",
            ability="Prankster",
            moves=("Moonblast", "Tailwind", "Encore", "Protect"),
            nature="Timid",
        ),
        TeamMember(
            species="Garchomp",
            item="Sitrus Berry",
            ability="Rough Skin",
            moves=("Earthquake", "Dragon Claw", "Rock Slide", "Protect"),
            nature="Jolly",
        ),
        TeamMember(
            species="Kingambit",
            item="Black Glasses",
            ability="Defiant",
            moves=("Kowtow Cleave", "Sucker Punch", "Protect", "Low Kick"),
            nature="Adamant",
        ),
        TeamMember(
            species="Glimmora",
            item="Shuca Berry",
            ability="Toxic Debris",
            moves=("Power Gem", "Sludge Bomb", "Earth Power", "Protect"),
            nature="Modest",
        ),
    )
    return TeamRecord(
        team=CanonicalTeam(members),
        spreads=tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=TeamMetadata(
            source_series=("series-1",),
            source_replays=("series-1-game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=1,
            archetype_tags=("balance",),
        ),
    )


def test_validate_many_empty_returns_empty_tuple() -> None:
    calls: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs.get("input"))
        return subprocess.CompletedProcess(args[0], 0, stdout="[]", stderr="")

    assert validate_many((), runner=runner) == ()
    assert validate_many_batched((), runner=runner) == ()
    assert len(calls) == 0


def test_validate_many_batched_splits_chunks_and_preserves_order() -> None:
    calls: list[str] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = str(kwargs.get("input", ""))
        calls.append(payload)
        items = json.loads(payload)
        response = [
            {"valid": True, "packedTeam": f"packed_{index}", "problems": []}
            for index, _ in enumerate(items)
        ]
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(response), stderr="")

    variants = (
        _variant_team_validation_batch("Pikachu"),
        _variant_team_validation_batch("Raichu"),
        _variant_team_validation_batch("Zapdos"),
    )
    results = validate_many_batched(variants, batch_size=2, runner=runner)
    assert len(results) == 3
    assert [result.team_hash for result in results] == [
        variant.team.team_hash for variant in variants
    ]
    assert results[0].packed_team == "packed_0"
    assert len(calls) == 2
    assert len(json.loads(calls[0])) == 2
    assert len(json.loads(calls[1])) == 1


def test_validate_many_delegates_to_batched_runner() -> None:
    calls: list[str] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = str(kwargs.get("input", ""))
        calls.append(payload)
        items = json.loads(payload)
        response = [{"valid": True, "packedTeam": "packed_team", "problems": []} for _ in items]
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(response), stderr="")

    variants = (_variant_team_validation_batch("Pikachu"), _variant_team_validation_batch("Raichu"))
    results = validate_many(variants, runner=runner)
    assert len(results) == 2
    assert len(calls) == 1
    assert len(json.loads(calls[0])) == 2


def test_validate_many_batched_error_handling():
    def _test_validate_many_batched_handles_process_failure():
        def failing_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="Node error")

        with pytest.raises(RuntimeError, match="failed: Node error"):
            validate_many_batched((_variant_team_validation_batch(),), runner=failing_runner)

    _test_validate_many_batched_handles_process_failure()

    def _test_validate_many_batched_handles_timeout():
        def timeout_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(args[0], 30.0)

        with pytest.raises(RuntimeError, match="timed out"):
            validate_many_batched((_variant_team_validation_batch(),), runner=timeout_runner)

    _test_validate_many_batched_handles_timeout()

    def _test_validate_many_batched_handles_malformed_json():
        def malformed_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args[0], 0, stdout="not json", stderr="")

        with pytest.raises(RuntimeError, match="malformed response"):
            validate_many_batched((_variant_team_validation_batch(),), runner=malformed_runner)

    _test_validate_many_batched_handles_malformed_json()


def test_persistent_showdown_validator_lifecycle_and_validation() -> None:
    calls: list[str] = []
    closed = [False]

    class MockProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.stdin = self
            self.stdout = self
            self.stderr = self
            self.returncode = None

        def write(self, data: str) -> None:
            trimmed = data.strip()
            calls.append(trimmed)
            try:
                payload = json.loads(trimmed)
                if payload.get("command") == "stop":
                    closed[0] = True
            except Exception:
                pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> str:
            if not calls:
                return ""
            payload = json.loads(calls[-1])
            if payload.get("command") == "stop":
                return ""
            items = payload.get("batch", [])
            response = [
                {"valid": True, "packedTeam": f"persistent_{index}", "problems": []}
                for index, _ in enumerate(items)
            ]
            return json.dumps({"status": "ok", "results": response}) + "\n"

        def terminate(self) -> None:
            closed[0] = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def mock_popen(*args: Any, **kwargs: Any) -> Any:
        return MockProcess()

    variants = (_variant_team_validation_batch("Pikachu"), _variant_team_validation_batch("Raichu"))
    with PersistentShowdownValidator(popen_factory=mock_popen) as validator:
        results = validator.validate_many(variants)
        assert len(results) == 2
        assert results[0].packed_team == "persistent_0"
        assert results[1].packed_team == "persistent_1"

    assert closed[0] is True
    assert len(calls) >= 1
    first_request = json.loads(calls[0])
    assert len(first_request["batch"]) == 2


def test_single_team_validation_reports_pinned_runner_failures() -> None:
    variant = _variant_team_validation_batch()

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess("node", 1, stdout="", stderr="invalid team")

    with pytest.raises(RuntimeError, match="validator failed: invalid team"):
        validate_variant(variant, runner=runner)


def test_batched_team_validation_rejects_timeout_crash_and_malformed_output() -> None:
    variant = _variant_team_validation_batch()

    def timeout_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise subprocess.TimeoutExpired("node", 0.1)

    with pytest.raises(RuntimeError, match="timed out"):
        validate_many_batched((variant,), runner=timeout_runner)

    def malformed_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess("node", 0, stdout="{}", stderr="")

    with pytest.raises(RuntimeError, match="malformed"):
        validate_many_batched((variant,), runner=malformed_runner)
