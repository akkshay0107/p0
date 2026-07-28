"""Tests for team corpus construction pipeline and content audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from p0.format_config import FORMAT
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import (
    CorpusSplit,
    TeamCorpusManifest,
)
from p0.teams.corpus_build import (
    CorpusBuilder,
    SplitPolicy,
    audit_corpus,
    populate_pool_directories,
)
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamVariant
from p0.teams.validation import AdmissionResult


def _mock_vocab() -> dict[str, dict[str, int]]:
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


def _variant(
    species: str = "Pikachu",
    source_series: tuple[str, ...] = ("series-1",),
    usage_count: int = 1,
    archetypes: tuple[str, ...] = ("balance",),
    move: str = "Fake Out",
) -> TeamVariant:
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
    return TeamVariant(
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


def _mock_validator(variants: Sequence[TeamVariant], **kwargs: Any) -> tuple[AdmissionResult, ...]:
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
    tokenizer = PokemonTokenizer(_mock_vocab())
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256="a" * 64,
        format_id=FORMAT.battle_format,
    )
    v1 = _variant("Pikachu", usage_count=5)
    v2 = _variant("Charizard", source_series=("series-2",), usage_count=3)
    manifest, audit = builder.build((v1, v2))
    assert len(manifest.entries) == 2
    assert manifest.format_id == FORMAT.battle_format
    assert manifest.runtime_contract_sha256 == "a" * 64
    assert audit.admitted_count == 2
    assert audit.rejected_count == 0
    assert set(audit.species_coverage) >= {"pikachu", "charizard"}


def test_corpus_builder_rejects_oov_species() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256="a" * 64,
    )
    v_valid = _variant("Pikachu")
    v_oov = _variant("Missingno")
    manifest, audit = builder.build((v_valid, v_oov))
    assert len(manifest.entries) == 1
    assert audit.admitted_count == 1
    assert audit.rejected_count == 1
    assert any("oov" in reason for reason in audit.rejections_by_reason)


def test_corpus_builder_rejects_showdown_invalid() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())

    def failing_validator(
        variants: Sequence[TeamVariant], **kwargs: Any
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

    builder = CorpusBuilder(
        tokenizer=tokenizer, validator=failing_validator, runtime_contract_sha256="a" * 64
    )
    v1 = _variant("Pikachu", source_series=("s1",))
    v2 = _variant("Charizard", source_series=("s2",))
    manifest, audit = builder.build((v1, v2))
    assert len(manifest.entries) == 1
    assert audit.rejected_count == 1
    assert any("showdown_invalid" in reason for reason in audit.rejections_by_reason)


def test_split_assignment_prevents_series_leakage() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256="a" * 64,
        split_policy=SplitPolicy(
            ratio_train=0.5, ratio_val=0.5, ratio_test=0.0, held_out_tags=("held_out",)
        ),
    )
    v1 = _variant("Pikachu", source_series=("shared-series",))
    v2 = _variant("Charizard", source_series=("shared-series",))
    v3 = _variant("Whimsicott", source_series=("other-series",), archetypes=("held_out",))
    manifest, _ = builder.build((v1, v2, v3))
    assert len(manifest.entries) == 3
    by_species = {entry.canonical_hash: entry.split for entry in manifest.entries}
    assert by_species[v1.team.team_hash] == by_species[v2.team.team_hash]
    assert by_species[v3.team.team_hash] == CorpusSplit.HELD_OUT_ARCHETYPE


def test_audit_corpus_and_coverage() -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256="b" * 64,
    )
    v1 = _variant("Pikachu", usage_count=10, archetypes=("hyperoffense",))
    v2 = _variant("Charizard", source_series=("s2",), usage_count=5, archetypes=("balance",))
    manifest, audit = builder.build((v1, v2))
    re_audit = audit_corpus(manifest)
    assert re_audit.admitted_count == 2
    assert "pikachu" in re_audit.species_coverage
    assert "hyperoffense" in re_audit.archetype_counts
    assert re_audit.archetype_counts["hyperoffense"] == 1


def test_populate_pool_directories(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256="c" * 64,
    )
    # Each variant needs a unique species or move so canonical_hash is distinct
    unique_variants = tuple(
        _variant(
            "Pikachu" if i % 2 == 0 else "Charizard",
            source_series=(f"series-{i}",),
            usage_count=i * 10,
            move="Fake Out" if i < 3 else "Protect",
        )
        for i in range(1, 5)
    )
    manifest, _ = builder.build(unique_variants)
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
