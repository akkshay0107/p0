"""Unit tests for canonical Champions team records and deduplication."""

from __future__ import annotations

from dataclasses import replace

import pytest

from p0.teams.stat_points import StatPoints
from p0.teams.team import (
    CanonicalTeam,
    TeamMember,
    TeamMetadata,
    TeamRecord,
    deduplicate_variants,
    normalize_id,
)
from tests.team_fixtures import metadata, team_variant


class TestNormalizeId:
    def test_strips_punctuation_and_lowercases(self) -> None:
        assert normalize_id("Tapu Koko") == "tapukoko"
        assert normalize_id("Choice Band-") == "choiceband"
        assert normalize_id("Iron_Bundle") == "ironbundle"


class TestTeamMember:
    def test_canonicalization(self) -> None:
        member = TeamMember(
            species="Pikachu",
            item="Light Ball",
            ability="Static",
            moves=("Thunderbolt", "Protect", "Electroweb", "Fake Out"),
            nature="Jolly",
            gender="m",
            level=50,
        )
        canon = member.canonical()
        assert canon.species == "pikachu"
        assert canon.item == "lightball"
        assert canon.ability == "static"
        assert canon.nature == "jolly"
        assert canon.gender == "M"
        assert canon.moves == ("electroweb", "fakeout", "protect", "thunderbolt")

    def test_missing_species_or_nature_raises(self) -> None:
        with pytest.raises(ValueError, match="require species and nature"):
            TeamMember(species="", item="", ability="", moves=("protect",), nature="jolly")
        with pytest.raises(ValueError, match="require species and nature"):
            TeamMember(species="pikachu", item="", ability="", moves=("protect",), nature="")

    def test_invalid_move_count_raises(self) -> None:
        with pytest.raises(ValueError, match="one to four moves"):
            TeamMember(species="pikachu", item="", ability="", moves=(), nature="jolly")
        with pytest.raises(ValueError, match="one to four moves"):
            TeamMember(
                species="pikachu",
                item="",
                ability="",
                moves=("m1", "m2", "m3", "m4", "m5"),
                nature="jolly",
            )

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level must be in \\[1, 100\\]"):
            TeamMember(
                species="pikachu", item="", ability="", moves=("protect",), nature="jolly", level=0
            )
        with pytest.raises(ValueError, match="level must be in \\[1, 100\\]"):
            TeamMember(
                species="pikachu",
                item="",
                ability="",
                moves=("protect",),
                nature="jolly",
                level=101,
            )

    def test_member_dict_roundtrip(self) -> None:
        member = TeamMember(
            species="Pikachu",
            item="Light Ball",
            ability="Static",
            moves=("Protect", "Thunderbolt"),
            nature="Jolly",
        )
        d = member.to_dict()
        restored = TeamMember.from_dict(d)
        assert restored == member.canonical()


class TestCanonicalTeam:
    def test_must_have_exactly_six_members(self) -> None:
        m = TeamMember(species="Pikachu", item="", ability="", moves=("protect",), nature="jolly")
        with pytest.raises(ValueError, match="exactly six members"):
            CanonicalTeam((m,) * 5)
        with pytest.raises(ValueError, match="exactly six members"):
            CanonicalTeam((m,) * 7)

    def test_team_hash_ignores_display_and_member_order(self) -> None:
        first = team_variant()
        reversed_members = tuple(reversed(first.team.members))
        second = team_variant(members=reversed_members)
        assert first.team.team_hash == second.team.team_hash

    def test_team_dict_roundtrip(self) -> None:
        team = team_variant().team
        d = team.to_dict()
        restored = CanonicalTeam.from_dict(d)
        assert restored == team.canonical()


class TestTeamMetadata:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="Usage count must be positive"):
            TeamMetadata(
                source_series=(),
                source_replays=(),
                first_seen="2026-01-01T00:00:00Z",
                last_seen="2026-01-01T00:00:00Z",
                usage_count=0,
            )

    def test_dict_roundtrip(self) -> None:
        meta = metadata()
        assert TeamMetadata.from_dict(meta.to_dict()) == meta


class TestTeamRecord:
    def test_spread_count_must_match_members(self) -> None:
        variant = team_variant()
        with pytest.raises(ValueError, match="Each team member requires one Stat Point spread"):
            TeamRecord(
                team=variant.team,
                spreads=variant.spreads[:5],
                metadata=variant.metadata,
            )

    def test_serialization_round_trip_is_strict(self) -> None:
        variant = team_variant()
        assert TeamRecord.from_dict(variant.to_dict()) == replace(
            variant, team=variant.team.canonical()
        )
        with pytest.raises(ValueError, match="fields"):
            TeamRecord.from_dict({**variant.to_dict(), "unexpected": True})


class TestDeduplicateVariants:
    def test_merges_metadata_and_preserves_spread_variants(self) -> None:
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
