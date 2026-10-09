"""Unit tests for canonical Champions team records and deduplication."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from p0.teams.stat_points import StatPoints
from p0.teams.team import (
    CanonicalTeam,
    TeamMember,
    TeamRecord,
    deduplicate_variants,
    normalize_id,
)
from tests.team_fixtures import team_variant


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
            gender="F",
            level=73,
        )
        d = member.to_dict()
        assert d == {
            "species": "pikachu",
            "item": "lightball",
            "ability": "static",
            "moves": ["protect", "thunderbolt"],
            "nature": "jolly",
            "gender": "F",
            "level": 73,
        }
        restored = TeamMember.from_dict(d)
        assert restored == member.canonical()


class TestCanonicalTeam:
    def test_must_have_exactly_six_members(self) -> None:
        m = TeamMember(species="Pikachu", item="", ability="", moves=("protect",), nature="jolly")
        with pytest.raises(ValueError, match="exactly six members"):
            CanonicalTeam((m,) * 5)
        with pytest.raises(ValueError, match="exactly six members"):
            CanonicalTeam((m,) * 7)

    def test_team_key_ignores_display_and_member_order(self) -> None:
        first = team_variant()
        reversed_members = tuple(reversed(first.team.members))
        second = team_variant(members=reversed_members)
        assert first.team.team_key == second.team.team_key

        pikachu = first.team.members[0]
        display_variant = replace(
            pikachu,
            species="PIKACHU",
            item="Light-Ball",
            ability="STATIC",
            moves=tuple(move.upper() for move in reversed(pikachu.moves)),
            nature="JOLLY",
        )
        display_team = CanonicalTeam((display_variant, *first.team.members[1:]))
        assert display_team.team_key == first.team.team_key

        semantic_variant = replace(display_variant, ability="Lightning Rod")
        semantic_team = CanonicalTeam((semantic_variant, *first.team.members[1:]))
        assert semantic_team.team_key != first.team.team_key

    def test_team_dict_roundtrip(self) -> None:
        variant = team_variant()
        pikachu = replace(variant.team.members[0], gender="M", level=73)
        team = CanonicalTeam((pikachu, *variant.team.members[1:]))
        d = team.to_dict()
        assert cast(list[object], d["members"])[4] == {
            "species": "pikachu",
            "item": "lightball",
            "ability": "static",
            "moves": ["electroweb", "fakeout", "protect", "thunderbolt"],
            "nature": "jolly",
            "gender": "M",
            "level": 73,
        }
        restored = CanonicalTeam.from_dict(d)
        assert restored == team.canonical()


class TestTeamRecord:
    def test_spread_count_must_match_members(self) -> None:
        variant = team_variant()
        with pytest.raises(ValueError, match="Each team member requires one Stat Point spread"):
            TeamRecord(
                team=variant.team,
                spreads=variant.spreads[:5],
            )


class TestDeduplicateVariants:
    def test_merges_reordered_duplicates_and_preserves_spread_variants(self) -> None:
        first = team_variant()
        first = replace(
            first,
            spreads=(
                StatPoints(hp=1),
                StatPoints(atk=2),
                StatPoints(defense=3),
                StatPoints(spa=4),
                StatPoints(spd=5),
                StatPoints(spe=6),
            ),
        )
        reordered = replace(
            first,
            team=CanonicalTeam(tuple(reversed(first.team.members))),
            spreads=tuple(reversed(first.spreads)),
        )
        alternate = replace(
            first,
            spreads=tuple(StatPoints(hp=32, defense=17, spd=17) for _ in first.spreads),
        )
        result = deduplicate_variants((reordered, alternate, first))
        assert len(result) == 2
        merged = next(item for item in result if len(set(item.spreads)) == 6)
        associations = tuple(
            (member.canonical().species, spread.as_tuple())
            for member, spread in sorted(
                zip(merged.team.members, merged.spreads, strict=True),
                key=lambda pair: pair[0].canonical().species,
            )
        )
        assert associations == (
            ("charizard", (0, 2, 0, 0, 0, 0)),
            ("garchomp", (0, 0, 0, 4, 0, 0)),
            ("glimmora", (0, 0, 0, 0, 0, 6)),
            ("kingambit", (0, 0, 0, 0, 5, 0)),
            ("pikachu", (1, 0, 0, 0, 0, 0)),
            ("whimsicott", (0, 0, 3, 0, 0, 0)),
        )
