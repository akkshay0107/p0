from dataclasses import replace

import pytest

from src.team_data.stat_points import StatPoints
from src.team_data.team_corpus import (
    CanonicalTeam,
    TeamMember,
    TeamMetadata,
    TeamVariant,
    deduplicate_variants,
    validate_evidence_cutoff,
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


def _variant(members=None, metadata=None, spreads=None):
    members = members or _members()
    return TeamVariant(
        CanonicalTeam(tuple(members)),
        spreads or tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata or _metadata(),
    )


def test_team_hash_ignores_display_and_member_order():
    first = _variant()
    reversed_members = tuple(reversed(_members()))
    second = _variant(reversed_members)
    assert first.team.team_hash == second.team.team_hash


def test_deduplication_merges_metadata_but_preserves_spread_variants():
    first = _variant()
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


@pytest.mark.integration
def test_pinned_showdown_admits_and_packs_legal_variant():
    result = validate_variant(_variant())
    assert result.valid, result.problems
    assert result.packed_team
