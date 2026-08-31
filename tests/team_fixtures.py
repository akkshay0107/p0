"""Plain team value fixtures shared by unit and integration tests."""

from __future__ import annotations

from collections.abc import Sequence

from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord


def metadata(source: str = "series-1", usage: int = 1) -> TeamMetadata:
    """Build stable provenance metadata for a team fixture."""
    return TeamMetadata(
        source_series=(source,),
        source_replays=(f"{source}-game-1",),
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-02T00:00:00Z",
        usage_count=usage,
    )


def team_variant(
    species: str = "Pikachu",
    item: str = "Light Ball",
    source_series: tuple[str, ...] = ("series-1",),
    usage_count: int = 1,
    move: str = "Fake Out",
    members: Sequence[TeamMember] | None = None,
    spreads: tuple[StatPoints, ...] | None = None,
    metadata: TeamMetadata | None = None,
) -> TeamRecord:
    """Build a valid six-member team record for corpus and validation tests."""
    if members is None:
        members = (
            TeamMember(
                species=species,
                item=item,
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
        team=CanonicalTeam(tuple(members)),
        spreads=spreads or tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=metadata
        or TeamMetadata(
            source_series=source_series,
            source_replays=(f"{source_series[0] if source_series else 'series-1'}-game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=usage_count,
        ),
    )
