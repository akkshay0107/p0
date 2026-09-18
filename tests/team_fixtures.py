"""Plain team value fixtures shared by unit and integration tests."""

from __future__ import annotations

from collections.abc import Sequence

from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord


def metadata(usage: int = 1) -> TeamMetadata:
    """Build stable metadata for a team fixture."""
    return TeamMetadata(usage_count=usage)


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
        metadata=metadata or TeamMetadata(usage_count=usage_count),
    )


def sample_vocabulary() -> dict[str, dict[str, int]]:
    """Standard test vocabulary covering standard fixture species, items, abilities, and moves."""
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


# Standard Pikachu/Charizard test team used across integration and unit tests
DEFAULT_TEST_TEAM = """
Pikachu @ Light Ball
Ability: Static
Level: 50
Jolly Nature
- Fake Out
- Protect
- Thunderbolt
- Electroweb

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Corrosion
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""
