import random
import subprocess

import pytest

from p0.teams.source import FileTeamSource, FixedTeamSource, ValidatedTeam
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamVariant
from p0.teams.validation import validate_many

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


def _variant():
    members = tuple(
        TeamMember(species, "item", "ability", ("Protect",), "Serious")
        for species in ("Pikachu", "Charizard", "Whimsicott", "Garchomp", "Kingambit", "Glimmora")
    )
    return TeamVariant(
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
            args[0], 0, stdout='{"valid": true, "packedTeam": "packed", "problems": []}', stderr=""
        )

    variants = (_variant(), _variant())
    results = validate_many(variants, runner=runner)
    assert [result.team_hash for result in results] == [item.team.team_hash for item in variants]
    assert len(calls) == 2


def test_validated_team_rejects_untrusted_packed_values():
    with pytest.raises(ValueError, match="SHA-256"):
        ValidatedTeam("packed", "short")
