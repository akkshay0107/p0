import pytest

from p0.teams.validation import validate_variant
from tests.unit.test_teams import _mock_team_variant


@pytest.mark.integration
def test_pinned_showdown_admits_and_packs_legal_variant() -> None:
    variant = _mock_team_variant()
    result = validate_variant(variant)
    assert result.valid, result.problems
    assert result.team_hash == variant.team.team_hash
    assert result.packed_team
