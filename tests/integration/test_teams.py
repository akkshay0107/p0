import pytest

from p0.teams.validation import validate_variant
from tests.team_fixtures import team_variant


class TestTeams:
    @pytest.mark.integration
    def test_pinned_showdown_admits_and_packs_legal_variant(self) -> None:
        """
        Verify that Showdown's native validator accepts a legal team variant and generates packed output.

        Ensures that our fixture legal team complies with Showdown's format rules (moves, abilities,
        item clauses, EV limits) and that team validation produces a valid packed wire representation
        with matching hash identity.
        """
        variant = team_variant()
        result = validate_variant(variant)
        assert result.valid, result.problems
        assert result.team_hash == variant.team.team_hash
        assert result.packed_team
