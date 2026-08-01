import pytest

from p0.teams.validation import (
    validate_variant,
)
from tests.unit.test_teams import _variant_team_corpus as _variant


@pytest.mark.integration
def test_pinned_showdown_admits_and_packs_legal_variant():
    result = validate_variant(_variant())
    assert result.valid, result.problems
    assert result.packed_team
