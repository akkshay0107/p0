import pytest

from p0.teams.validation import (
    validate_many_batched,
    validate_variant,
)
from tests.unit.test_team_validation_batch import _variant


@pytest.mark.integration
def test_batched_validator_integration_matches_single_validator() -> None:
    variants = (_variant("Pikachu"), _variant("Raichu"))
    single_results = tuple(validate_variant(variant) for variant in variants)
    batched_results = validate_many_batched(variants)
    assert len(single_results) == len(batched_results)
    for single, batched in zip(single_results, batched_results, strict=True):
        assert single.team_hash == batched.team_hash
        assert single.valid == batched.valid
        assert single.packed_team == batched.packed_team
        assert single.problems == batched.problems
