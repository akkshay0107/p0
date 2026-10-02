import json

import pytest

from p0.format_config import FORMAT
from p0.teams.validation import (
    showdown_payload,
    validate_many_batched,
    validate_variant,
)
from tests.stress._helpers import stress_count, stress_random_team_record, stress_rng
from tests.team_fixtures import team_variant


class TestTeams:
    @pytest.mark.integration
    @pytest.mark.heavy
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


@pytest.mark.integration
class TestTeamScale:
    @pytest.mark.heavy
    def test_batched_team_validation_preserves_identity_and_payload_contract(
        self, showdown_assets
    ) -> None:
        """Check real batched Showdown validation across many seeded team variants."""
        del showdown_assets
        count = stress_count("P0_STRESS_TEAM_VARIANTS", 1024)
        rng = stress_rng()
        variants = tuple(
            stress_random_team_record(rng, label=f"batched-stress-{index}")
            for index in range(count)
        )
        batch_size = stress_count("P0_STRESS_TEAM_BATCH_SIZE", 64)
        results = validate_many_batched(
            variants,
            batch_size=batch_size,
            format_id=FORMAT.bo3_format,
        )

        assert [result.team_hash for result in results] == [
            variant.team.team_hash for variant in variants
        ]
        assert all(result.valid for result in results)
        assert all(result.packed_team for result in results)

        payload = json.loads(showdown_payload(variants[0], format_id=FORMAT.bo3_format))
        assert payload["format"] == FORMAT.bo3_format
        assert len(payload["team"]) == 6
        assert {"species", "moves", "evs", "ivs"} <= payload["team"][0].keys()
