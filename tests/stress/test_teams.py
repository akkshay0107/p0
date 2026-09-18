from __future__ import annotations

import json

import pytest

from p0.format_config import FORMAT
from p0.teams.validation import (
    showdown_payload,
    validate_many_batched,
)
from tests.stress._helpers import stress_count, stress_random_team_record, stress_rng


class TestTeams:
    @pytest.mark.stress
    def test_species_aware_random_teams_are_valid_for_bo3(self, showdown_assets) -> None:
        """Check generated teams directly against the pinned Bo3 Showdown validator."""
        del showdown_assets
        count = stress_count("P0_STRESS_LEGAL_TEAM_VARIANTS", 32)
        rng = stress_rng()
        variants = tuple(
            stress_random_team_record(rng, label=f"legal-stress-{index}") for index in range(count)
        )

        results = validate_many_batched(
            variants,
            batch_size=min(32, count),
            format_id=FORMAT.bo3_format,
        )
        failures = [
            (index, result.problems) for index, result in enumerate(results) if not result.valid
        ]
        assert not failures, f"Generated teams were rejected by the Bo3 validator: {failures[:3]}"

    @pytest.mark.stress
    def test_batched_team_validation_preserves_identity_and_payload_contract(
        self, showdown_assets
    ) -> None:
        """Stress real batched Showdown validation across many generated team variants."""
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
