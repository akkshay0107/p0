import json

import pytest
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

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
    @pytest.mark.parametrize(("move", "valid"), (("Fake Out", True), ("Spore", False)))
    def test_pinned_showdown_admits_legal_moves_and_rejects_unlearnable_moves(
        self, move: str, valid: bool
    ) -> None:
        variant = team_variant(move=move)
        result = validate_variant(variant)
        assert result.valid is valid, result.problems
        assert result.team_key == variant.team.team_key
        if valid:
            assert result.packed_team
            assert result.problems == ()
        else:
            assert result.packed_team is None
            assert any("Pikachu can't learn Spore" in problem for problem in result.problems)


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
        variants = tuple(stress_random_team_record(rng) for _ in range(count))
        batch_size = stress_count("P0_STRESS_TEAM_BATCH_SIZE", 64)
        results = validate_many_batched(
            variants,
            batch_size=batch_size,
            format_id=FORMAT.bo3_format,
        )

        assert [result.team_key for result in results] == [
            variant.team.team_key for variant in variants
        ]
        assert all(result.valid for result in results)
        assert all(result.packed_team for result in results)
        for variant, result in zip(variants, results, strict=True):
            assert result.packed_team is not None
            packed_members = Teambuilder.parse_packed_team(result.packed_team)
            expected_members = {
                to_id_str(member.species): (member, spread)
                for member, spread in zip(variant.team.members, variant.spreads, strict=True)
            }
            assert len(packed_members) == 6
            for packed in packed_members:
                species = packed.species or packed.nickname
                assert species is not None
                member, spread = expected_members.pop(to_id_str(species))
                assert to_id_str(packed.item or "") == to_id_str(member.item)
                assert to_id_str(packed.ability or "") == to_id_str(member.ability)
                assert sorted(packed.moves) == sorted(to_id_str(move) for move in member.moves)
                assert to_id_str(packed.nature or "") == to_id_str(member.nature)
                assert packed.evs == list(spread.as_tuple())
            assert not expected_members

        payload = json.loads(showdown_payload(variants[0], format_id=FORMAT.bo3_format))
        assert payload["format"] == FORMAT.bo3_format
        assert len(payload["team"]) == 6
        assert {"species", "moves", "evs", "ivs"} <= payload["team"][0].keys()
