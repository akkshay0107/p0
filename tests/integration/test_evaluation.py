from __future__ import annotations

import urllib.parse

import pytest

from p0.evaluation.harness import (
    DEFAULT_TEST_TEAM,
    EvaluationHarness,
)
from p0.runtime import poke_env_patches
from p0.teams.source import FixedTeamSource


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluation_harness_completes_matchup(showdown_server) -> None:
    """Verify EvaluationHarness runs a live matchup between baseline random players and tracks stats.
    
    Checks that the matchup completes without unhandled exceptions, computes win rates,
    aggregates results per team archetype, and produces a complete dictionary serialization
    containing confidence intervals and metadata suitable for logging.
    """
    parsed = urllib.parse.urlparse(showdown_server.websocket_url)
    assert parsed.port is not None
    poke_env_patches.install()

    harness = EvaluationHarness(
        episodes_per_matchup=1,
        seed=42,
        port=parsed.port,
    )
    fallback = FixedTeamSource(DEFAULT_TEST_TEAM)

    try:
        result = await harness.run_matchup(
            name_a="RandomA",
            policy_a=None,
            name_b="RandomB",
            policy_b=None,
            team_category="fallback",
            team_source=fallback,
            server_configuration=showdown_server,
        )
    finally:
        poke_env_patches.uninstall_for_tests()

    assert result.policy_a == "RandomA"
    assert result.policy_b == "RandomB"
    assert result.team_category == "fallback"
    assert result.total_games == 1
    assert result.wins_a + result.wins_b == 1
    assert result.ties == 0
    assert result.win_rate_a in (0.0, 1.0)
    assert len(result.per_team_results) == 1
    stats = next(iter(result.per_team_results.values()))
    assert stats["games"] == 1
    assert stats["wins"] == result.wins_a
    assert stats["win_rate"] == pytest.approx(stats["wins"] / stats["games"])

    # Verify dictionary serialization schema matches downstream reporting contracts
    dct = result.to_dict()
    assert set(dct) == {
        "policy_a",
        "policy_b",
        "team_category",
        "total_games",
        "wins_a",
        "wins_b",
        "ties",
        "win_rate_a",
        "confidence_interval_a",
        "per_team_results",
        "source_description",
        "per_team_a_results",
        "per_team_b_results",
    }
    assert dct["confidence_interval_a"] == list(result.confidence_interval_a)
    assert dct["total_games"] == 1
    assert len(dct["per_team_results"]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("policy_side", ("a", "b"))
async def test_live_evaluation_runs_model_policy_against_random_opponent(
    showdown_server,
    model_policy,
    policy_side: str,
) -> None:
    """Verify live evaluation works symmetrically when the neural policy is player A or player B."""
    parsed = urllib.parse.urlparse(showdown_server.websocket_url)
    assert parsed.port is not None
    # Assign the model policy to the parametrized player side and leave the other as None (RandomPlayer)
    policy_a = model_policy if policy_side == "a" else None
    policy_b = model_policy if policy_side == "b" else None
    name_a = "ModelA" if policy_a is not None else "RandomA"
    name_b = "ModelB" if policy_b is not None else "RandomB"

    poke_env_patches.install()
    try:
        result = await EvaluationHarness(
            episodes_per_matchup=1,
            seed=23,
            port=parsed.port,
        ).run_matchup(
            name_a,
            policy_a,
            name_b,
            policy_b,
            "fallback",
            FixedTeamSource(DEFAULT_TEST_TEAM),
            showdown_server,
        )
    finally:
        poke_env_patches.uninstall_for_tests()

    assert (result.policy_a, result.policy_b) == (name_a, name_b)
    assert result.total_games == 1
    assert result.wins_a + result.wins_b + result.ties == result.total_games
    assert result.win_rate_a == pytest.approx(result.wins_a / result.total_games)
    assert result.per_team_a_results
    assert result.per_team_b_results
    assert sum(stats["games"] for stats in result.per_team_a_results.values()) == 1
    assert sum(stats["games"] for stats in result.per_team_b_results.values()) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_evaluation_matchup_serializes_per_team_outcomes(showdown_server) -> None:
    """Verify multi-game evaluation records and aggregates per-team win/loss statistics correctly."""
    poke_env_patches.install()
    try:
        harness = EvaluationHarness(episodes_per_matchup=2, seed=17)
        result = await harness.run_matchup(
            "random-a",
            None,
            "random-b",
            None,
            "fallback",
            FixedTeamSource(DEFAULT_TEST_TEAM),
            showdown_server,
        )
    finally:
        poke_env_patches.uninstall_for_tests()

    assert result.total_games == 2
    assert result.wins_a + result.wins_b + result.ties == result.total_games

    # Validate accounting consistency across overall per_team, player A per-team, and player B per-team maps
    aggregates = (
        (result.per_team_results, result.wins_a),
        (result.per_team_a_results, result.wins_a),
        (result.per_team_b_results, result.wins_b),
    )
    for stats_by_team, expected_wins in aggregates:
        assert stats_by_team
        assert all(set(stats) == {"wins", "games", "win_rate"} for stats in stats_by_team.values())
        # The sum of games across all team keys must equal the total matchup episode count
        assert sum(stats["games"] for stats in stats_by_team.values()) == result.total_games
        # The sum of recorded wins across teams must equal the corresponding player's total wins
        assert sum(stats["wins"] for stats in stats_by_team.values()) == expected_wins
        assert all(
            0 <= stats["wins"] <= stats["games"]
            and stats["win_rate"] == pytest.approx(stats["wins"] / stats["games"])
            for stats in stats_by_team.values()
        )
