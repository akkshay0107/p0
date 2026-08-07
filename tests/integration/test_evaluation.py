from __future__ import annotations

import pytest

from p0.evaluation.harness import (
    DEFAULT_TEST_TEAM,
    EvaluationHarness,
)
from p0.teams.source import FixedTeamSource


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluation_harness_completes_matchup(showdown_server) -> None:
    from p0.runtime import poke_env_patches

    poke_env_patches.install()

    import urllib.parse

    parsed = urllib.parse.urlparse(showdown_server.websocket_url)
    port = parsed.port or 8120

    harness = EvaluationHarness(
        episodes_per_matchup=1,
        seed=42,
        port=port,
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

    # Check dictionary serialization
    dct = result.to_dict()
    assert dct["total_games"] == 1
    assert len(dct["per_team_results"]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_evaluation_matchup_serializes_per_team_outcomes(showdown_server) -> None:
    from p0.runtime import poke_env_patches

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
    assert result.wins_a + result.wins_b == result.total_games
    assert result.ties == 0
    assert all(
        set(stats) == {"wins", "games", "win_rate"} for stats in result.per_team_results.values()
    )
    assert sum(stats["games"] for stats in result.per_team_results.values()) == result.total_games
    assert all(
        0 <= stats["wins"] <= stats["games"]
        and stats["win_rate"] == pytest.approx(stats["wins"] / stats["games"])
        for stats in result.per_team_results.values()
    )
