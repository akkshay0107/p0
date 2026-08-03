from __future__ import annotations

import pytest

from p0.evaluation.harness import (
    EvaluationHarness,
    MatchupResult,
    hashlib_team,
    wilson_score_interval,
)


@pytest.mark.stress
def test_evaluation_confidence_intervals_and_matchup_serialization_are_deterministic() -> None:
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    lower, upper = wilson_score_interval(3, 5)
    assert 0.0 < lower < 0.6 < upper < 1.0
    team_hash = hashlib_team("team-data")
    assert team_hash == hashlib_team("team-data")
    result = MatchupResult(
        policy_a="live",
        policy_b="random",
        team_category="seen",
        total_games=5,
        wins_a=3,
        wins_b=2,
        win_rate_a=0.6,
        confidence_interval_a=(lower, upper),
        per_team_results={team_hash: {"wins": 3, "games": 5, "win_rate": 0.6}},
    )
    serialized = result.to_dict()
    assert serialized["confidence_interval_a"] == [lower, upper]
    assert serialized["per_team_results"][team_hash]["games"] == 5


@pytest.mark.stress
def test_evaluation_team_source_fallback_is_repeatable_without_a_corpus(tmp_path) -> None:
    first = EvaluationHarness(
        corpus_path=tmp_path / "missing.json",
        corpus_hash="missing",
        episodes_per_matchup=5,
        seed=91,
    )
    second = EvaluationHarness(
        corpus_path=tmp_path / "missing.json",
        corpus_hash="missing",
        episodes_per_matchup=5,
        seed=91,
    )
    first_sources = first.build_team_sources()
    second_sources = second.build_team_sources()
    assert tuple(first_sources) == tuple(second_sources)
    for key in first_sources:
        assert (
            first_sources[key].sample(first.rng).packed
            == second_sources[key].sample(second.rng).packed
        )


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_live_evaluation_matchup_serializes_per_team_outcomes(showdown_server) -> None:
    from p0.evaluation.harness import DEFAULT_TEST_TEAM
    from p0.teams.source import FixedTeamSource

    harness = EvaluationHarness(episodes_per_matchup=1, seed=17)
    result = await harness.run_matchup(
        "random-a",
        None,
        "random-b",
        None,
        "seen",
        FixedTeamSource(DEFAULT_TEST_TEAM),
        showdown_server,
    )
    assert result.total_games == 1
    assert result.wins_a + result.wins_b <= result.total_games
    assert all(
        set(stats) == {"wins", "games", "win_rate"} for stats in result.per_team_results.values()
    )
