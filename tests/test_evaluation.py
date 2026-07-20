"""Tests for the policy evaluation harness and command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.evaluation.harness import (
    DEFAULT_TEST_TEAM,
    EvaluationHarness,
    wilson_score_interval,
)
from p0.teams.source import FixedTeamSource


def test_wilson_score_interval_boundaries() -> None:
    # Test zero total games
    assert wilson_score_interval(0, 0) == (0.0, 0.0)

    # Test 100% wins
    lower, upper = wilson_score_interval(10, 10)
    assert lower > 0.5
    assert upper == 1.0

    # Test 0% wins
    lower, upper = wilson_score_interval(0, 10)
    assert lower == 0.0
    assert upper < 0.5

    # Test 50% wins
    lower, upper = wilson_score_interval(5, 10)
    assert lower < 0.5 < upper


def test_evaluation_harness_falls_back_without_corpus(tmp_path: Path) -> None:
    harness = EvaluationHarness(
        corpus_path=tmp_path / "nonexistent_manifest.json",
        corpus_hash="nonexistent",
        episodes_per_matchup=5,
        seed=123,
    )
    sources = harness._build_team_sources()
    assert len(sources) == 4
    for key, source in sources.items():
        assert isinstance(source, FixedTeamSource)
        # Sampled team should match DEFAULT_TEST_TEAM
        team = source.sample(harness.rng)
        assert "Pikachu" in team.packed


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
    assert result.win_rate_a in (0.0, 1.0)
    assert len(result.per_team_results) == 1

    # Check dictionary serialization
    dct = result.to_dict()
    assert dct["total_games"] == 1
    assert len(dct["per_team_results"]) == 1


def test_cli_parser_help() -> None:
    from p0.cli.eval import _parser

    parser = _parser()
    args = parser.parse_args(["--checkpoint", "dummy_checkpoint.pt", "--episodes", "5"])
    assert args.checkpoint == Path("dummy_checkpoint.pt")
    assert args.episodes == 5
    assert args.opponent_checkpoint is None
