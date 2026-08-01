"""Tests for the policy evaluation harness and command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.evaluation.harness import (
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


def test_cli_parser_help() -> None:
    from p0.cli.eval import _parser

    parser = _parser()
    args = parser.parse_args(["--checkpoint", "dummy_checkpoint.pt", "--episodes", "5"])
    assert args.checkpoint == Path("dummy_checkpoint.pt")
    assert args.episodes == 5
    assert args.opponent_checkpoint is None


def test_eval_player_early_validation_fails() -> None:
    import random

    from p0.evaluation.harness import EvalRandomPlayer

    with pytest.raises(
        ValueError, match="EvalPlayer requires either team_source or a team in kwargs"
    ):
        EvalRandomPlayer(team_rng=random.Random(0))
