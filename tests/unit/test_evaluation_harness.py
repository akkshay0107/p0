"""Tests for evaluation harness construction, player instantiation, and result serialization."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from p0.evaluation.harness import (
    EvalMaxBasePowerPlayer,
    EvalPlayer,
    EvalRandomPlayer,
    EvalSimpleHeuristicsPlayer,
    EvaluationHarness,
    MatchupResult,
    create_eval_player,
    wilson_score_interval,
)
from p0.format_config import FORMAT, current_manifest
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.teams.corpus import CorpusEntry, CorpusSplit, TeamCorpusManifest, corpus_content_hash
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FixedTeamSource
from tests.team_fixtures import DEFAULT_TEST_TEAM


class TestEvaluationHarness:
    def test_evaluation_harness_fails_fast_when_teams_path_missing(self, tmp_path: Path) -> None:
        """Verify EvaluationHarness fails fast with FileNotFoundError when teams path does not exist."""
        harness = EvaluationHarness(
            teams_path=tmp_path / "missing", episodes_per_matchup=5, seed=91
        )
        with pytest.raises(FileNotFoundError):
            harness.build_team_sources()

    def test_evaluation_harness_accepts_regular_manifest_for_bo3(self, tmp_path: Path) -> None:
        entries = tuple(
            CorpusEntry(
                canonical_hash=hashlib.sha256(f"canonical-{split}".encode()).hexdigest(),
                packed=f"packed-team-{split}",
                packed_sha256=hashlib.sha256(f"packed-team-{split}".encode()).hexdigest(),
                split=split,
                usage_count=1,
            )
            for split in (CorpusSplit.TRAIN, CorpusSplit.VALIDATION, CorpusSplit.TEST)
        )
        manifest = TeamCorpusManifest(
            global_contract_sha256=current_manifest().global_sha256,
            format_id=FORMAT.battle_format,
            corpus_hash=corpus_content_hash(entries),
            entries=entries,
            created_at="2026-08-19T00:00:00Z",
            sampling_metadata={},
        )
        pool_dir = tmp_path / "all"
        pool_dir.mkdir()
        (pool_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest.to_dict()), encoding="utf-8"
        )

        harness = EvaluationHarness(teams_path=pool_dir, format_id=FORMAT.bo3_format)
        source = harness.build_team_source()
        assert isinstance(source, CorpusTeamSource)

    def test_create_eval_player_constructs_supported_opponents(self) -> None:
        """Verify create_eval_player instantiates the expected baseline or policy players."""
        team_source = FixedTeamSource(DEFAULT_TEST_TEAM)
        rng = random.Random(42)
        server_config = LocalhostServerConfiguration
        account_config = AccountConfiguration("evaluser", None)

        random_player = create_eval_player(
            "random",
            team_rng=rng,
            team_source=team_source,
            battle_format=FORMAT.bo3_format,
            server_configuration=server_config,
            account_configuration=account_config,
            start_listening=False,
        )
        assert isinstance(random_player, EvalRandomPlayer)

        max_power_player = create_eval_player(
            "max_power",
            team_rng=rng,
            team_source=team_source,
            battle_format=FORMAT.bo3_format,
            server_configuration=server_config,
            account_configuration=account_config,
            start_listening=False,
        )
        assert isinstance(max_power_player, EvalMaxBasePowerPlayer)

        heuristics_player = create_eval_player(
            "simple_heuristics",
            team_rng=rng,
            team_source=team_source,
            battle_format=FORMAT.bo3_format,
            server_configuration=server_config,
            account_configuration=account_config,
            start_listening=False,
        )
        assert isinstance(heuristics_player, EvalSimpleHeuristicsPlayer)

        policy = build_policy(ModelConfig.baseline(), default_runtime_resources())
        policy_player = create_eval_player(
            policy,
            team_rng=rng,
            team_source=team_source,
            battle_format=FORMAT.bo3_format,
            server_configuration=server_config,
            account_configuration=account_config,
            start_listening=False,
        )
        assert isinstance(policy_player, EvalPlayer)

        with pytest.raises(ValueError, match="Unknown evaluation opponent"):
            create_eval_player(
                "unsupported_player",
                team_rng=rng,
                team_source=team_source,
                battle_format=FORMAT.bo3_format,
                server_configuration=server_config,
                account_configuration=account_config,
                start_listening=False,
            )

    def test_evaluation_confidence_intervals_and_matchup_serialization_are_deterministic(
        self,
    ) -> None:
        """Verify Wilson score confidence interval calculation and matchup result serialization."""
        assert wilson_score_interval(0, 0) == (0.0, 0.0)
        lower, upper = wilson_score_interval(3, 5)
        assert 0.0 < lower < 0.6 < upper < 1.0
        team_hash = hashlib.sha256(b"team-data").hexdigest()[:8]
        result = MatchupResult(
            policy_a="live",
            policy_b="random",
            team_category="seen",
            total_games=5,
            wins_a=3,
            wins_b=2,
            ties=0,
            win_rate_a=0.6,
            confidence_interval_a=(lower, upper),
            per_team_results={team_hash: {"wins": 3, "games": 5, "win_rate": 0.6}},
        )
        serialized = result.to_dict()
        assert serialized["confidence_interval_a"] == [lower, upper]
        assert serialized["per_team_results"][team_hash]["games"] == 5
