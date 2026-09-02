"""Tests for evaluation harness construction and result serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from p0.evaluation.harness import (
    EvaluationHarness,
    MatchupResult,
    hashlib_team,
    wilson_score_interval,
)
from p0.format_config import FORMAT, current_manifest
from p0.teams.corpus import CorpusEntry, CorpusSplit, TeamCorpusManifest, corpus_content_hash
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FixedTeamSource


class TestEvaluationHarness:
    def test_evaluation_harness_falls_back_without_corpus_repeatably(self, tmp_path: Path) -> None:
        """Verify EvaluationHarness falls back to deterministic built-in team pools when corpus file is missing."""
        first = EvaluationHarness(
            teams_path=tmp_path / "missing",
            episodes_per_matchup=5,
            seed=91,
            smoke_test=True,
        )
        second = EvaluationHarness(
            teams_path=tmp_path / "missing",
            episodes_per_matchup=5,
            seed=91,
            smoke_test=True,
        )
        first_sources = first.build_team_sources()
        second_sources = second.build_team_sources()
        assert len(first_sources) == 3
        assert tuple(first_sources) == tuple(second_sources)
        for key, source in first_sources.items():
            assert isinstance(source, FixedTeamSource)
            assert first.category_metadata[key]["fallback"] is True
            first_team = source.sample(first.rng)
            second_team = second_sources[key].sample(second.rng)
            assert "Pikachu" in first_team.packed
            assert first_team.packed == second_team.packed

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
        sources = harness.build_team_sources()
        assert set(sources) == {"seen", "validation_unseen_canonical", "test_unseen_canonical"}
        assert all(isinstance(source, CorpusTeamSource) for source in sources.values())

    def test_evaluation_confidence_intervals_and_matchup_serialization_are_deterministic(
        self,
    ) -> None:
        """Verify Wilson score confidence interval calculation and matchup result serialization."""
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
            ties=0,
            win_rate_a=0.6,
            confidence_interval_a=(lower, upper),
            per_team_results={team_hash: {"wins": 3, "games": 5, "win_rate": 0.6}},
        )
        serialized = result.to_dict()
        assert serialized["confidence_interval_a"] == [lower, upper]
        assert serialized["per_team_results"][team_hash]["games"] == 5
