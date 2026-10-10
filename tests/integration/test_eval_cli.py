"""End-to-end report publication for the evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.cli.eval import main
from p0.teams.corpus_build import write_corpus_manifest
from tests.team_fixtures import default_test_corpus


@pytest.mark.heavy
class TestEvaluationCLI:
    @pytest.mark.integration
    def test_completed_matchup_writes_report(self, tmp_path: Path) -> None:
        team_path = tmp_path / "teams"
        write_corpus_manifest(default_test_corpus(), team_path)
        report_dir = tmp_path / "reports"

        code = main(
            [
                "--teams-path",
                str(team_path),
                "--episodes",
                "1",
                "--seed",
                "17",
                "--report-dir",
                str(report_dir),
            ]
        )

        assert code == 0
        report = json.loads((report_dir / "evaluation_report.json").read_text())
        assert report["episodes"] == 1
        assert report["matchup"]["total_games"] == 1
