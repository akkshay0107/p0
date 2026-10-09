"""End-to-end report publication for the evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.cli.eval import main
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
class TestEvaluationCLI:
    @pytest.mark.integration
    def test_completed_matchup_writes_report(self, tmp_path: Path) -> None:
        team_path = tmp_path / "team.txt"
        team_path.write_text(DEFAULT_TEST_TEAM)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("{}\n", encoding="utf-8")
        report_dir = tmp_path / "reports"

        code = main(
            [
                "--config",
                str(config_path),
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
