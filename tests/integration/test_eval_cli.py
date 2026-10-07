"""End-to-end report publication for the evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.cli.eval import main
from p0.runtime.showdown import allocate_loopback_ports
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
class TestEvaluationCLI:
    @pytest.mark.integration
    def test_completed_matchup_writes_report(self, tmp_path: Path) -> None:
        team_path = tmp_path / "team.txt"
        team_path.write_text(DEFAULT_TEST_TEAM)
        report_dir = tmp_path / "reports"
        port = allocate_loopback_ports(1)[0]

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
                "--port",
                str(port),
            ]
        )

        assert code == 0
        report = json.loads((report_dir / "evaluation_report.json").read_text())
        assert report["episodes"] == 1
        assert report["matchup"]["total_games"] == 1
        assert report["matchups"] == [report["matchup"]]
