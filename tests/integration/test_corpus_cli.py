"""Integration tests for Showdown-backed corpus CLI operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.cli.corpus import main as corpus_main
from p0.format_config import FORMAT
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
@pytest.mark.integration
class TestCorpusCLI:
    def test_cli_build_and_audit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], showdown_assets: None
    ) -> None:
        del showdown_assets
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        team_text_1 = DEFAULT_TEST_TEAM
        team_text_2 = DEFAULT_TEST_TEAM.replace("Pikachu @ Light Ball", "Raichu @ Light Ball", 1)
        (input_dir / "v1.txt").write_text(team_text_1, encoding="utf-8")
        (input_dir / "v2.txt").write_text(team_text_2, encoding="utf-8")

        all_dir = tmp_path / "pools" / "all"

        corpus_main(
            [
                "build",
                "--input",
                str(input_dir),
                "--output-dir",
                str(all_dir),
                "--format-id",
                FORMAT.battle_format,
            ]
        )

        assert (all_dir / "corpus_manifest.json").is_file()

        captured = capsys.readouterr()
        audit_data = json.loads(captured.out.split("\n")[-2]) if captured.out.strip() else {}
        assert audit_data["admitted_count"] == 2
        assert audit_data["rejected_count"] == 0

        corpus_main(["audit", "--path", str(all_dir)])
        audit_captured = capsys.readouterr()
        re_audit_data = (
            json.loads(audit_captured.out.split("\n")[-2]) if audit_captured.out.strip() else {}
        )
        assert re_audit_data["admitted_count"] == 2
