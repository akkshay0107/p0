"""Initialization rejects the wrong checkout and invalid formats before generation."""

import json
import shutil
import subprocess
from pathlib import Path

from p0.paths import DEFAULT_PATHS


class TestInitData:
    def test_wrong_checkout_leaves_completed_data_untouched(self, tmp_path: Path) -> None:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        shutil.copyfile(
            DEFAULT_PATHS.repository_root / "scripts/init-data.sh", scripts / "init-data.sh"
        )
        subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "--quiet",
                str(DEFAULT_PATHS.showdown_root),
                "pokemon-showdown",
            ],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=p0-test",
                "-c",
                "user.email=p0-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "Record simulator gitlink",
            ],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=p0-test",
                "-c",
                "user.email=p0-test@example.invalid",
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "Exercise mismatched checkout",
            ],
            cwd=tmp_path / "pokemon-showdown",
            check=True,
        )
        data = tmp_path / "data"
        data.mkdir()
        manifest = data / "runtime_manifest.json"
        shutil.copyfile(DEFAULT_PATHS.data_root / "runtime_manifest.json", manifest)
        before = manifest.read_bytes()

        result = subprocess.run(
            ["bash", str(scripts / "init-data.sh")], capture_output=True, text=True
        )

        assert result.returncode != 0
        assert "committed gitlink" in result.stderr
        assert manifest.read_bytes() == before

    def test_invalid_format_fails_export_preflight_without_writing_data(self) -> None:
        dex = DEFAULT_PATHS.data_root / "champions_dex.json"
        before = dex.read_bytes()
        revision = json.loads(before)["source"]["commit"]

        result = subprocess.run(
            [
                "node",
                str(DEFAULT_PATHS.repository_root / "scripts/dump_champions_dex.js"),
                "gen9championsmissingformat",
                revision,
                "--check",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Unsupported Champions format" in result.stderr
        assert dex.read_bytes() == before
