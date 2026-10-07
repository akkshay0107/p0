"""Tests for runtime format and artifact contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from p0.format_config import (
    active_global_contract,
    checkpoint_contract_compatibility,
    validate_artifact_runtime_contract,
)
from p0.paths import DEFAULT_PATHS


class TestFormatConfig:
    def test_artifact_validation_uses_only_the_active_global_contract(self) -> None:
        """Verify validate_artifact_runtime_contract accepts artifacts matching current global contract and rejects mismatches."""
        manifest = active_global_contract()
        artifact = {"global_contract_sha256": manifest.global_sha256}
        assert validate_artifact_runtime_contract(artifact) == manifest
        artifact["global_contract_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="incompatible"):
            validate_artifact_runtime_contract(artifact)

    def test_checkpoint_contract_compatibility(self) -> None:
        """Verify checkpoint_contract_compatibility verifies embedded contract snapshots."""
        active = active_global_contract()
        artifact = {
            "global_contract_sha256": active.global_sha256,
            "global_contract": active.to_dict(),
        }
        compat = checkpoint_contract_compatibility(artifact)
        assert compat.is_compatible
        assert compat.status == "compatible"

        # Tampered hash
        corrupt = {
            "global_contract_sha256": "0" * 64,
            "global_contract": active.to_dict(),
        }
        with pytest.raises(ValueError, match="does not match its embedded snapshot"):
            checkpoint_contract_compatibility(corrupt)

    @pytest.mark.heavy
    @pytest.mark.parametrize("resource", ("champions_dex", "spread_usage"))
    def test_runtime_resources_reject_modified_active_resource(
        self, tmp_path: Path, resource: str
    ) -> None:
        # A real, isolated project gives the loader its default manifest without changing
        # the live checkout or substituting its filesystem/path dependencies.
        shutil.copytree(
            DEFAULT_PATHS.repository_root / "src/p0",
            tmp_path / "src/p0",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(DEFAULT_PATHS.repository_root / "pyproject.toml", tmp_path / "pyproject.toml")
        data = tmp_path / "data"
        data.mkdir()
        for name in ("runtime_manifest", "vocab", "champions_dex", "spread_usage"):
            shutil.copy2(
                DEFAULT_PATHS.repository_root / "data" / f"{name}.json", data / f"{name}.json"
            )
        script = textwrap.dedent("""\
            import json
            import sys
            from pathlib import Path
            from p0.format_config import load_active_global_contract
            from p0.paths import DEFAULT_PATHS

            assert DEFAULT_PATHS.repository_root == Path.cwd()
            load_active_global_contract()
            resource = sys.argv[1]
            path = DEFAULT_PATHS.data_root / (resource + ".json")
            if resource == "champions_dex":
                dex = json.loads(path.read_text())
                dex["moves"][0]["basePower"] = int(dex["moves"][0].get("basePower", 0)) + 1
                path.write_text(json.dumps(dex))
            else:
                path.write_bytes(path.read_bytes() + b" ")
            try:
                load_active_global_contract()
            except ValueError as error:
                assert "Global contract does not describe active resources" in str(error)
                assert resource + "=" in str(error)
            else:
                raise AssertionError("Modified active resource was accepted")
            """)

        result = subprocess.run(
            [sys.executable, "-c", script, resource],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, result.stdout + result.stderr
