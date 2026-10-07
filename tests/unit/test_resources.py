"""Tests for validated runtime resources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from p0.format_config import (
    ACTION_CONTRACT,
    FORMAT,
    GlobalContract,
)
from p0.model.resources import cache_default_dex, default_runtime_resources
from p0.model.tokenizer import PokemonTokenizer

ROOT = Path(__file__).resolve().parents[2]


class TestRuntimeResources:
    def test_only_default_dex_tables_are_singletons(self) -> None:
        lookup = cache_default_dex(dict)
        dex = default_runtime_resources().dex
        assert lookup(dex) is lookup(dex)
        custom = {"species": "original"}
        original = lookup(custom)
        custom["species"] = "changed"
        assert lookup(custom)["species"] == "changed"
        assert original["species"] == "original"
        assert lookup(dex) is lookup(dex)

    def test_active_contract_is_reg_m_b_and_manifest_matches_sources(self) -> None:
        """Verify default runtime_manifest matches Champions Regulation M-B battle formats and action contract."""
        manifest = GlobalContract.from_dict(
            json.loads((ROOT / "data/runtime_manifest.json").read_text())
        )
        assert FORMAT.battle_format == "gen9championsvgc2026regmb"
        assert FORMAT.bo3_format == "gen9championsvgc2026regmbbo3"
        assert manifest.battle_format == FORMAT.battle_format
        assert manifest.bo3_format == FORMAT.bo3_format
        assert manifest.action == ACTION_CONTRACT
        assert len(manifest.global_sha256) == 64

    @pytest.mark.heavy
    @pytest.mark.parametrize("resource", ("champions_dex", "spread_usage"))
    def test_runtime_resources_reject_modified_active_resource(
        self, tmp_path: Path, resource: str
    ) -> None:
        # A real, isolated project gives the loader its default manifest without changing
        # the live checkout or substituting its filesystem/path dependencies.
        shutil.copytree(
            ROOT / "src/p0", tmp_path / "src/p0", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
        data = tmp_path / "data"
        data.mkdir()
        for name in ("runtime_manifest", "vocab", "champions_dex", "spread_usage"):
            shutil.copy2(ROOT / "data" / f"{name}.json", data / f"{name}.json")
        script = textwrap.dedent("""            import json
            import sys
            from pathlib import Path
            from p0.model.resources import RuntimeResources
            from p0.paths import DEFAULT_PATHS

            assert DEFAULT_PATHS.repository_root == Path.cwd()
            manifest = DEFAULT_PATHS.data_root / "runtime_manifest.json"
            RuntimeResources.from_manifest(manifest)
            resource = sys.argv[1]
            path = manifest.with_name(resource + ".json")
            if resource == "champions_dex":
                dex = json.loads(path.read_text())
                dex["moves"][0]["basePower"] = int(dex["moves"][0].get("basePower", 0)) + 1
                path.write_text(json.dumps(dex))
            else:
                path.write_bytes(path.read_bytes() + b" ")
            try:
                RuntimeResources.from_manifest(manifest)
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

    def test_every_legal_content_key_resolves(self) -> None:
        """Verify all legal species, items, abilities, and moves in champions_dex resolve to known tokenizer IDs."""
        vocab = json.loads((ROOT / "data/vocab.json").read_text())
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        tokenizer_instance = PokemonTokenizer(vocab)
        for table in ("species", "items", "abilities", "moves"):
            for key in dex["legality"][table]:
                assert tokenizer_instance.resolve(table, key)[1] == "known", (table, key)

    def test_reg_mb_legality_inventory_uses_resolved_showdown_rules(self) -> None:
        """Verify Regulation M-B format legality whitelist includes legal items/moves and excludes banned content."""
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        assert "pikachu" in dex["legality"]["species"]
        assert "protect" in dex["legality"]["moves"]
        assert "ababo" not in dex["legality"]["species"]
        assert "berserkgene" not in dex["legality"]["items"]
