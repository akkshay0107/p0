"""Tests for vocabulary generation from the pinned dex."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from p0.cli.build_vocab import build

ROOT = Path(__file__).resolve().parents[2]


class TestBuildVocab:
    def test_generation_is_deterministic_and_nonlegal_effects_are_reported(
        self, tmp_path: Path
    ) -> None:
        """Verify build_vocab execution is bitwise deterministic and unsupported nonlegal effects are logged in coverage audit."""
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        dex["protocolEffects"].append("nonlegaltesteffect")
        dex_path = tmp_path / "dex.json"
        dex_path.write_text(json.dumps(dex), encoding="utf-8")
        outputs = []
        for suffix in ("a", "b"):
            vocab = tmp_path / f"vocab-{suffix}.json"
            manifest = tmp_path / f"manifest-{suffix}.json"
            coverage = tmp_path / f"coverage-{suffix}.json"
            build(dex_path, vocab, manifest, coverage)
            outputs.append((vocab.read_bytes(), manifest.read_bytes(), coverage.read_bytes()))
        assert outputs[0] == outputs[1]
        report = json.loads(outputs[0][2])
        assert "condition:nonlegaltesteffect" in report["unsupportedNonlegalEffects"]

    def test_unknown_legal_effect_namespace_fails_generation(self, tmp_path: Path) -> None:
        """Verify build_vocab fails with ValueError if an unknown effect namespace is introduced into legal effects."""
        dex = deepcopy(json.loads((ROOT / "data/champions_dex.json").read_text()))
        dex["legalProtocolEffects"]["unmapped_family"] = ["reachableeffect"]
        dex_path = tmp_path / "dex.json"
        dex_path.write_text(json.dumps(dex), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown legal protocol-effect namespace"):
            build(
                dex_path,
                tmp_path / "vocab.json",
                tmp_path / "manifest.json",
                tmp_path / "coverage.json",
            )
