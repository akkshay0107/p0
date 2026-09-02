"""Tests for runtime format and artifact contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from p0.format_config import (
    ACTION_CONTRACT,
    RESOURCE_FEATURE_ABI,
    TENSOR_ABI,
    RuntimeManifest,
    active_global_contract,
    canonical_json_sha256,
    current_manifest,
    validate_artifact_runtime_contract,
)
from p0.model.config import ModelConfig
from p0.training.config import (
    BCConfig,
    GlobalConfig,
    TrainingConfig,
    load_config,
)


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _resources(
    tmp_path: Path, *, extra_species: bool = False, base_power: int = 90
) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


ROOT = Path(__file__).resolve().parents[2]


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


class TestFormatConfig:
    def test_runtime_manifest_round_trips_one_readable_contract(self, tmp_path: Path) -> None:
        """Verify RuntimeManifest serializes and deserializes losslessly while preserving ABI invariants and contract hashes."""
        vocab, dex = _resources(tmp_path)
        manifest = current_manifest(vocab_path=vocab, dex_path=dex)
        restored = RuntimeManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

        assert restored == manifest
        assert restored.tensor_abi == TENSOR_ABI
        assert restored.resource_feature_abi == RESOURCE_FEATURE_ABI
        assert restored.action == ACTION_CONTRACT
        assert restored.global_sha256 == manifest.global_sha256

    def test_canonical_hash_ignores_object_order_but_not_required_semantics(self) -> None:
        """Verify canonical JSON SHA-256 is insensitive to dictionary key ordering but sensitive to value changes."""
        first = {"shape": [31, 10], "dtype": "int64"}
        reordered = {"dtype": "int64", "shape": [31, 10]}
        changed = {"dtype": "int64", "shape": [32, 10]}

        assert canonical_json_sha256(first) == canonical_json_sha256(reordered)
        assert canonical_json_sha256(first) != canonical_json_sha256(changed)
        with pytest.raises(ValueError, match="unsupported value"):
            canonical_json_sha256({"scale": 0.5})

    def test_vocabulary_expansion_breaks_contract_but_dex_change_does_not(
        self, tmp_path: Path
    ) -> None:
        """Verify that vocabulary alterations modify global contract hash, while minor dex stats updates only alter dex checksum."""
        vocab, dex = _resources(tmp_path, base_power=90)
        original = current_manifest(vocab_path=vocab, dex_path=dex)

        # Modifying dex move basePower changes dex sha256 without breaking tensor shapes or global contract hash
        dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": 80}]}))
        rebalanced = current_manifest(vocab_path=vocab, dex_path=dex)
        assert rebalanced.global_sha256 == original.global_sha256
        assert rebalanced.champions_dex_sha256 != original.champions_dex_sha256

        # Adding new species to vocabulary changes tensor vocab sizes and breaks global contract hash
        vocab, dex = _resources(tmp_path, extra_species=True, base_power=80)
        expanded = current_manifest(vocab_path=vocab, dex_path=dex)
        assert expanded.global_sha256 != original.global_sha256

    def test_global_contract_freezes_payloads_and_bumps_only_the_required_identity(self) -> None:
        """Verify immutable subsystem payloads and version bumping mechanics on active global contract."""
        contract = active_global_contract()
        with pytest.raises(TypeError):
            contract.payload("actions", "major")["action_count"] = 50  # type: ignore[index]

        # Minor subsystem update increments minor version without altering global major contract SHA-256
        changed_minor = contract.with_subsystem_update(
            "resources",
            minor_payload={**contract.payload("resources", "minor"), "showdown_commit": "next"},
        )
        assert changed_minor.global_sha256 == contract.global_sha256
        assert (
            changed_minor.subsystem("resources").minor_version
            == contract.subsystem("resources").minor_version + 1
        )

        # Major subsystem update alters tensor ABI, bumps major version, and recomputes global SHA-256
        changed_major = contract.with_subsystem_update(
            "model",
            major_payload={**contract.payload("model", "major"), "tensor_abi": "next"},
        )
        assert changed_major.global_sha256 != contract.global_sha256
        assert (
            changed_major.subsystem("model").major_version
            == contract.subsystem("model").major_version + 1
        )

        # Major subsystem bump resets minor version counter to 0
        bumped = contract.with_subsystem_update(
            "resources",
            major_payload={
                **contract.payload("resources", "major"),
                "resource_feature_abi": "next",
            },
        )
        assert contract.subsystem("resources").minor_version > 0
        assert bumped.subsystem("resources").minor_version == 0

    def test_global_contract_rejects_hash_valid_but_malformed_subsystem_payload(self) -> None:
        """Verify RuntimeManifest.create validates non-empty payloads for all registered subsystems."""
        contract = active_global_contract()
        contracts = {
            name: {
                "major": dict(contract.payload(name, "major")),
                "minor": dict(contract.payload(name, "minor")),
            }
            for name in contract.subsystems
        }
        contracts["actions"]["major"] = {}
        with pytest.raises(ValueError, match="actions major payload"):
            RuntimeManifest.create(contracts, contract.subsystems)

    def test_artifact_validation_uses_only_the_active_global_contract(self) -> None:
        """Verify validate_artifact_runtime_contract accepts artifacts matching current global contract and rejects mismatches."""
        manifest = active_global_contract()
        artifact = {"global_contract_sha256": manifest.global_sha256}
        assert validate_artifact_runtime_contract(artifact) == manifest
        artifact["global_contract_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="incompatible"):
            validate_artifact_runtime_contract(artifact)

    def test_model_config_has_only_scaling_fields(self) -> None:
        """Verify ModelConfig accepts valid scaling architectures and rejects deprecated or incompatible parameters."""
        config = ModelConfig.baseline()
        assert config.dim_feedforward == 1536
        assert ModelConfig.from_dict(config.to_dict()) == config
        enabled = ModelConfig(
            d_model=64,
            nhead=4,
            reducer_layers=1,
            dim_feedforward=128,
        )
        assert ModelConfig.from_dict(enabled.to_dict()) == enabled
        stale = config.to_dict()
        stale["history_tokens"] = 8
        with pytest.raises(ValueError, match=r"unknown=.*history_tokens"):
            ModelConfig.from_dict(stale)
        with pytest.raises(ValueError, match="low-width event channel"):
            ModelConfig(d_model=96, nhead=3, reducer_layers=1, dim_feedforward=128)

    def test_config_sections(self, tmp_path: Path) -> None:
        """Verify current config sections load correctly and disallow conflicting objective parameters."""
        config = load_config("config.example.yaml")
        assert config.bc.batch_decisions == 256
        assert config.bc.gamma == config.training.gamma
        assert config.bc.value_coef == config.training.value_coef
        assert config.teams.all == (ROOT / "teams" / "all").resolve()
        assert config.teams.reduced == (ROOT / "teams" / "reduced").resolve()
        assert config.evaluation.episodes_per_matchup == 20
        bad = tmp_path / "config.yaml"
        bad.write_text("bc:\n  bogus: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown BCConfig field"):
            load_config(bad)

        duplicate_objective = tmp_path / "duplicate-objective.yaml"
        duplicate_objective.write_text(
            "training:\n  gamma: 0.95\nbc:\n  gamma: 0.9\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="bc.gamma is derived from training"):
            load_config(duplicate_objective)

        with pytest.raises(ValueError, match="bc.gamma must match training.gamma"):
            GlobalConfig(training=TrainingConfig(gamma=0.95), bc=BCConfig(gamma=0.9))

    def test_schema_modules_stay_pure(self) -> None:
        """Verify intermediate representation modules (p0.replays.schema, p0.battle.series) stay pure without importing torch or runtime."""
        code = (
            "import sys\n"
            "import p0.replays.schema, p0.battle.series\n"
            "assert 'torch' not in sys.modules, 'IR layer must stay torch-free'\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
            "import p0.replays.shards\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
