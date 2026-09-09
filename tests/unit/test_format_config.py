"""Tests for runtime format and artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.format_config import (
    ACTION_CONTRACT,
    RESOURCE_FEATURE_ABI,
    TENSOR_ABI,
    RuntimeManifest,
    active_global_contract,
    canonical_json_sha256,
    checkpoint_contract_compatibility,
    compare_global_contracts,
    current_manifest,
    load_active_global_contract,
    validate_artifact_runtime_contract,
)


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

    def test_compare_global_contracts_detects_compatibility_levels(self) -> None:
        """Verify compare_global_contracts distinguishes compatible, warning, and incompatible contract transitions."""
        active = active_global_contract()

        # Same contract is fully compatible
        compat = compare_global_contracts(active, active)
        assert compat.is_compatible
        assert compat.status == "compatible"
        assert compat.major_differences == ()
        assert compat.minor_differences == ()

        # Minor difference yields warning
        minor_bump = active.with_subsystem_update(
            "resources",
            minor_payload={**active.payload("resources", "minor"), "showdown_commit": "next"},
        )
        compat_minor = compare_global_contracts(minor_bump, active)
        assert compat_minor.is_compatible
        assert compat_minor.status == "warning"
        assert len(compat_minor.minor_differences) == 1
        assert "resources" in compat_minor.minor_differences[0]

        # Major difference yields incompatible
        major_bump = active.with_subsystem_update(
            "model",
            major_payload={**active.payload("model", "major"), "tensor_abi": "next"},
        )
        compat_major = compare_global_contracts(major_bump, active)
        assert not compat_major.is_compatible
        assert compat_major.status == "incompatible"
        assert len(compat_major.major_differences) == 1
        assert "model" in compat_major.major_differences[0]

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

    def test_single_runtime_policy_rejection(self, tmp_path: Path) -> None:
        """Verify functions reject non-default manifest paths enforcing single-runtime-per-process."""
        fake_path = tmp_path / "custom_manifest.json"
        with pytest.raises(
            ValueError, match="The active runtime contract is always the default global manifest"
        ):
            load_active_global_contract(fake_path)

        active = active_global_contract()
        artifact = {"global_contract_sha256": active.global_sha256}
        with pytest.raises(
            ValueError, match="The active runtime contract is always the default global manifest"
        ):
            validate_artifact_runtime_contract(artifact, fake_path)

        checkpoint_artifact = {
            "global_contract_sha256": active.global_sha256,
            "global_contract": active.to_dict(),
        }
        with pytest.raises(
            ValueError, match="The active runtime contract is always the default global manifest"
        ):
            checkpoint_contract_compatibility(checkpoint_artifact, fake_path)
