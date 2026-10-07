"""Tests for source-owned schemas, resource hashes and checkpoint compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.contracts import (
    GlobalContract,
    build_global_contract,
    canonical_json_sha256,
    compare_global_contracts,
)
from p0.format_config import active_global_contract
from p0.paths import DEFAULT_PATHS


def _historical_contract(
    base: GlobalContract, name: str, *, major_payload=None, minor_payload=None
) -> GlobalContract:
    contracts = base.to_dict()["contracts"]
    versions = {
        name: (entry.major_version, entry.minor_version) for name, entry in base.subsystems.items()
    }
    previous = base.subsystems[name]
    if major_payload is not None:
        contracts[name]["major"] = major_payload
        versions[name] = (previous.major_version + 1, 0)
    else:
        contracts[name]["minor"] = minor_payload
        versions[name] = (previous.major_version, previous.minor_version + 1)
    return GlobalContract.create(contracts, versions)


class TestContracts:
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
        source = json.loads((DEFAULT_PATHS.data_root / "champions_dex.json").read_text())["source"]
        vocab = tmp_path / "vocab.json"
        dex = tmp_path / "champions_dex.json"
        vocab.write_text('{"species":{"pikachu":1}}')
        dex.write_text(json.dumps({"source": source, "moves": [{"id": "test", "basePower": 90}]}))
        (tmp_path / "spread_usage.json").write_text("{}")
        original = build_global_contract(tmp_path)

        # Modifying dex move basePower changes dex sha256 without breaking tensor shapes or global contract hash
        dex.write_text(json.dumps({"source": source, "moves": [{"id": "test", "basePower": 80}]}))
        rebalanced = build_global_contract(tmp_path)
        assert rebalanced.global_sha256 == original.global_sha256
        assert compare_global_contracts(original, rebalanced).status == "warning"
        assert (
            rebalanced.subsystems["resources"].major_version
            == original.subsystems["resources"].major_version
        )
        assert (
            rebalanced.subsystems["resources"].minor_version
            == original.subsystems["resources"].minor_version
        )

        # Adding new species to vocabulary changes tensor vocab sizes and breaks global contract hash
        vocab.write_text('{"species":{"pikachu":1,"raichu":2}}')
        expanded = build_global_contract(tmp_path)
        assert expanded.global_sha256 != original.global_sha256

    def test_contract_freezes_nested_payloads(self) -> None:
        contract = active_global_contract()
        with pytest.raises(TypeError):
            contract.payload("actions", "major")["action_count"] = 50  # type: ignore[index]

    def test_global_contract_rejects_hash_valid_but_malformed_subsystem_payload(self) -> None:
        """Verify GlobalContract.create validates non-empty payloads for all registered subsystems."""
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
            GlobalContract.create(
                contracts,
                {
                    name: (entry.major_version, entry.minor_version)
                    for name, entry in contract.subsystems.items()
                },
            )

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
        minor_bump = _historical_contract(
            active,
            "resources",
            minor_payload={**active.payload("resources", "minor"), "showdown_commit": "next"},
        )
        compat_minor = compare_global_contracts(minor_bump, active)
        assert compat_minor.is_compatible
        assert compat_minor.status == "warning"
        assert len(compat_minor.minor_differences) == 1
        assert "resources" in compat_minor.minor_differences[0]

        # Major difference yields incompatible
        major_bump = _historical_contract(
            active,
            "model",
            major_payload={**active.payload("model", "major"), "pokemon_count": 14},
        )
        compat_major = compare_global_contracts(major_bump, active)
        assert not compat_major.is_compatible
        assert compat_major.status == "incompatible"
        assert len(compat_major.major_differences) == 1
        assert "model" in compat_major.major_differences[0]
