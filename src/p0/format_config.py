"""Validated runtime format defaults and artifact compatibility checks."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import orjson

from p0.contracts import (
    ContractCompatibility,
    FormatSpec,
    GlobalContract,
    build_global_contract,
    compare_global_contracts,
    is_sha256,
)
from p0.paths import DEFAULT_PATHS

DEFAULT_RUNTIME_MANIFEST = DEFAULT_PATHS.data_root / "runtime_manifest.json"


def load_active_global_contract() -> GlobalContract:
    """Validate the completed default resources against their manifest."""
    path = DEFAULT_RUNTIME_MANIFEST
    try:
        value = orjson.loads(path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Global runtime manifest not found: {path}. Run bash scripts/init-data.sh."
        ) from None
    except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
        raise ValueError(f"Malformed global runtime manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Global runtime manifest must be a JSON object: {path}")
    contract = GlobalContract.from_dict(value)

    try:
        expected = build_global_contract(DEFAULT_PATHS.data_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Incomplete runtime resources. Run bash scripts/init-data.sh."
        ) from exc
    if contract != expected:
        differences = [
            f"{field.removesuffix('_sha256')}={contract.payload('resources', level)[field]}, actual={actual}"
            for level in ("major", "minor")
            for field, actual in expected.payload("resources", level).items()
            if contract.payload("resources", level)[field] != actual
        ]
        detail = "; ".join(differences) or "source schemas changed"
        raise ValueError(
            f"Global contract does not describe active resources: {detail}. Run bash scripts/init-data.sh."
        )
    return contract


@lru_cache(maxsize=1)
def active_global_contract() -> GlobalContract:
    """Return the one active default-manifest object for runtime consumers."""
    return load_active_global_contract()


def validate_artifact_runtime_contract(
    artifact: Mapping[str, Any],
) -> GlobalContract:
    """Validate an artifact reference against the active global contract."""
    reference = artifact.get("global_contract_sha256")
    if not is_sha256(reference):
        raise ValueError("Artifact has no valid global_contract_sha256 reference")
    contract = active_global_contract()
    if reference != contract.global_sha256:
        raise ValueError(
            "Artifact global contract is incompatible with the active runtime: "
            f"artifact={reference}, active={contract.global_sha256}"
        )
    return contract


def checkpoint_contract_compatibility(
    artifact: Mapping[str, Any],
) -> ContractCompatibility:
    """Validate an embedded checkpoint snapshot against the active contract."""
    reference = artifact.get("global_contract_sha256")
    snapshot = artifact.get("global_contract")
    if not is_sha256(reference) or not isinstance(snapshot, Mapping):
        raise ValueError("Checkpoint has no valid embedded global contract snapshot")
    historical = GlobalContract.from_dict(snapshot)
    if historical.global_sha256 != reference:
        raise ValueError("Checkpoint global_contract_sha256 does not match its embedded snapshot")
    return compare_global_contracts(historical, active_global_contract())


def _format_spec_from_contract(contract: GlobalContract) -> FormatSpec:
    resources = contract.payload("resources", "minor")
    actions = contract.payload("actions", "major")
    return FormatSpec(
        battle_format=resources["battle_format"],
        bo3_format=resources["bo3_format"],
        action_size=actions["action_count"],
    )


FORMAT = _format_spec_from_contract(active_global_contract())


def is_corpus_format_compatible(model_format_id: str, corpus_format_id: str) -> bool:
    """Return whether a corpus format can provide teams to a model format."""
    return corpus_format_id == model_format_id or (
        model_format_id == FORMAT.bo3_format and corpus_format_id == FORMAT.battle_format
    )
