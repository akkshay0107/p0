"""Checked-in protocol contract for the pinned Showdown replay surface."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from p0.paths import DEFAULT_PATHS
from p0.replays.identity import normalize_showdown_id
from p0.replays.reconstruction.classification import (
    CLASSIFICATION_REGISTRY,
    UNSUPPORTED_PREDICATES,
    UNSUPPORTED_TAGS,
)

SHOWDOWN_COMMIT = "8282e63102fa824fd2f7472778ec09793ceb7cac"
CONTRACT_PATH = DEFAULT_PATHS.data_root / "replay_protocol_contract.json"
RAW_INVENTORY_PATH = DEFAULT_PATHS.data_root / "showdown_raw_emission_inventory.json"
DEX_PATH = DEFAULT_PATHS.data_root / "champions_dex.json"


def load_protocol_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the immutable machine-readable protocol inventory."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("showdown_commit") != SHOWDOWN_COMMIT:
        raise ValueError("protocol contract has an unexpected Showdown source revision")
    if not isinstance(value.get("entries"), list):
        raise ValueError("protocol contract entries must be a list")
    return value


def load_raw_emission_inventory(path: Path = RAW_INVENTORY_PATH) -> dict[str, Any]:
    """Load the AST inventory; call validate_raw_emission_inventory explicitly."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("showdown_commit") != SHOWDOWN_COMMIT:
        raise ValueError("raw emission inventory has an unexpected Showdown revision")
    return value


def validate_raw_emission_inventory(value: dict[str, Any] | None = None) -> None:
    """Verify revision and source hashes against the current Showdown checkout."""
    if value is None:
        value = RAW_EMISSION_INVENTORY
    showdown_root = DEFAULT_PATHS.showdown_root
    gitdir_line = (showdown_root / ".git").read_text(encoding="utf-8").strip()
    gitdir = (showdown_root / gitdir_line.removeprefix("gitdir: ")).resolve()
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    revision = (
        (gitdir / head.removeprefix("ref: ")).read_text(encoding="utf-8").strip()
        if head.startswith("ref: ")
        else head
    )
    if revision != SHOWDOWN_COMMIT:
        raise ValueError("checked-out Showdown revision does not match protocol inventory")
    for item in value.get("files", ()):
        source = showdown_root / item["path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Showdown source drifted: {item['path']}")


PROTOCOL_CONTRACT = load_protocol_contract()
RAW_EMISSION_INVENTORY = load_raw_emission_inventory()
with DEX_PATH.open(encoding="utf-8") as _dex_handle:
    _DEX_CATALOG = json.load(_dex_handle)
LEGAL_EFFECT_IDS = {
    kind: frozenset(_DEX_CATALOG.get("legality", {}).get(kind, ()))
    for kind in ("moves", "items", "abilities")
}
LEGAL_PROTOCOL_EFFECTS = frozenset(_DEX_CATALOG.get("legalProtocolEffects", {}).get("effect", ()))
LEGAL_PROTOCOL_STATUSES = frozenset(_DEX_CATALOG.get("legalProtocolEffects", {}).get("status", ()))
ALL_PROTOCOL_EFFECT_IDS = frozenset(_DEX_CATALOG.get("protocolEffects", ()))
SOURCE_DYNAMIC_CAUSE_IDS = frozenset({"recoil", "drain", "stealeat", "confusion"})
VOLATILE_CONDITIONS = tuple(RAW_EMISSION_INVENTORY.get("volatile_conditions", ()))
ALL_LEGAL_EFFECT_NAMES = frozenset().union(
    *LEGAL_EFFECT_IDS.values(),
    LEGAL_PROTOCOL_EFFECTS,
    LEGAL_PROTOCOL_STATUSES,
    ALL_PROTOCOL_EFFECT_IDS,
    SOURCE_DYNAMIC_CAUSE_IDS,
    (row["id"] for row in VOLATILE_CONDITIONS),
)
ACTIVATION_EFFECTS = frozenset(RAW_EMISSION_INVENTORY.get("activation_effects", ()))
# The AST inventory records dynamic source expressions as witnesses too.  They
# are useful for the audit, but are not protocol values and must never become
# accidental allowlist entries (for example ``'+effect.name``).
KNOWN_ACTIVATION_EFFECTS = frozenset(
    normalize_showdown_id(value.split(":", 1)[1].strip() if ":" in value else value)
    for value in ACTIVATION_EFFECTS
    if not value.startswith("'+")
)
COPYABLE_VOLATILES = frozenset(
    row["id"] for row in VOLATILE_CONDITIONS if row["exists"] and not row["noCopy"]
)

# Source-derived effect families used to reject invented wire values.  The
# inventory records literal move/ability/item arguments; dynamic expressions
# are represented by the explicit family extensions below.
_EFFECTS_BY_TAG: dict[str, set[tuple[str, str]]] = {}
for _witness in PROTOCOL_CONTRACT["raw_witnesses"]:
    _tag = _witness.get("tag") or ((_witness.get("resolved_tags") or [None])[0])
    if not _tag:
        continue
    for _effect in _witness.get("normalized_effects", ()):
        _EFFECTS_BY_TAG.setdefault(_tag, set()).add(("", normalize_showdown_id(_effect)))
    for _argument in _witness.get("arguments", ()):
        _literal = re.fullmatch(r"['\"]([A-Za-z][A-Za-z0-9 .'_-]*)['\"]", _argument)
        if _literal:
            _EFFECTS_BY_TAG.setdefault(_tag, set()).add(
                ("", normalize_showdown_id(_literal.group(1)))
            )
_EFFECTS_BY_TAG.setdefault("-activate", set()).update(
    ("", value) for value in KNOWN_ACTIVATION_EFFECTS
)
_EFFECTS_BY_TAG["-weather"] = {
    ("", value) for value in {"sunnyday", "raindance", "sandstorm", "snowscape", "none"}
}
ALLOWED_EFFECTS_BY_TAG = {tag: frozenset(values) for tag, values in _EFFECTS_BY_TAG.items()}
ALLOWED_CAUSE_NAMESPACES = frozenset(
    {"move", "ability", "item", "condition", "status", "pokemon", "format", "gem"}
)
NO_COPY_VOLATILES = frozenset(
    row["id"] for row in VOLATILE_CONDITIONS if row["exists"] and row["noCopy"]
)


def validate_protocol_contract(contract: dict[str, Any] = PROTOCOL_CONTRACT) -> None:
    """Ensure each inventory entry has an executable classifier disposition."""
    allowed_dispositions = {
        "no_state_change",
        "public_state",
        "action_execution",
        "boundary_signal",
        "unsupported_state",
        "malformed",
        "input_invalid",
    }
    if any(entry.get("disposition") not in allowed_dispositions for entry in contract["entries"]):
        raise ValueError("protocol contract contains an unknown disposition")
    tags = {entry.get("tag") for entry in contract["entries"]}
    missing = set(CLASSIFICATION_REGISTRY) - tags
    if missing:
        raise ValueError(f"protocol contract is missing classifiers: {sorted(missing)}")
    if set(contract.get("unsupported_tags", ())) != set(UNSUPPORTED_TAGS):
        raise ValueError("protocol contract unsupported tag set drifted")
    expected = [list(item) for item in sorted(UNSUPPORTED_PREDICATES)]
    if contract.get("unsupported_predicates", ()) != expected:
        raise ValueError("protocol contract unsupported predicate set drifted")
    if contract.get("review_status") != "source_inventory_reachable_sites_resolved":
        raise ValueError("protocol contract contains unresolved source emission sites")
    witnesses = contract.get("raw_witnesses")
    if not isinstance(witnesses, list):
        raise ValueError("protocol contract must include raw source witnesses")
    for witness in witnesses:
        tags_for_witness = ([witness["tag"]] if witness.get("tag") is not None else []) + list(
            witness.get("resolved_tags", ())
        )
        if not tags_for_witness or any(
            tag not in CLASSIFICATION_REGISTRY and tag != "" for tag in tags_for_witness
        ):
            raise ValueError(f"raw source witness has no classifier mapping: {witness}")
        # The stream uses an empty tag for out-of-band text and separators;
        # this is an explicit protocol disposition rather than an unresolved
        # dynamic site.
        if tags_for_witness == [""]:
            if witness.get("disposition") != "no_state_change":
                raise ValueError(f"blank source witness must be state neutral: {witness}")
            continue
        classified_tags = [tag for tag in tags_for_witness if tag]
        expected_dispositions = {
            CLASSIFICATION_REGISTRY[tag].classification.value for tag in classified_tags
        }
        if (
            len(expected_dispositions) != 1
            or witness.get("disposition") not in expected_dispositions
        ):
            raise ValueError(f"raw source witness disposition disagrees with classifier: {witness}")
    witness_ids = [witness.get("id") for witness in witnesses]
    if any(not witness_id for witness_id in witness_ids) or len(set(witness_ids)) != len(
        witness_ids
    ):
        raise ValueError("raw source witness IDs must be present and unique")
    test_nodes = contract.get("test_nodes")
    if not isinstance(test_nodes, list) or not test_nodes:
        raise ValueError("protocol contract must include discovered test witness IDs")
    repository_root = DEFAULT_PATHS.repository_root
    for node in test_nodes:
        parts = node.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"malformed test witness ID: {node}")
        test_path, function_name = parts
        source = repository_root / test_path
        if not source.is_file():
            raise ValueError(f"missing test witness: {node}")
        if not any(
            line.strip().startswith(f"def {function_name}(")
            for line in source.read_text(encoding="utf-8").splitlines()
        ):
            raise ValueError(f"missing test witness function: {node}")
    stateful = contract.get("stateful_witnesses")
    if not isinstance(stateful, dict):
        raise ValueError("protocol contract must include stateful test witnesses")
    test_node_set = set(test_nodes)
    expected_stateful = set()
    for raw in contract["raw_witnesses"]:
        if raw.get("reachability") not in {"reachable-potential", "reachable-resolved"}:
            continue
        tags = ([raw["tag"]] if raw.get("tag") else []) + list(raw.get("resolved_tags", ()))
        for tag in tags:
            if (
                tag in CLASSIFICATION_REGISTRY
                and CLASSIFICATION_REGISTRY[tag].classification.value == "public_state"
                and tag not in UNSUPPORTED_TAGS
            ):
                expected_stateful.add(tag)
    if set(stateful) != expected_stateful:
        raise ValueError(
            "stateful witness keys must equal every reachable handled public-state tag: "
            f"missing={sorted(expected_stateful - set(stateful))}, "
            f"extra={sorted(set(stateful) - expected_stateful)}"
        )
    for tag, witness in stateful.items():
        if (
            tag not in CLASSIFICATION_REGISTRY
            or CLASSIFICATION_REGISTRY[tag].classification.value != "public_state"
        ):
            raise ValueError(f"invalid stateful witness tag: {tag}")
        transition_tests = witness.get("transition_tests") if isinstance(witness, dict) else None
        if not transition_tests or not set(transition_tests) <= test_node_set:
            raise ValueError(f"stateful witness tests are missing or unknown: {tag}")
        for node in transition_tests:
            test_path, function_name = node.rsplit(":", 1)
            source = repository_root / test_path
            body = source.read_text(encoding="utf-8")
            start = body.find(f"def {function_name}(")
            if start < 0 or "reduce_replay_state" not in body[start:]:
                raise ValueError(f"stateful witness is not a public reducer test: {node}")
            end = body.find("\n    def ", start + 1)
            function_body = body[start:] if end < 0 else body[start:end]
            if tag not in function_body:
                raise ValueError(
                    f"stateful witness does not contain an exact {tag!r} event: {node}"
                )


__all__ = [
    "CONTRACT_PATH",
    "ACTIVATION_EFFECTS",
    "ALLOWED_EFFECTS_BY_TAG",
    "ALLOWED_CAUSE_NAMESPACES",
    "LEGAL_EFFECT_IDS",
    "LEGAL_PROTOCOL_EFFECTS",
    "LEGAL_PROTOCOL_STATUSES",
    "KNOWN_ACTIVATION_EFFECTS",
    "COPYABLE_VOLATILES",
    "NO_COPY_VOLATILES",
    "PROTOCOL_CONTRACT",
    "RAW_EMISSION_INVENTORY",
    "RAW_INVENTORY_PATH",
    "VOLATILE_CONDITIONS",
    "SHOWDOWN_COMMIT",
    "load_protocol_contract",
    "load_raw_emission_inventory",
    "validate_raw_emission_inventory",
    "validate_protocol_contract",
]
