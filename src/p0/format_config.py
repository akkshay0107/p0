"""Format configuration and the load-breaking runtime contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from p0.paths import DEFAULT_PATHS

RUNTIME_MANIFEST_SCHEMA = 2
TENSOR_ABI = "champions-observation-event-v2"
RESOURCE_FEATURE_ABI = "champions-dex-features-v1"

# This is deliberately data, not a hash of the action-encoding implementation.
ACTION_CONTRACT: dict[str, Any] = {
    "joint_width": 2,
    "action_count": 49,
    "ranges": [
        {"start": 0, "end": 1, "meaning": "pass"},
        {"start": 1, "end": 7, "meaning": "switch", "roster_slots": 6},
        {
            "start": 7,
            "end": 27,
            "meaning": "move",
            "move_slots": 4,
            "targets": [-2, -1, 0, 1, 2],
        },
        {
            "start": 27,
            "end": 47,
            "meaning": "mega_move",
            "move_slots": 4,
            "targets": [-2, -1, 0, 1, 2],
        },
        {"start": 47, "end": 48, "meaning": "mega_forced_move"},
        {"start": 48, "end": 49, "meaning": "forced_move"},
    ],
    "team_preview": {
        "encoding": "ordered_roster_pair",
        "roster_size": 6,
        "joint_unique": True,
    },
    "joint_constraints": ["no_duplicate_switch", "at_most_one_mega"],
}

_MANIFEST_FIELDS = frozenset(
    {"manifest_schema", "runtime_contract", "runtime_contract_sha256", "mechanics_provenance"}
)
_RUNTIME_CONTRACT_FIELDS = frozenset(
    {"tensor_abi", "vocabulary_sha256", "action", "resource_feature_abi"}
)
_MECHANICS_PROVENANCE_FIELDS = frozenset(
    {"champions_dex_sha256", "showdown_commit", "battle_format", "bo3_format"}
)


@dataclass(frozen=True, slots=True)
class FormatSpec:
    battle_format: str = "gen9championsvgc2026regmb"
    bo3_format: str = "gen9championsvgc2026regmbbo3"
    showdown_commit: str = "8282e63102fa824fd2f7472778ec09793ceb7cac"
    generation: int = 9
    mod: str = "champions"
    action_size: int = 49


FORMAT = FormatSpec()
DEFAULT_RUNTIME_MANIFEST = DEFAULT_PATHS.data_root / "runtime_manifest.json"


def _validate_json_value(value: Any, location: str = "contract") -> None:
    """Accept the small, unambiguous JSON subset used by compatibility contracts."""
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string object key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} contains unsupported value {value!r}")


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON semantics independently of whitespace and object-key order."""
    _validate_json_value(value)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_file(path: str | Path) -> str:
    """Hash parsed JSON so formatting-only edits do not break compatibility."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON resource: {path}") from exc
    return canonical_json_sha256(value)


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """One human-readable contract for every model-tensor artifact."""

    tensor_abi: str
    vocabulary_sha256: str
    action: Mapping[str, Any]
    resource_feature_abi: str
    runtime_contract_sha256: str
    champions_dex_sha256: str
    showdown_commit: str
    battle_format: str
    bo3_format: str
    manifest_schema: int = RUNTIME_MANIFEST_SCHEMA

    def runtime_contract(self) -> dict[str, Any]:
        return {
            "tensor_abi": self.tensor_abi,
            "vocabulary_sha256": self.vocabulary_sha256,
            "action": dict(self.action),
            "resource_feature_abi": self.resource_feature_abi,
        }

    def mechanics_provenance(self) -> dict[str, str]:
        return {
            "champions_dex_sha256": self.champions_dex_sha256,
            "showdown_commit": self.showdown_commit,
            "battle_format": self.battle_format,
            "bo3_format": self.bo3_format,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema": self.manifest_schema,
            "runtime_contract": self.runtime_contract(),
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "mechanics_provenance": self.mechanics_provenance(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeManifest:
        data = dict(value)
        _validate_exact_fields(data, _MANIFEST_FIELDS, "runtime manifest")
        if data["manifest_schema"] != RUNTIME_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported runtime manifest schema {data['manifest_schema']!r}; "
                f"expected {RUNTIME_MANIFEST_SCHEMA}"
            )

        contract = data["runtime_contract"]
        provenance = data["mechanics_provenance"]
        if not isinstance(contract, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("Runtime contract and mechanics provenance must be JSON objects")
        _validate_exact_fields(contract, _RUNTIME_CONTRACT_FIELDS, "runtime contract")
        _validate_exact_fields(provenance, _MECHANICS_PROVENANCE_FIELDS, "mechanics provenance")
        _validate_json_value(contract, "runtime contract")

        digest = data["runtime_contract_sha256"]
        if not _is_sha256(digest):
            raise ValueError("runtime_contract_sha256 must be a lowercase SHA-256 digest")
        actual = canonical_json_sha256(contract)
        if digest != actual:
            raise ValueError(
                "runtime_contract_sha256 does not match the embedded runtime contract: "
                f"declared={digest}, actual={actual}"
            )
        vocabulary_digest = contract["vocabulary_sha256"]
        dex_digest = provenance["champions_dex_sha256"]
        if not _is_sha256(vocabulary_digest) or not _is_sha256(dex_digest):
            raise ValueError("Vocabulary and dex identities must be lowercase SHA-256 digests")
        for field in ("tensor_abi", "resource_feature_abi"):
            if not isinstance(contract[field], str) or not contract[field]:
                raise ValueError(f"Runtime contract {field} must be a non-empty string")
        if not isinstance(contract["action"], Mapping):
            raise ValueError("Runtime action contract must be a JSON object")
        for field in _MECHANICS_PROVENANCE_FIELDS - {"champions_dex_sha256"}:
            if not isinstance(provenance[field], str) or not provenance[field]:
                raise ValueError(f"Mechanics provenance {field} must be a non-empty string")

        return cls(
            manifest_schema=RUNTIME_MANIFEST_SCHEMA,
            tensor_abi=contract["tensor_abi"],
            vocabulary_sha256=vocabulary_digest,
            action=dict(contract["action"]),
            resource_feature_abi=contract["resource_feature_abi"],
            runtime_contract_sha256=digest,
            champions_dex_sha256=dex_digest,
            showdown_commit=provenance["showdown_commit"],
            battle_format=provenance["battle_format"],
            bo3_format=provenance["bo3_format"],
        )


def _validate_exact_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def current_manifest(
    *,
    vocab_path: str | Path = DEFAULT_PATHS.data_root / "vocab.json",
    dex_path: str | Path = DEFAULT_PATHS.data_root / "champions_dex.json",
) -> RuntimeManifest:
    """Build the current strict contract plus non-blocking mechanics provenance."""
    contract = {
        "tensor_abi": TENSOR_ABI,
        "vocabulary_sha256": sha256_json_file(vocab_path),
        "action": ACTION_CONTRACT,
        "resource_feature_abi": RESOURCE_FEATURE_ABI,
    }
    return RuntimeManifest(
        tensor_abi=TENSOR_ABI,
        vocabulary_sha256=contract["vocabulary_sha256"],
        action=ACTION_CONTRACT,
        resource_feature_abi=RESOURCE_FEATURE_ABI,
        runtime_contract_sha256=canonical_json_sha256(contract),
        # Dex identity is provenance rather than compatibility, so an exact file
        # digest is sufficient and permits ordinary numeric mechanics values.
        champions_dex_sha256=sha256_file(dex_path),
        showdown_commit=FORMAT.showdown_commit,
        battle_format=FORMAT.battle_format,
        bo3_format=FORMAT.bo3_format,
    )


def load_runtime_manifest(path: str | Path = DEFAULT_RUNTIME_MANIFEST) -> RuntimeManifest:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Global runtime manifest not found: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed global runtime manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Global runtime manifest must be a JSON object: {path}")
    return RuntimeManifest.from_dict(value)


def load_active_runtime_manifest(
    path: str | Path = DEFAULT_RUNTIME_MANIFEST,
) -> RuntimeManifest:
    """Validate the persisted contract against this build and its vocabulary."""
    path = Path(path)
    manifest = load_runtime_manifest(path)
    mismatches = []
    if manifest.tensor_abi != TENSOR_ABI:
        mismatches.append(f"tensor_abi={manifest.tensor_abi!r}, code={TENSOR_ABI!r}")
    if manifest.resource_feature_abi != RESOURCE_FEATURE_ABI:
        mismatches.append(
            f"resource_feature_abi={manifest.resource_feature_abi!r}, code={RESOURCE_FEATURE_ABI!r}"
        )
    if manifest.action != ACTION_CONTRACT:
        mismatches.append("action contract differs from the compiled action-encoding contract")
    vocab_path = path.with_name("vocab.json")
    actual_vocab = sha256_json_file(vocab_path)
    if manifest.vocabulary_sha256 != actual_vocab:
        mismatches.append(f"vocabulary={manifest.vocabulary_sha256}, actual={actual_vocab}")
    if mismatches:
        raise ValueError(
            "Runtime manifest does not describe the active runtime: " + "; ".join(mismatches)
        )
    return manifest


def validate_artifact_runtime_contract(
    artifact: Mapping[str, Any], path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> RuntimeManifest:
    """Reject artifacts whose tensor interpretation differs from the active runtime."""
    if "runtime_manifest_sha256" in artifact:
        raise ValueError(
            "Unsupported legacy checkpoint format containing runtime_manifest_sha256; "
            "create a new checkpoint with the checkpoint-4 artifact schema"
        )
    reference = artifact.get("runtime_contract_sha256")
    if not _is_sha256(reference):
        raise ValueError("Artifact has no valid runtime_contract_sha256 reference")
    manifest = load_active_runtime_manifest(path)
    if reference != manifest.runtime_contract_sha256:
        raise ValueError(
            "Artifact runtime contract is incompatible with the active runtime: "
            f"artifact={reference}, active={manifest.runtime_contract_sha256}"
        )
    return manifest
