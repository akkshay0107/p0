"""Authoritative, hashed runtime contract for every generated p0 artifact."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import orjson

from p0.paths import DEFAULT_PATHS

RUNTIME_MANIFEST_SCHEMA = 3
GLOBAL_CONTRACT_SCHEMA = RUNTIME_MANIFEST_SCHEMA

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBSYSTEM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SUBSYSTEM_NAMES = frozenset({"actions", "model", "resources", "replays", "checkpoints", "teams"})

DEFAULT_RUNTIME_MANIFEST = DEFAULT_PATHS.data_root / "runtime_manifest.json"


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """Format metadata projected from the resources contract for legacy callers."""

    battle_format: str
    bo3_format: str
    showdown_commit: str
    generation: int = 9
    mod: str = "champions"
    action_size: int = 49


def _validate_json_value(value: Any, location: str = "contract") -> None:
    """Accept the small, unambiguous JSON subset used by compatibility contracts."""
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, (list, tuple)):
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
    return hashlib.sha256(orjson.dumps(_thaw(value), option=orjson.OPT_SORT_KEYS)).hexdigest()


def _domain_sha256(domain: str, value: Mapping[str, Any]) -> str:
    if not domain.endswith("\0"):
        raise ValueError("Hash domain must end with a NUL separator")
    _validate_json_value(value)
    encoded = orjson.dumps(_thaw(value), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(domain.encode("ascii") + encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of exact file bytes using C-level file_digest."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_json_file(path: str | Path) -> str:
    """Hash parsed JSON so formatting-only edits do not break compatibility."""
    path = Path(path)
    try:
        value = orjson.loads(path.read_bytes())
    except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JSON resource: {path}") from exc
    return canonical_json_sha256(value)


@dataclass(frozen=True, slots=True)
class SubsystemContract:
    """A versioned major/minor identity shared by every global subsystem."""

    major_sha256: str
    minor_sha256: str
    major_version: int
    minor_version: int

    _FIELDS = frozenset({"major_sha256", "minor_sha256", "major_version", "minor_version"})

    def __post_init__(self) -> None:
        if not _is_sha256(self.major_sha256) or not _is_sha256(self.minor_sha256):
            raise ValueError("Subsystem contract hashes must be lowercase SHA-256 digests")
        if type(self.major_version) is not int or self.major_version < 0:
            raise ValueError("Subsystem major_version must be a non-negative integer")
        if type(self.minor_version) is not int or self.minor_version < 0:
            raise ValueError("Subsystem minor_version must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "major_sha256": self.major_sha256,
            "minor_sha256": self.minor_sha256,
            "major_version": self.major_version,
            "minor_version": self.minor_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SubsystemContract:
        _validate_exact_fields(value, cls._FIELDS, "subsystem contract")
        return cls(
            major_sha256=value["major_sha256"],
            minor_sha256=value["minor_sha256"],
            major_version=value["major_version"],
            minor_version=value["minor_version"],
        )


@dataclass(frozen=True, slots=True)
class GlobalContract:
    """The only authoritative contract for active runtime interpretation."""

    subsystems: Mapping[str, SubsystemContract]
    contracts: Mapping[str, Mapping[str, Mapping[str, Any]]]
    global_sha256: str
    manifest_schema: int = GLOBAL_CONTRACT_SCHEMA

    _FIELDS = frozenset({"manifest_schema", "subsystems", "contracts", "global_sha256"})

    def __post_init__(self) -> None:
        _validate_contract_structure(self.subsystems, self.contracts)
        if self.manifest_schema != GLOBAL_CONTRACT_SCHEMA:
            raise ValueError(
                f"Unsupported global contract schema {self.manifest_schema!r}; "
                f"expected {GLOBAL_CONTRACT_SCHEMA}"
            )
        if not _is_sha256(self.global_sha256):
            raise ValueError("global_sha256 must be a lowercase SHA-256 digest")

        expected = _build_subsystems(self.contracts, self.subsystems)
        if dict(self.subsystems) != expected:
            raise ValueError("Subsystem fingerprints do not match their embedded contract payloads")
        actual = _global_sha256(expected)
        if self.global_sha256 != actual:
            raise ValueError(
                "global_sha256 does not match the embedded subsystem contracts: "
                f"declared={self.global_sha256}, actual={actual}"
            )
        object.__setattr__(self, "subsystems", MappingProxyType(dict(self.subsystems)))
        object.__setattr__(self, "contracts", _freeze(self.contracts))

    @property
    def tensor_abi(self) -> str:
        return self.payload("model", "major")["tensor_abi"]

    @property
    def vocabulary_sha256(self) -> str:
        return self.payload("resources", "major")["vocabulary_sha256"]

    @property
    def action(self) -> Mapping[str, Any]:
        return self.payload("actions", "major")

    @property
    def resource_feature_abi(self) -> str:
        return self.payload("resources", "major")["resource_feature_abi"]

    @property
    def champions_dex_sha256(self) -> str:
        return self.payload("resources", "minor")["champions_dex_sha256"]

    @property
    def showdown_commit(self) -> str:
        return self.payload("resources", "minor")["showdown_commit"]

    @property
    def battle_format(self) -> str:
        return self.payload("resources", "minor")["battle_format"]

    @property
    def bo3_format(self) -> str:
        return self.payload("resources", "minor")["bo3_format"]

    def subsystem(self, name: str) -> SubsystemContract:
        try:
            return self.subsystems[name]
        except KeyError as exc:
            raise ValueError(f"Unknown global contract subsystem {name!r}") from exc

    def payload(self, name: str, level: str) -> Mapping[str, Any]:
        if level not in {"major", "minor"}:
            raise ValueError(f"Unknown contract level {level!r}")
        try:
            return self.contracts[name][level]
        except KeyError as exc:
            raise ValueError(f"Missing {level} payload for subsystem {name!r}") from exc

    def with_subsystem_update(
        self,
        name: str,
        *,
        major_payload: Mapping[str, Any] | None = None,
        minor_payload: Mapping[str, Any] | None = None,
    ) -> GlobalContract:
        """Apply one explicit major or minor version bump and recompute every affected hash."""
        if major_payload is None and minor_payload is None:
            raise ValueError("Provide a major_payload or minor_payload")
        previous = self.subsystem(name)
        contracts = {
            subsystem: {
                "major": _thaw(self.contracts[subsystem]["major"]),
                "minor": _thaw(self.contracts[subsystem]["minor"]),
            }
            for subsystem in self.subsystems
        }
        versions = dict(self.subsystems)
        if major_payload is not None:
            contracts[name]["major"] = dict(major_payload)
            if minor_payload is not None:
                contracts[name]["minor"] = dict(minor_payload)
            versions[name] = SubsystemContract("0" * 64, "0" * 64, previous.major_version + 1, 0)
        else:
            assert minor_payload is not None
            contracts[name]["minor"] = dict(minor_payload)
            versions[name] = SubsystemContract(
                "0" * 64, "0" * 64, previous.major_version, previous.minor_version + 1
            )
        return GlobalContract.create(contracts, versions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema": self.manifest_schema,
            "subsystems": {
                name: self.subsystems[name].to_dict() for name in sorted(self.subsystems)
            },
            "contracts": {name: _thaw(self.contracts[name]) for name in sorted(self.contracts)},
            "global_sha256": self.global_sha256,
        }

    @classmethod
    def create(
        cls,
        contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
        versions: Mapping[str, SubsystemContract],
    ) -> GlobalContract:
        """Create a self-validating contract after an explicit version bump."""
        _validate_contract_structure(versions, contracts)
        built = _build_subsystems(contracts, versions)
        return cls(built, contracts, _global_sha256(built))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GlobalContract:
        _validate_exact_fields(value, cls._FIELDS, "global contract")
        if value["manifest_schema"] != GLOBAL_CONTRACT_SCHEMA:
            raise ValueError(
                f"Unsupported global contract schema {value['manifest_schema']!r}; "
                f"expected {GLOBAL_CONTRACT_SCHEMA}"
            )
        raw_subsystems = value["subsystems"]
        raw_contracts = value["contracts"]
        if not isinstance(raw_subsystems, Mapping) or not isinstance(raw_contracts, Mapping):
            raise ValueError("Global contract subsystems and contracts must be JSON objects")
        if any(
            not isinstance(name, str) or not isinstance(entry, Mapping)
            for name, entry in raw_subsystems.items()
        ):
            raise ValueError("Global contract subsystem entries must be JSON objects")
        if any(
            not isinstance(name, str)
            or not isinstance(entry, Mapping)
            or any(
                not isinstance(level, str) or not isinstance(payload, Mapping)
                for level, payload in entry.items()
            )
            for name, entry in raw_contracts.items()
        ):
            raise ValueError("Global contract payload entries must be JSON objects")
        return cls(
            subsystems={
                name: SubsystemContract.from_dict(entry) for name, entry in raw_subsystems.items()
            },
            contracts={
                name: {level: dict(payload) for level, payload in entry.items()}
                for name, entry in raw_contracts.items()
            },
            global_sha256=value["global_sha256"],
            manifest_schema=value["manifest_schema"],
        )


RuntimeManifest = GlobalContract


@dataclass(frozen=True, slots=True)
class ContractCompatibility:
    """Compatibility result for a historical contract against the active contract."""

    status: str
    major_differences: tuple[str, ...]
    minor_differences: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return self.status != "incompatible"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _validate_exact_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_non_empty_string(value: Any, owner: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string")


def _validate_actions_payload(payload: Mapping[str, Any]) -> None:
    _validate_exact_fields(
        payload,
        frozenset({"joint_width", "action_count", "ranges", "team_preview", "joint_constraints"}),
        "actions major payload",
    )
    if type(payload["joint_width"]) is not int or payload["joint_width"] <= 0:
        raise ValueError("actions major payload joint_width must be positive")
    if type(payload["action_count"]) is not int or payload["action_count"] <= 0:
        raise ValueError("actions major payload action_count must be positive")
    ranges = payload["ranges"]
    if not isinstance(ranges, (list, tuple)) or not ranges:
        raise ValueError("actions major payload ranges must be a non-empty array")
    expected_start = 0
    meanings: set[str] = set()
    required_ranges = {
        "pass": frozenset({"start", "end", "meaning"}),
        "switch": frozenset({"start", "end", "meaning", "roster_slots"}),
        "move": frozenset({"start", "end", "meaning", "move_slots", "targets"}),
        "mega_move": frozenset({"start", "end", "meaning", "move_slots", "targets"}),
        "mega_forced_move": frozenset({"start", "end", "meaning"}),
        "forced_move": frozenset({"start", "end", "meaning"}),
    }
    for entry in ranges:
        if not isinstance(entry, Mapping):
            raise ValueError("actions major payload range must be an object")
        for field in ("start", "end", "meaning"):
            if field not in entry:
                raise ValueError(f"actions major payload range is missing {field}")
        if type(entry["start"]) is not int or type(entry["end"]) is not int:
            raise ValueError("actions major payload range bounds must be integers")
        if entry["start"] != expected_start or entry["end"] <= entry["start"]:
            raise ValueError("actions major payload ranges must be contiguous and non-empty")
        _require_non_empty_string(entry["meaning"], "actions major payload range meaning")
        expected_fields = required_ranges.get(entry["meaning"])
        if expected_fields is None:
            raise ValueError(f"Unsupported actions major payload range {entry['meaning']!r}")
        _validate_exact_fields(entry, expected_fields, f"actions major payload {entry['meaning']}")
        for field in ("roster_slots", "move_slots"):
            if field in entry and (type(entry[field]) is not int or entry[field] <= 0):
                raise ValueError(
                    f"actions major payload {entry['meaning']} {field} must be positive"
                )
        if "targets" in entry and (
            not isinstance(entry["targets"], (list, tuple))
            or not entry["targets"]
            or any(type(target) is not int for target in entry["targets"])
        ):
            raise ValueError(
                f"actions major payload {entry['meaning']} targets must be integer values"
            )
        if entry["meaning"] in meanings:
            raise ValueError("actions major payload range meanings must be unique")
        expected_start = entry["end"]
        meanings.add(entry["meaning"])
    if expected_start != payload["action_count"]:
        raise ValueError("actions major payload ranges must cover action_count")
    if meanings != set(required_ranges):
        raise ValueError("actions major payload ranges must define every supported action meaning")
    preview = payload["team_preview"]
    if not isinstance(preview, Mapping):
        raise ValueError("actions major payload team_preview must be an object")
    _validate_exact_fields(
        preview,
        frozenset({"encoding", "roster_size", "joint_unique"}),
        "actions major payload team_preview",
    )
    _require_non_empty_string(preview["encoding"], "actions major payload team_preview encoding")
    if type(preview["roster_size"]) is not int or preview["roster_size"] <= 0:
        raise ValueError("actions major payload team_preview roster_size must be positive")
    if type(preview["joint_unique"]) is not bool:
        raise ValueError("actions major payload team_preview joint_unique must be a boolean")
    if not isinstance(payload["joint_constraints"], (list, tuple)) or not all(
        isinstance(item, str) and item for item in payload["joint_constraints"]
    ):
        raise ValueError("actions major payload joint_constraints must contain non-empty strings")


def _validate_subsystem_payload(name: str, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    major = payloads["major"]
    minor = payloads["minor"]
    if name == "actions":
        _validate_actions_payload(major)
        _validate_exact_fields(minor, frozenset(), "actions minor payload")
    elif name == "model":
        _validate_exact_fields(
            major,
            frozenset(
                {
                    "tensor_abi",
                    "observation_schema_version",
                    "observation_entity_count",
                    "pokemon_count",
                    "owner_count",
                    "raw_event_count",
                    "history_window",
                    "series_tokens_per_game",
                    "max_prior_games",
                    "event_raw_width",
                    "pooled_event_count",
                    "self_target_sentinel",
                    "structured_observation_abi",
                    "observation_layout_sha256",
                }
            ),
            "model major payload",
        )
        for field in ("tensor_abi", "structured_observation_abi"):
            _require_non_empty_string(major[field], f"model major payload {field}")
        if not _is_sha256(major["observation_layout_sha256"]):
            raise ValueError(
                "model major payload observation_layout_sha256 must be a SHA-256 digest"
            )
        for field in set(major) - {
            "tensor_abi",
            "structured_observation_abi",
            "observation_layout_sha256",
            "self_target_sentinel",
        }:
            if type(major[field]) is not int or major[field] <= 0:
                raise ValueError(f"model major payload {field} must be positive")
        if type(major["self_target_sentinel"]) is not int:
            raise ValueError("model major payload self_target_sentinel must be an integer")
        _validate_exact_fields(minor, frozenset(), "model minor payload")
    elif name == "resources":
        _validate_exact_fields(
            major,
            frozenset({"resource_feature_abi", "vocabulary_sha256"}),
            "resources major payload",
        )
        _require_non_empty_string(
            major["resource_feature_abi"], "resources major payload resource_feature_abi"
        )
        if not _is_sha256(major["vocabulary_sha256"]):
            raise ValueError("resources major payload vocabulary_sha256 must be a SHA-256 digest")
        _validate_exact_fields(
            minor,
            frozenset({"champions_dex_sha256", "showdown_commit", "battle_format", "bo3_format"}),
            "resources minor payload",
        )
        if not _is_sha256(minor["champions_dex_sha256"]):
            raise ValueError(
                "resources minor payload champions_dex_sha256 must be a SHA-256 digest"
            )
        for field in ("showdown_commit", "battle_format", "bo3_format"):
            _require_non_empty_string(minor[field], f"resources minor payload {field}")
    elif name == "replays":
        _validate_exact_fields(
            major,
            frozenset(
                {
                    "replay_ir_schema_version",
                    "shard_artifact_schema",
                    "split_artifact_schema",
                    "compilation_semantics",
                    "parser_version",
                    "compiler_version",
                    "imputation_algorithm",
                    "imputation_version",
                }
            ),
            "replays major payload",
        )
        if (
            type(major["replay_ir_schema_version"]) is not int
            or major["replay_ir_schema_version"] <= 0
        ):
            raise ValueError("replays major payload replay_ir_schema_version must be positive")
        for field in (
            "shard_artifact_schema",
            "split_artifact_schema",
            "compilation_semantics",
            "imputation_algorithm",
        ):
            _require_non_empty_string(major[field], f"replays major payload {field}")
        for field in ("parser_version", "compiler_version", "imputation_version"):
            if type(major[field]) is not int or major[field] <= 0:
                raise ValueError(f"replays major payload {field} must be positive")
        _validate_exact_fields(minor, frozenset(), "replays minor payload")
    elif name == "checkpoints":
        _validate_exact_fields(major, frozenset({"artifact_schema"}), "checkpoints major payload")
        _require_non_empty_string(
            major["artifact_schema"], "checkpoints major payload artifact_schema"
        )
        _validate_exact_fields(minor, frozenset(), "checkpoints minor payload")
    else:
        _validate_exact_fields(
            major,
            frozenset({"corpus_manifest_schema", "stat_point_imputer_version"}),
            "teams major payload",
        )
        _require_non_empty_string(
            major["corpus_manifest_schema"], "teams major payload corpus_manifest_schema"
        )
        if (
            type(major["stat_point_imputer_version"]) is not int
            or major["stat_point_imputer_version"] <= 0
        ):
            raise ValueError("teams major payload stat_point_imputer_version must be positive")
        _validate_exact_fields(minor, frozenset(), "teams minor payload")


def _validate_contract_structure(
    subsystems: Mapping[str, SubsystemContract],
    contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    if set(subsystems) != _SUBSYSTEM_NAMES or set(contracts) != _SUBSYSTEM_NAMES:
        raise ValueError(
            "Global contract must contain exactly the known subsystems; "
            f"subsystems={sorted(subsystems)}, contracts={sorted(contracts)}"
        )
    for name in _SUBSYSTEM_NAMES:
        if not _SUBSYSTEM_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid subsystem name {name!r}")
        if not isinstance(subsystems[name], SubsystemContract):
            raise ValueError(f"Subsystem {name!r} has an invalid fingerprint")
        payloads = contracts[name]
        _validate_exact_fields(payloads, frozenset({"major", "minor"}), f"{name} payloads")
        _validate_json_value(payloads["major"], f"{name}.major")
        _validate_json_value(payloads["minor"], f"{name}.minor")
        _validate_subsystem_payload(name, payloads)


def _subsystem_major_sha256(name: str, version: int, payload: Mapping[str, Any]) -> str:
    return _domain_sha256(
        "p0/subsystem-major/v1\0", {"name": name, "major_version": version, "contract": payload}
    )


def _subsystem_minor_sha256(
    name: str, major_sha256: str, version: int, payload: Mapping[str, Any]
) -> str:
    return _domain_sha256(
        "p0/subsystem-minor/v1\0",
        {
            "name": name,
            "major_sha256": major_sha256,
            "minor_version": version,
            "contract": payload,
        },
    )


def _build_subsystems(
    contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    versions: Mapping[str, SubsystemContract],
) -> dict[str, SubsystemContract]:
    built: dict[str, SubsystemContract] = {}
    for name in sorted(_SUBSYSTEM_NAMES):
        previous = versions[name]
        major = _subsystem_major_sha256(name, previous.major_version, contracts[name]["major"])
        minor = _subsystem_minor_sha256(
            name, major, previous.minor_version, contracts[name]["minor"]
        )
        built[name] = SubsystemContract(
            major, minor, previous.major_version, previous.minor_version
        )
    return built


def _global_sha256(subsystems: Mapping[str, SubsystemContract]) -> str:
    return _domain_sha256(
        "p0/global-contract/v1\0",
        {
            "subsystems": {
                name: {
                    "major_version": subsystems[name].major_version,
                    "major_sha256": subsystems[name].major_sha256,
                }
                for name in sorted(subsystems)
            }
        },
    )


def compare_global_contracts(
    historical: GlobalContract, active: GlobalContract
) -> ContractCompatibility:
    """Classify a snapshot without allowing minor drift to hide major divergence."""
    major: list[str] = []
    minor: list[str] = []
    names = sorted(set(historical.subsystems) | set(active.subsystems))
    for name in names:
        before = historical.subsystems.get(name)
        after = active.subsystems.get(name)
        if before is None or after is None:
            major.append(f"{name}: subsystem is missing")
            continue
        if before.major_version != after.major_version or before.major_sha256 != after.major_sha256:
            major.append(
                f"{name}: major {before.major_version}/{before.major_sha256} -> "
                f"{after.major_version}/{after.major_sha256}"
            )
        elif (
            before.minor_version != after.minor_version or before.minor_sha256 != after.minor_sha256
        ):
            minor.append(
                f"{name}: minor {before.minor_version}/{before.minor_sha256} -> "
                f"{after.minor_version}/{after.minor_sha256}"
            )
    if major:
        status = "incompatible"
    elif minor:
        status = "warning"
    else:
        status = "compatible"
    return ContractCompatibility(status, tuple(major), tuple(minor))


def current_manifest(
    *,
    vocab_path: str | Path = DEFAULT_PATHS.data_root / "vocab.json",
    dex_path: str | Path = DEFAULT_PATHS.data_root / "champions_dex.json",
) -> GlobalContract:
    """Project resource files onto the checked-in contract with explicit version bumps."""
    return update_resource_contract(
        active_global_contract(), vocab_path=vocab_path, dex_path=dex_path
    )


def update_resource_contract(
    base: GlobalContract, *, vocab_path: str | Path, dex_path: str | Path
) -> GlobalContract:
    """Return a version-bumped resources contract for newly written resource files."""
    major = _thaw(base.payload("resources", "major"))
    minor = _thaw(base.payload("resources", "minor"))
    major["vocabulary_sha256"] = sha256_json_file(vocab_path)
    minor["champions_dex_sha256"] = sha256_file(dex_path)
    if major != _thaw(base.payload("resources", "major")):
        return base.with_subsystem_update("resources", major_payload=major, minor_payload=minor)
    if minor != _thaw(base.payload("resources", "minor")):
        return base.with_subsystem_update("resources", minor_payload=minor)
    return base


def load_global_contract(path: str | Path = DEFAULT_RUNTIME_MANIFEST) -> GlobalContract:
    """Load and validate the authoritative global contract from disk."""
    path = Path(path)
    try:
        value = orjson.loads(path.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(f"Global runtime manifest not found: {path}") from None
    except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
        raise ValueError(f"Malformed global runtime manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Global runtime manifest must be a JSON object: {path}")
    return GlobalContract.from_dict(value)


load_runtime_manifest = load_global_contract


def load_active_global_contract(path: str | Path = DEFAULT_RUNTIME_MANIFEST) -> GlobalContract:
    """Load a contract and verify its resources still match the active runtime files."""
    path = Path(path)
    if path.resolve() != DEFAULT_RUNTIME_MANIFEST.resolve():
        raise ValueError("The active runtime contract is always the default global manifest")
    contract = load_global_contract(path)
    resources = contract.payload("resources", "major")
    minor_resources = contract.payload("resources", "minor")
    vocab = sha256_json_file(Path(path).with_name("vocab.json"))
    dex = sha256_file(Path(path).with_name("champions_dex.json"))
    mismatches = []
    if resources["vocabulary_sha256"] != vocab:
        mismatches.append(f"vocabulary={resources['vocabulary_sha256']}, actual={vocab}")
    if minor_resources["champions_dex_sha256"] != dex:
        mismatches.append(f"champions_dex={minor_resources['champions_dex_sha256']}, actual={dex}")
    if mismatches:
        raise ValueError(
            "Global contract does not describe active resources: " + "; ".join(mismatches)
        )
    return contract


load_active_runtime_manifest = load_active_global_contract


@lru_cache(maxsize=1)
def active_global_contract() -> GlobalContract:
    """Return the one active default-manifest object for runtime consumers."""
    return load_active_global_contract(DEFAULT_RUNTIME_MANIFEST)


def validate_artifact_runtime_contract(
    artifact: Mapping[str, Any], path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> GlobalContract:
    """Validate an artifact's reference to the active global major contract."""
    reference = artifact.get("global_contract_sha256")
    if not _is_sha256(reference):
        raise ValueError("Artifact has no valid global_contract_sha256 reference")
    contract = load_active_global_contract(path)
    if reference != contract.global_sha256:
        raise ValueError(
            "Artifact global contract is incompatible with the active runtime: "
            f"artifact={reference}, active={contract.global_sha256}"
        )
    return contract


def checkpoint_contract_compatibility(
    artifact: Mapping[str, Any], path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> ContractCompatibility:
    """Validate an embedded checkpoint snapshot against the active contract."""
    reference = artifact.get("global_contract_sha256")
    snapshot = artifact.get("global_contract")
    if not _is_sha256(reference) or not isinstance(snapshot, Mapping):
        raise ValueError("Checkpoint has no valid embedded global contract snapshot")
    historical = GlobalContract.from_dict(snapshot)
    if historical.global_sha256 != reference:
        raise ValueError("Checkpoint global_contract_sha256 does not match its embedded snapshot")
    return compare_global_contracts(historical, load_active_global_contract(path))


def _format_spec_from_contract(contract: GlobalContract) -> FormatSpec:
    resources = contract.payload("resources", "minor")
    actions = contract.payload("actions", "major")
    return FormatSpec(
        battle_format=resources["battle_format"],
        bo3_format=resources["bo3_format"],
        showdown_commit=resources["showdown_commit"],
        action_size=actions["action_count"],
    )


FORMAT = _format_spec_from_contract(active_global_contract())

# Compatibility exports are projections of the loaded global contract.
TENSOR_ABI = active_global_contract().payload("model", "major")["tensor_abi"]
RESOURCE_FEATURE_ABI = active_global_contract().payload("resources", "major")[
    "resource_feature_abi"
]
ACTION_CONTRACT = dict(active_global_contract().payload("actions", "major"))
