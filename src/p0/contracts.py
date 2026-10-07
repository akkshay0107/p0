"""Source-owned schemas, hashes and checkpoint compatibility rules."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

import orjson

GLOBAL_CONTRACT_SCHEMA = 1

# Change these versions only when the maintained schema changes.
SCHEMA_VERSIONS = {"actions": (1, 0), "model": (1, 0), "resources": (1, 0)}
ACTION_CONTRACT = {
    "action_count": 49,
    "joint_constraints": ["no_duplicate_switch", "at_most_one_mega"],
    "joint_width": 2,
    "ranges": [
        {"end": 1, "meaning": "pass", "start": 0},
        {"end": 7, "meaning": "switch", "roster_slots": 6, "start": 1},
        {"end": 27, "meaning": "move", "move_slots": 4, "start": 7, "targets": [-2, -1, 0, 1, 2]},
        {
            "end": 47,
            "meaning": "mega_move",
            "move_slots": 4,
            "start": 27,
            "targets": [-2, -1, 0, 1, 2],
        },
        {"end": 48, "meaning": "mega_forced_move", "start": 47},
        {"end": 49, "meaning": "forced_move", "start": 48},
    ],
    "team_preview": {"encoding": "ordered_roster_pair", "joint_unique": True, "roster_size": 6},
}
MODEL_CONTRACT = {
    "event_raw_width": 128,
    "history_window": 48,
    "max_prior_games": 2,
    "observation_entity_count": 15,
    "observation_layout_sha256": "7dca9c231a7dfc3ee26646e5ccbbd6885b90b29293b8257683275d6c4f05c51a",
    "owner_count": 3,
    "pokemon_count": 12,
    "pooled_event_count": 4,
    "raw_event_count": 32,
    "self_target_sentinel": -1,
    "series_tokens_per_game": 4,
}


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUBSYSTEM_NAMES = frozenset({"actions", "model", "resources"})


class FormatSpec(NamedTuple):
    """Format metadata projected from the runtime resources and actions contract."""

    battle_format: str
    bo3_format: str
    action_size: int


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
    return hashlib.sha256(
        orjson.dumps(value, default=dict, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def _domain_sha256(domain: str, value: Mapping[str, Any]) -> str:
    _validate_json_value(value)
    encoded = orjson.dumps(value, default=dict, option=orjson.OPT_SORT_KEYS)
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

    def __post_init__(self) -> None:
        if not is_sha256(self.major_sha256) or not is_sha256(self.minor_sha256):
            raise ValueError("Subsystem contract hashes must be lowercase SHA-256 digests")
        if type(self.major_version) is not int or self.major_version < 0:
            raise ValueError("Subsystem major_version must be a non-negative integer")
        if type(self.minor_version) is not int or self.minor_version < 0:
            raise ValueError("Subsystem minor_version must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SubsystemContract:
        require_dataclass_fields(value, cls, "subsystem contract")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class GlobalContract:
    """The only authoritative contract for active runtime interpretation."""

    subsystems: Mapping[str, SubsystemContract]
    contracts: Mapping[str, Mapping[str, Mapping[str, Any]]]
    global_sha256: str
    manifest_schema: int = GLOBAL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        _validate_contract_structure(self.subsystems, self.contracts)
        if self.manifest_schema != GLOBAL_CONTRACT_SCHEMA:
            raise ValueError(
                f"Unsupported global contract schema {self.manifest_schema!r}; "
                f"expected {GLOBAL_CONTRACT_SCHEMA}"
            )
        if not is_sha256(self.global_sha256):
            raise ValueError("global_sha256 must be a lowercase SHA-256 digest")

        expected = _build_subsystems(
            self.contracts,
            {
                name: (entry.major_version, entry.minor_version)
                for name, entry in self.subsystems.items()
            },
        )
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

    def payload(self, name: str, level: str) -> Mapping[str, Any]:
        if level not in {"major", "minor"}:
            raise ValueError(f"Unknown contract level {level!r}")
        try:
            return self.contracts[name][level]
        except KeyError as exc:
            raise ValueError(f"Missing {level} payload for subsystem {name!r}") from exc

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
        versions: Mapping[str, tuple[int, int]],
    ) -> GlobalContract:
        """Create a self-validating snapshot using explicit source schema versions."""
        built = _build_subsystems(contracts, versions)
        return cls(built, contracts, _global_sha256(built))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GlobalContract:
        require_dataclass_fields(value, cls, "global contract")
        if value["manifest_schema"] != GLOBAL_CONTRACT_SCHEMA:
            raise ValueError(
                f"Unsupported global contract schema {value['manifest_schema']!r}; "
                f"expected {GLOBAL_CONTRACT_SCHEMA}"
            )
        raw_subsystems = value["subsystems"]
        raw_contracts = value["contracts"]
        if not isinstance(raw_subsystems, Mapping) or not isinstance(raw_contracts, Mapping):
            raise ValueError("Global contract subsystems and contracts must be JSON objects")
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


@dataclass(frozen=True, slots=True)
class ContractCompatibility:
    """Compatibility result for a historical contract against the active contract."""

    status: str
    major_differences: tuple[str, ...]
    minor_differences: tuple[str, ...]

    @property
    def is_compatible(self) -> bool:
        return self.status != "incompatible"


def is_sha256(value: Any) -> bool:
    """Return whether value is a lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    """Reject a serialized object whose keys are not exactly the expected fields."""
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


@lru_cache(maxsize=None)
def _dataclass_field_names(cls: type) -> frozenset[str]:
    return frozenset(field.name for field in fields(cls))


def require_dataclass_fields(value: Mapping[str, Any], cls: type, owner: str = "") -> None:
    """Reject a serialized object whose keys are not exactly the dataclass's fields."""
    require_exact_fields(value, _dataclass_field_names(cls), owner or cls.__name__)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _require_positive_int(value: Any, owner: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{owner} must be positive")


def _require_non_empty_string(value: Any, owner: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string")


def _validate_actions_payload(payload: Mapping[str, Any]) -> None:
    require_exact_fields(
        payload,
        frozenset({"joint_width", "action_count", "ranges", "team_preview", "joint_constraints"}),
        "actions major payload",
    )
    _require_positive_int(payload["joint_width"], "actions major payload joint_width")
    _require_positive_int(payload["action_count"], "actions major payload action_count")
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
        require_exact_fields(entry, expected_fields, f"actions major payload {entry['meaning']}")
        for field in ("roster_slots", "move_slots"):
            if field in entry:
                _require_positive_int(
                    entry[field], f"actions major payload {entry['meaning']} {field}"
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
    require_exact_fields(
        preview,
        frozenset({"encoding", "roster_size", "joint_unique"}),
        "actions major payload team_preview",
    )
    _require_non_empty_string(preview["encoding"], "actions major payload team_preview encoding")
    _require_positive_int(preview["roster_size"], "actions major payload team_preview roster_size")
    if type(preview["joint_unique"]) is not bool:
        raise ValueError("actions major payload team_preview joint_unique must be a boolean")
    if not isinstance(payload["joint_constraints"], (list, tuple)) or not all(
        isinstance(item, str) and item for item in payload["joint_constraints"]
    ):
        raise ValueError("actions major payload joint_constraints must contain non-empty strings")


_RESOURCE_SOURCE_FIELDS = ("showdown_commit", "battle_format", "bo3_format")
_RESOURCE_HASH_FIELDS = ("champions_dex_sha256", "spread_usage_sha256")


def _require_sha256(value: Any, owner: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{owner} must be a SHA-256 digest")


def _validate_subsystem_payload(name: str, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    major = payloads["major"]
    minor = payloads["minor"]
    if name == "actions":
        _validate_actions_payload(major)
    elif name == "model":
        require_exact_fields(major, frozenset(MODEL_CONTRACT), "model major payload")
        for field, value in major.items():
            owner = f"model major payload {field}"
            if field == "observation_layout_sha256":
                _require_sha256(value, owner)
            elif field == "self_target_sentinel":
                if type(value) is not int:
                    raise ValueError(f"{owner} must be an integer")
            else:
                _require_positive_int(value, owner)
    else:
        require_exact_fields(major, frozenset({"vocabulary_sha256"}), "resources major payload")
        _require_sha256(major["vocabulary_sha256"], "resources major payload vocabulary_sha256")

    if name == "resources":
        require_exact_fields(
            minor,
            frozenset((*_RESOURCE_SOURCE_FIELDS, *_RESOURCE_HASH_FIELDS)),
            "resources minor payload",
        )
        for field in _RESOURCE_SOURCE_FIELDS:
            _require_non_empty_string(minor[field], f"resources minor payload {field}")
        for field in _RESOURCE_HASH_FIELDS:
            _require_sha256(minor[field], f"resources minor payload {field}")
    else:
        require_exact_fields(minor, frozenset(), f"{name} minor payload")


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
        if not isinstance(subsystems[name], SubsystemContract):
            raise ValueError(f"Subsystem {name!r} has an invalid fingerprint")
        payloads = contracts[name]
        require_exact_fields(payloads, frozenset({"major", "minor"}), f"{name} payloads")
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
    versions: Mapping[str, tuple[int, int]],
) -> dict[str, SubsystemContract]:
    require_exact_fields(contracts, _SUBSYSTEM_NAMES, "global contract payloads")
    require_exact_fields(versions, _SUBSYSTEM_NAMES, "global contract versions")
    built: dict[str, SubsystemContract] = {}
    for name in sorted(_SUBSYSTEM_NAMES):
        require_exact_fields(contracts[name], frozenset({"major", "minor"}), f"{name} payloads")
        major_version, minor_version = versions[name]
        major = _subsystem_major_sha256(name, major_version, contracts[name]["major"])
        minor = _subsystem_minor_sha256(name, major, minor_version, contracts[name]["minor"])
        built[name] = SubsystemContract(major, minor, major_version, minor_version)
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
    """Classify differences between a historical contract snapshot and the active contract."""
    major: list[str] = []
    minor: list[str] = []
    for name in sorted(_SUBSYSTEM_NAMES):
        before, after = historical.subsystems[name], active.subsystems[name]
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
    status = "incompatible" if major else ("warning" if minor else "compatible")
    return ContractCompatibility(status, tuple(major), tuple(minor))


ACTION_CONTRACT = _freeze(ACTION_CONTRACT)
MODEL_CONTRACT = _freeze(MODEL_CONTRACT)


def build_global_contract(data_root: Path) -> GlobalContract:
    """Construct the existing compatibility snapshot from completed resources."""
    source = orjson.loads((data_root / "champions_dex.json").read_bytes())["source"]
    contracts = {
        "actions": {"major": ACTION_CONTRACT, "minor": {}},
        "model": {"major": MODEL_CONTRACT, "minor": {}},
        "resources": {
            "major": {"vocabulary_sha256": sha256_json_file(data_root / "vocab.json")},
            "minor": {
                "showdown_commit": source["commit"],
                "battle_format": source["battleFormat"],
                "bo3_format": source["bo3Format"],
                "champions_dex_sha256": sha256_file(data_root / "champions_dex.json"),
                "spread_usage_sha256": sha256_file(data_root / "spread_usage.json"),
            },
        },
    }
    return GlobalContract.create(contracts, SCHEMA_VERSIONS)
