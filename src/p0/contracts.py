"""Model compatibility and shared serialized-field checks."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import orjson

# Bump when observation/action meanings or same-shape model behavior become incompatible.
MODEL_ENCODING_VERSION = 1


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    """Vocabulary/encoding changes break compatibility; dex changes only warn."""

    major_sha256: str
    minor_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeContract:
        require_dataclass_fields(value, cls)
        return cls(**value)

    @classmethod
    def from_resources(cls, vocab: Mapping[str, Any], dex: Mapping[str, Any]) -> RuntimeContract:
        return cls(
            canonical_json_sha256({"encoding_version": MODEL_ENCODING_VERSION, "vocab": vocab}),
            canonical_json_sha256(dex),
        )


def build_runtime_contract(data_root: Path) -> RuntimeContract:
    vocab = orjson.loads((data_root / "vocab.json").read_bytes())
    dex = orjson.loads((data_root / "champions_dex.json").read_bytes())
    return RuntimeContract.from_resources(vocab, dex)


def compare_runtime_contracts(historical: RuntimeContract, active: RuntimeContract) -> str:
    if historical.major_sha256 != active.major_sha256:
        return "incompatible"
    return "warning" if historical.minor_sha256 != active.minor_sha256 else "compatible"


def require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


@lru_cache(maxsize=None)
def _dataclass_field_names(cls: type) -> frozenset[str]:
    return frozenset(field.name for field in fields(cls))


def require_dataclass_fields(value: Mapping[str, Any], cls: type, owner: str = "") -> None:
    require_exact_fields(value, _dataclass_field_names(cls), owner or cls.__name__)
