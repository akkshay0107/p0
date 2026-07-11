"""Versioned format and artifact contracts shared by runtime and training code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

FORMAT_SPEC_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
ACTION_SCHEMA_VERSION = "champions-mega-v1"
EVENT_SCHEMA_VERSION = 1
VOCAB_SCHEMA_VERSION = 1
STAT_POINT_IMPUTER_VERSION = 0
REPLAY_COMPILER_VERSION = 0

MANIFEST_FIELDS = frozenset(
    {
        "format",
        "format_spec_version",
        "observation_schema_version",
        "action_schema_version",
        "event_schema_version",
        "vocab_schema_version",
        "vocab_sha256",
        "champions_dex_sha256",
        "stat_point_imputer_version",
        "replay_compiler_version",
        "model_config",
    }
)
FORMAT_FIELDS = frozenset(
    {
        "battle_format",
        "bo3_format",
        "showdown_commit",
        "generation",
        "mod",
        "action_schema",
        "action_size",
    }
)


@dataclass(frozen=True, slots=True)
class FormatSpec:
    battle_format: str = "gen9championsvgc2026regmb"
    bo3_format: str = "gen9championsvgc2026regmbbo3"
    showdown_commit: str = "8282e63102fa824fd2f7472778ec09793ceb7cac"
    generation: int = 9
    mod: str = "champions"
    action_schema: str = ACTION_SCHEMA_VERSION
    action_size: int = 49


FORMAT = FormatSpec()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize contract data consistently for manifests and hash inputs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """Identity of every generated artifact that can contain model tensors."""

    format: FormatSpec = FORMAT
    format_spec_version: int = FORMAT_SPEC_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    action_schema_version: str = ACTION_SCHEMA_VERSION
    event_schema_version: int = EVENT_SCHEMA_VERSION
    vocab_schema_version: int = VOCAB_SCHEMA_VERSION
    vocab_sha256: str = ""
    champions_dex_sha256: str = ""
    stat_point_imputer_version: int = STAT_POINT_IMPUTER_VERSION
    replay_compiler_version: int = REPLAY_COMPILER_VERSION
    model_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["format"] = asdict(self.format)
        value["model_config"] = dict(self.model_config)
        return value

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeManifest:
        data = dict(value)
        missing = MANIFEST_FIELDS - data.keys()
        unknown = data.keys() - MANIFEST_FIELDS
        if missing or unknown:
            raise ValueError(
                f"Invalid runtime manifest fields; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        format_value = data["format"]
        if not isinstance(format_value, Mapping):
            raise ValueError("Runtime manifest 'format' must be a mapping")
        format_missing = FORMAT_FIELDS - format_value.keys()
        format_unknown = format_value.keys() - FORMAT_FIELDS
        if format_missing or format_unknown:
            raise ValueError(
                f"Invalid format spec fields; missing={sorted(format_missing)}, "
                f"unknown={sorted(format_unknown)}"
            )
        if not isinstance(data["model_config"], Mapping):
            raise ValueError("Runtime manifest 'model_config' must be a mapping")
        data["format"] = FormatSpec(**format_value)
        return cls(**data)

    def validate_compatible(self, expected: RuntimeManifest | None = None) -> None:
        """Raise before tensors are loaded if artifact assumptions differ."""
        expected = expected or RuntimeManifest()
        actual = self.to_dict()
        required = expected.to_dict()
        # Model configuration may legitimately be checked by a caller that knows the
        # requested architecture; all other fields are always contract identity.
        for key, expected_value in required.items():
            if key == "model_config" and not expected_value:
                continue
            if actual.get(key) != expected_value:
                raise ValueError(
                    f"Incompatible runtime manifest field {key!r}: "
                    f"artifact={actual.get(key)!r}, expected={expected_value!r}"
                )


def current_manifest(
    *,
    vocab_path: str | Path | None = None,
    dex_path: str | Path | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> RuntimeManifest:
    """Build the current process manifest without requiring the pending dex dump."""
    if vocab_path and not Path(vocab_path).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    return RuntimeManifest(
        vocab_sha256=sha256_file(vocab_path) if vocab_path else "",
        champions_dex_sha256=sha256_file(dex_path) if dex_path and Path(dex_path).exists() else "",
        model_config=dict(model_config or {}),
    )


def policy_model_config(policy: Any) -> dict[str, Any]:
    """Extract the constructor-critical dimensions from a PolicyNet-like object."""
    if not all(hasattr(policy, name) for name in ("seq_len", "feat_dim", "act_size", "d_model")):
        # Lightweight policy doubles used by pool-management tests and tooling do
        # not carry architecture metadata. They can be persisted, but cannot be
        # loaded as a PolicyNet until a real manifest is supplied.
        return {}
    return {
        "obs_dim": [int(policy.seq_len), int(policy.feat_dim)],
        "act_size": int(policy.act_size),
        "d_model": int(policy.d_model),
        "nhead": int(policy.actor.reducer.encoder.layers[0].nhead),
        "nlayer": int(len(policy.actor.reducer.encoder.layers)),
    }


def runtime_manifest_for_policy(policy: Any) -> RuntimeManifest:
    root = Path(__file__).resolve().parents[1]
    return current_manifest(
        vocab_path=root / "data" / "vocab.json",
        dex_path=root / "data" / "champions_dex.json",
        model_config=policy_model_config(policy),
    )


def runtime_manifest_for_data() -> RuntimeManifest:
    """Return the contract for tensor artifacts that do not contain model weights."""
    root = Path(__file__).resolve().parents[1]
    return current_manifest(vocab_path=root / "data" / "vocab.json")
