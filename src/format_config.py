"""Versioned format and artifact contracts shared by runtime and training code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

FORMAT_SPEC_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 2
ACTION_SCHEMA_VERSION = "champions-mega-v1"
EVENT_SCHEMA_VERSION = 2
VOCAB_SCHEMA_VERSION = 1
STAT_POINT_IMPUTER_VERSION = 1
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
    }
)
FORMAT_FIELDS = frozenset(
    {
        "battle_format",
        "bo3_format",
        "showdown_commit",
        "generation",
        "mod",
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
    action_size: int = 49


FORMAT = FormatSpec()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["format"] = asdict(self.format)
        return value

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
        data["format"] = FormatSpec(**format_value)
        return cls(**data)

def current_manifest(
    *,
    vocab_path: str | Path | None = None,
    dex_path: str | Path | None = None,
) -> RuntimeManifest:
    """Build a global manifest from generated vocabulary and dex files."""
    if vocab_path and not Path(vocab_path).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    return RuntimeManifest(
        vocab_sha256=sha256_file(vocab_path) if vocab_path else "",
        champions_dex_sha256=sha256_file(dex_path) if dex_path and Path(dex_path).exists() else "",
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


DEFAULT_RUNTIME_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "runtime_manifest.json"


def load_runtime_manifest(path: str | Path = DEFAULT_RUNTIME_MANIFEST) -> tuple[RuntimeManifest, str]:
    """Strictly parse the global manifest and return it with its exact file identity."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Global runtime manifest not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed global runtime manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Global runtime manifest must be a JSON object: {path}")
    manifest = RuntimeManifest.from_dict(value)
    return manifest, sha256_file(path)


def runtime_manifest_sha256(path: str | Path = DEFAULT_RUNTIME_MANIFEST) -> str:
    return load_runtime_manifest(path)[1]


def validate_artifact_manifest_reference(
    artifact: Mapping[str, Any], path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> RuntimeManifest:
    """Fail closed when an artifact does not reference the available global manifest."""
    manifest, actual_hash = load_runtime_manifest(path)
    reference = artifact.get("runtime_manifest_sha256")
    if not isinstance(reference, str) or len(reference) != 64:
        raise ValueError("Artifact has no valid runtime_manifest_sha256 reference")
    if reference != actual_hash:
        raise ValueError(
            "Artifact runtime_manifest_sha256 does not match the global runtime manifest: "
            f"artifact={reference}, global={actual_hash}"
        )
    return manifest
