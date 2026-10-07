"""Prepared team sources and sampling used by runtime composition."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import orjson
from poke_env.teambuilder import Teambuilder

from p0.format_config import is_corpus_format_compatible
from p0.teams.corpus import (
    CorpusEntry,
    TeamCorpusManifest,
    load_corpus_manifest,
)

JsonScalar = str | int | float | bool | None
CORPUS_MANIFEST_NAME = "corpus_manifest.json"


class _Packer(Teambuilder):
    def yield_team(self) -> str:
        raise RuntimeError("The packing helper is not a runtime team builder")


_PACKER = _Packer()


@dataclass(frozen=True, slots=True)
class ValidatedTeam:
    packed: str
    team_hash: str

    def __post_init__(self) -> None:
        if not self.packed.strip():
            raise ValueError("A validated team must have a packed representation")

        if len(self.team_hash) != 64:
            raise ValueError("A validated team hash must be SHA-256")

    @classmethod
    def from_showdown(cls, text: str) -> ValidatedTeam:
        try:
            members = _PACKER.parse_showdown_team(text)
            if len(members) != 6:
                raise ValueError("Expected exactly six team members")
            packed = _PACKER.join_team(members)
        # The third-party teambuilder exposes several parser exception types;
        # normalize all of them at this boundary to the TeamSource contract.
        except Exception as exc:
            raise ValueError("Malformed Showdown team") from exc
        if not packed:
            raise ValueError("Malformed Showdown team")
        return cls(packed=packed, team_hash=hashlib.sha256(packed.encode()).hexdigest())


class TeamSource(Protocol):
    def sample(self, rng: random.Random) -> ValidatedTeam: ...

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]: ...


class FileTeamSource:
    """A deterministically discovered and eagerly prepared file pool."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Teams directory not found: {self.directory}")
        files = tuple(
            path
            for path in sorted(self.directory.iterdir(), key=lambda item: item.name)
            if path.is_file() and not path.name.startswith(".")
        )
        if not files:
            raise FileNotFoundError(f"No team files found in {self.directory}")
        self._initialize(files)

    @classmethod
    def from_files(cls, paths: Sequence[str | Path]) -> FileTeamSource:
        source = cls.__new__(cls)
        source.directory = Path(".")
        files = tuple(
            sorted(
                (Path(path) for path in paths if not Path(path).name.startswith(".")),
                key=lambda item: str(item),
            )
        )
        if not files:
            raise FileNotFoundError("No team files were provided")
        missing = tuple(path for path in files if not path.is_file())
        if missing:
            raise FileNotFoundError(f"Team file not found: {missing[0]}")
        source._initialize(files)
        return source

    def _initialize(self, files: tuple[Path, ...]) -> None:
        self._teams = tuple(self._read(path) for path in files)
        identity = "\n".join(team.team_hash for team in self._teams).encode()
        self._pool_id = hashlib.sha256(identity).hexdigest()

    @staticmethod
    def _read(path: Path) -> ValidatedTeam:
        try:
            return ValidatedTeam.from_showdown(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"Malformed team file: {path}") from exc

    def sample(self, rng: random.Random) -> ValidatedTeam:
        return rng.choice(self._teams)

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]:
        return {
            "kind": "file_pool",
            "format": "showdown-export",
            "pool_id": self._pool_id,
            "team_hashes": tuple(team.team_hash for team in self._teams),
        }


class FixedTeamSource:
    def __init__(self, team: ValidatedTeam | str):
        self._team = team if isinstance(team, ValidatedTeam) else ValidatedTeam.from_showdown(team)

    def sample(self, rng: random.Random) -> ValidatedTeam:
        return self._team

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]:
        return {"kind": "fixed", "team_hashes": (self._team.team_hash,)}


class CorpusTeamSource:
    """A load-time verified, corpus-backed team sampling source."""

    def __init__(
        self,
        manifest: TeamCorpusManifest,
        corpus_path: str = "",
    ) -> None:
        if not manifest.entries:
            raise ValueError("Corpus manifest contains no entries")

        by_canonical: dict[str, list[CorpusEntry]] = {}
        for entry in manifest.entries:
            by_canonical.setdefault(entry.canonical_hash, []).append(entry)

        self._manifest = manifest
        self._corpus_path = corpus_path
        self._canonical_pools = tuple(tuple(by_canonical[key]) for key in sorted(by_canonical))
        self._team_hashes = tuple(sorted(entry.packed_sha256 for entry in manifest.entries))

    @classmethod
    def from_path(cls, path: str | Path) -> CorpusTeamSource:
        """Load and validate one corpus manifest from disk."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Corpus manifest file not found: {resolved}")

        try:
            raw_data = orjson.loads(resolved.read_bytes())
        except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
            raise ValueError(f"Malformed corpus manifest file: {resolved}") from exc

        manifest = load_corpus_manifest(raw_data)
        return cls(manifest, corpus_path=str(resolved))

    def _sample_entry(
        self,
        rng: random.Random,
    ) -> CorpusEntry:
        return rng.choice(rng.choice(self._canonical_pools))

    def sample(self, rng: random.Random) -> ValidatedTeam:
        """Return a single validated team sampled uniformly by canonical team."""
        entry = self._sample_entry(rng)
        return ValidatedTeam(packed=entry.packed, team_hash=entry.packed_sha256)

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]:
        """Describe the active corpus pool and sampling configuration."""
        return {
            "kind": "corpus",
            "corpus_path": self._corpus_path,
            "corpus_hash": self._manifest.corpus_hash,
            "format_id": self._manifest.format_id,
            "sampling": "uniform_canonical",
            "pool_size": len(self._manifest.entries),
            "team_hashes": self._team_hashes,
        }


def corpus_manifest_path(path: str | Path) -> Path:
    """Return the manifest path represented by a team pool path."""
    resolved = Path(path)
    if resolved.is_dir():
        return resolved / CORPUS_MANIFEST_NAME
    if resolved.is_file() and resolved.name == CORPUS_MANIFEST_NAME:
        return resolved
    raise ValueError(f"Path is not a team pool directory or corpus manifest: {resolved}")


def build_team_source(
    path: str | Path,
    *,
    expected_format_id: str | None = None,
) -> TeamSource:
    """Build a corpus source when a pool manifest exists, otherwise a file source."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Team source path not found: {resolved}")

    if resolved.is_dir():
        manifest_path = resolved / CORPUS_MANIFEST_NAME
        if manifest_path.is_file():
            return _build_corpus_source(manifest_path, expected_format_id)
        return FileTeamSource(resolved)

    if resolved.is_file():
        if resolved.name == CORPUS_MANIFEST_NAME:
            return _build_corpus_source(resolved, expected_format_id)
        return FileTeamSource.from_files((resolved,))

    raise ValueError(f"Unsupported team source path: {resolved}")


def _build_corpus_source(
    manifest_path: Path,
    expected_format_id: str | None,
) -> CorpusTeamSource:
    try:
        manifest = load_corpus_manifest(orjson.loads(manifest_path.read_bytes()))
    except (OSError, UnicodeError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid corpus manifest: {manifest_path}") from exc

    if expected_format_id is not None and not is_corpus_format_compatible(
        expected_format_id, manifest.format_id
    ):
        raise ValueError(
            f"Corpus format mismatch: manifest={manifest.format_id!r}, "
            f"expected={expected_format_id!r}"
        )

    return CorpusTeamSource(manifest, corpus_path=str(manifest_path))
