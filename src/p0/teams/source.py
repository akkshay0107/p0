"""Prepared team sources used by runtime composition."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from poke_env.teambuilder import Teambuilder

JsonScalar = str | int | float | bool | None


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
        return self._teams[rng.randrange(len(self._teams))]

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
