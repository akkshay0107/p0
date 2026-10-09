"""The built team corpus that live players sample their teams from."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from poke_env.teambuilder import Teambuilder

from p0.contracts import require_dataclass_fields
from p0.format_config import is_corpus_format_compatible

CORPUS_MANIFEST_NAME = "corpus_manifest.json"
TEAM_SIZE = 6


class _ShowdownTeambuilder(Teambuilder):
    def yield_team(self) -> str:
        raise RuntimeError("The parsing helper is not a runtime team builder")


SHOWDOWN_TEAMBUILDER = _ShowdownTeambuilder()


@dataclass(frozen=True, slots=True)
class TeamCorpus:
    """Packed teams for one format; each group holds the variants of one canonical team."""

    format_id: str
    teams: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.format_id:
            raise ValueError("TeamCorpus.format_id must be non-empty")

        if not self.teams or not all(group and all(group) for group in self.teams):
            raise ValueError("A team corpus requires at least one team and no empty entries")

    def sample(self, rng: random.Random) -> str:
        """Return one packed team, uniform over canonical teams and then over variants."""
        return rng.choice(rng.choice(self.teams))

    def to_dict(self) -> dict[str, Any]:
        return {"format_id": self.format_id, "teams": [list(group) for group in self.teams]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TeamCorpus:
        require_dataclass_fields(value, cls)
        return cls(
            format_id=str(value["format_id"]),
            teams=tuple(tuple(str(team) for team in group) for group in value["teams"]),
        )


def pack_showdown_team(text: str) -> str:
    """Convert one six-member Showdown export into the packed team format."""
    try:
        members = SHOWDOWN_TEAMBUILDER.parse_showdown_team(text)
        if len(members) != TEAM_SIZE:
            raise ValueError(f"Expected exactly {TEAM_SIZE} team members")
        packed = SHOWDOWN_TEAMBUILDER.join_team(members)
    # The third-party teambuilder exposes several parser exception types;
    # normalize all of them at this boundary.
    except Exception as exc:
        raise ValueError("Malformed Showdown team") from exc

    if not packed:
        raise ValueError("Malformed Showdown team")
    return packed


def corpus_from_team_files(paths: Sequence[str | Path], format_id: str) -> TeamCorpus:
    """Build a corpus from export files of one team each, without Showdown validation."""
    teams: list[tuple[str, ...]] = []
    for path in paths:
        try:
            teams.append((pack_showdown_team(Path(path).read_text(encoding="utf-8")),))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Malformed team file: {path}") from exc

    return TeamCorpus(format_id=format_id, teams=tuple(teams))


def load_team_corpus(directory: str | Path, expected_format_id: str) -> TeamCorpus:
    """Load the built corpus of a team pool directory and check its format."""
    manifest_path = Path(directory) / CORPUS_MANIFEST_NAME
    build_command = f"`p0-corpus build --input {directory}`"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No team corpus at {manifest_path}; build it with {build_command}")

    try:
        corpus = TeamCorpus.from_dict(orjson.loads(manifest_path.read_bytes()))
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid corpus manifest {manifest_path}; rebuild it with {build_command}: {exc}"
        ) from exc

    if not is_corpus_format_compatible(expected_format_id, corpus.format_id):
        raise ValueError(
            f"Corpus format mismatch: manifest={corpus.format_id!r}, "
            f"expected={expected_format_id!r}"
        )
    return corpus
