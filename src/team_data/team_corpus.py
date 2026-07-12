"""Canonical Champions teams and offline Showdown corpus admission."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.format_config import FORMAT, STAT_POINT_IMPUTER_VERSION
from src.team_data.stat_points import StatPoints

_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR = _ROOT / "scripts" / "validate_champions_team.js"
_SHOWDOWN_VERSION = FORMAT.showdown_commit


def normalize_id(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


@dataclass(frozen=True, slots=True)
class TeamMember:
    species: str
    item: str
    ability: str
    moves: tuple[str, ...]
    nature: str
    gender: str = ""
    level: int = 50

    def __post_init__(self) -> None:
        if not self.species or not self.nature:
            raise ValueError("Team members require species and nature")
        if not 1 <= len(self.moves) <= 4:
            raise ValueError("Team members require one to four moves")
        if not 1 <= self.level <= 100:
            raise ValueError("Team member level must be in [1, 100]")

    def canonical(self) -> TeamMember:
        return TeamMember(
            species=normalize_id(self.species),
            item=normalize_id(self.item),
            ability=normalize_id(self.ability),
            moves=tuple(sorted(normalize_id(move) for move in self.moves)),
            nature=normalize_id(self.nature),
            gender=self.gender.upper(),
            level=self.level,
        )

    def identity(self) -> dict[str, object]:
        member = self.canonical()
        return {
            "species": member.species,
            "item": member.item,
            "ability": member.ability,
            "moves": member.moves,
            "nature": member.nature,
            "gender": member.gender,
            "level": member.level,
        }


@dataclass(frozen=True, slots=True)
class CanonicalTeam:
    members: tuple[TeamMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) != 6:
            raise ValueError("Champions teams must contain exactly six members")

    def canonical(self) -> CanonicalTeam:
        members = (member.canonical() for member in self.members)
        return CanonicalTeam(tuple(sorted(members, key=lambda member: member.species)))

    @property
    def team_hash(self) -> str:
        payload = [member.identity() for member in self.canonical().members]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TeamMetadata:
    source_series: tuple[str, ...]
    source_replays: tuple[str, ...]
    first_seen: str
    last_seen: str
    usage_count: int = 1
    archetype_tags: tuple[str, ...] = ()
    evidence_game: int = 0
    evidence_event: int = 0

    def __post_init__(self) -> None:
        if self.usage_count < 1 or self.evidence_game < 0 or self.evidence_event < 0:
            raise ValueError("Usage count must be positive and evidence cutoff nonnegative")
        for timestamp in (self.first_seen, self.last_seen):
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@dataclass(frozen=True, slots=True)
class TeamVariant:
    team: CanonicalTeam
    spreads: tuple[StatPoints, ...]
    metadata: TeamMetadata
    spread_provenance: str = "imputed"
    imputer_version: int = STAT_POINT_IMPUTER_VERSION
    validator_version: str = _SHOWDOWN_VERSION

    def __post_init__(self) -> None:
        if len(self.spreads) != len(self.team.members):
            raise ValueError("Each team member requires one Stat Point spread")
        if self.spread_provenance != "imputed":
            raise ValueError("Reconstructed public teams must retain imputed provenance")


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    team_hash: str
    valid: bool
    packed_team: str | None
    problems: tuple[str, ...]


def _showdown_team(variant: TeamVariant) -> list[dict[str, object]]:
    pairs = sorted(
        (
            (member.canonical(), spread)
            for member, spread in zip(variant.team.members, variant.spreads, strict=True)
        ),
        key=lambda pair: pair[0].species,
    )
    return [
        {
            "name": member.species,
            "species": member.species,
            "item": member.item,
            "ability": member.ability,
            "moves": list(member.moves),
            "nature": member.nature,
            "evs": spread.as_dict(),
            "ivs": {name: 31 for name in ("hp", "atk", "def", "spa", "spd", "spe")},
            "gender": member.gender,
            "level": member.level,
        }
        for member, spread in pairs
    ]


def validate_variant(variant: TeamVariant) -> AdmissionResult:
    payload = json.dumps({"format": FORMAT.battle_format, "team": _showdown_team(variant)})
    process = subprocess.run(
        ["node", str(_VALIDATOR)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=_ROOT,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Pinned Showdown validator failed: {process.stderr.strip()}")
    result = json.loads(process.stdout)
    return AdmissionResult(
        team_hash=variant.team.team_hash,
        valid=bool(result["valid"]),
        packed_team=result["packedTeam"],
        problems=tuple(result["problems"]),
    )


def deduplicate_variants(variants: Iterable[TeamVariant]) -> tuple[TeamVariant, ...]:
    by_hash: dict[tuple[str, tuple[StatPoints, ...]], TeamVariant] = {}
    for variant in variants:
        canonical_spreads = tuple(
            spread
            for _, spread in sorted(
                zip(variant.team.members, variant.spreads, strict=True),
                key=lambda pair: pair[0].canonical().species,
            )
        )
        key = (variant.team.team_hash, canonical_spreads)
        previous = by_hash.get(key)
        if previous is None:
            by_hash[key] = variant
            continue
        metadata = TeamMetadata(
            source_series=tuple(
                sorted(set(previous.metadata.source_series + variant.metadata.source_series))
            ),
            source_replays=tuple(
                sorted(set(previous.metadata.source_replays + variant.metadata.source_replays))
            ),
            first_seen=min(previous.metadata.first_seen, variant.metadata.first_seen),
            last_seen=max(previous.metadata.last_seen, variant.metadata.last_seen),
            usage_count=previous.metadata.usage_count + variant.metadata.usage_count,
            archetype_tags=tuple(
                sorted(set(previous.metadata.archetype_tags + variant.metadata.archetype_tags))
            ),
            evidence_game=min(previous.metadata.evidence_game, variant.metadata.evidence_game),
            evidence_event=min(previous.metadata.evidence_event, variant.metadata.evidence_event),
        )
        by_hash[key] = replace(previous, metadata=metadata)
    return tuple(by_hash[key] for key in sorted(by_hash, key=lambda item: (item[0], repr(item[1]))))


def validate_evidence_cutoff(
    *,
    own_team: bool,
    game_number: int,
    event_index: int,
    evidence_game: int,
    evidence_event: int,
) -> None:
    """Reject opponent estimates that consume evidence from a future game."""
    if min(event_index, evidence_game, evidence_event) < 0 or game_number < 1:
        raise ValueError("Game numbers must be positive and evidence cutoffs nonnegative")
    if not own_team and (evidence_game, evidence_event) > (game_number, event_index):
        raise ValueError("Opponent Stat Point evidence cannot come from the future")
