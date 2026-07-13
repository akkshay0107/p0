"""Durable canonical Champions team records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable, Mapping, cast

from p0.format_config import FORMAT, STAT_POINT_IMPUTER_VERSION
from p0.teams.stat_points import StatPoints


def normalize_id(value: str) -> str:
    """Normalize a display identifier for canonical team identity."""
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

    def to_dict(self) -> dict[str, object]:
        member = self.canonical()
        return {
            "species": member.species,
            "item": member.item,
            "ability": member.ability,
            "moves": list(member.moves),
            "nature": member.nature,
            "gender": member.gender,
            "level": member.level,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamMember:
        required = {"species", "item", "ability", "moves", "nature", "gender", "level"}
        if set(value) != required or not isinstance(value["moves"], list):
            raise ValueError("Invalid serialized team member")
        return cls(
            species=str(value["species"]),
            item=str(value["item"]),
            ability=str(value["ability"]),
            moves=tuple(str(move) for move in value["moves"]),
            nature=str(value["nature"]),
            gender=str(value["gender"]),
            level=int(cast(int, value["level"])),
        )


@dataclass(frozen=True, slots=True)
class CanonicalTeam:
    members: tuple[TeamMember, ...]

    def __post_init__(self) -> None:
        if len(self.members) != 6:
            raise ValueError("Champions teams must contain exactly six members")

    def canonical(self) -> CanonicalTeam:
        return CanonicalTeam(
            tuple(
                sorted(
                    (member.canonical() for member in self.members), key=lambda item: item.species
                )
            )
        )

    @property
    def team_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"members": [member.to_dict() for member in self.canonical().members]}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CanonicalTeam:
        if set(value) != {"members"} or not isinstance(value["members"], list):
            raise ValueError("Invalid serialized canonical team")
        return cls(tuple(TeamMember.from_dict(member) for member in value["members"])).canonical()


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

    def to_dict(self) -> dict[str, object]:
        return {
            "source_series": list(self.source_series),
            "source_replays": list(self.source_replays),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "usage_count": self.usage_count,
            "archetype_tags": list(self.archetype_tags),
            "evidence_game": self.evidence_game,
            "evidence_event": self.evidence_event,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamMetadata:
        expected = {
            "source_series",
            "source_replays",
            "first_seen",
            "last_seen",
            "usage_count",
            "archetype_tags",
            "evidence_game",
            "evidence_event",
        }
        if set(value) != expected:
            raise ValueError("Invalid serialized team metadata fields")
        try:
            return cls(
                source_series=tuple(
                    str(item) for item in cast(list[object], value["source_series"])
                ),
                source_replays=tuple(
                    str(item) for item in cast(list[object], value["source_replays"])
                ),
                first_seen=str(value["first_seen"]),
                last_seen=str(value["last_seen"]),
                usage_count=int(cast(int, value["usage_count"])),
                archetype_tags=tuple(
                    str(item) for item in cast(list[object], value["archetype_tags"])
                ),
                evidence_game=int(cast(int, value["evidence_game"])),
                evidence_event=int(cast(int, value["evidence_event"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid serialized team metadata") from exc


@dataclass(frozen=True, slots=True)
class TeamVariant:
    team: CanonicalTeam
    spreads: tuple[StatPoints, ...]
    metadata: TeamMetadata
    spread_provenance: str = "imputed"
    imputer_version: int = STAT_POINT_IMPUTER_VERSION
    validator_version: str = FORMAT.showdown_commit

    def __post_init__(self) -> None:
        if len(self.spreads) != len(self.team.members):
            raise ValueError("Each team member requires one Stat Point spread")
        if self.spread_provenance != "imputed":
            raise ValueError("Reconstructed public teams must retain imputed provenance")

    def to_dict(self) -> dict[str, object]:
        pairs = sorted(
            zip(self.team.members, self.spreads, strict=True),
            key=lambda pair: pair[0].canonical().species,
        )
        return {
            "team": CanonicalTeam(tuple(member for member, _ in pairs)).to_dict(),
            "spreads": [spread.as_dict() for _, spread in pairs],
            "metadata": self.metadata.to_dict(),
            "spread_provenance": self.spread_provenance,
            "imputer_version": self.imputer_version,
            "validator_version": self.validator_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TeamVariant:
        expected = {
            "team",
            "spreads",
            "metadata",
            "spread_provenance",
            "imputer_version",
            "validator_version",
        }
        if set(value) != expected:
            raise ValueError("Invalid serialized team variant fields")
        try:
            spreads = tuple(
                StatPoints(
                    hp=int(cast(dict[str, int], spread)["hp"]),
                    atk=int(cast(dict[str, int], spread)["atk"]),
                    defense=int(cast(dict[str, int], spread)["def"]),
                    spa=int(cast(dict[str, int], spread)["spa"]),
                    spd=int(cast(dict[str, int], spread)["spd"]),
                    spe=int(cast(dict[str, int], spread)["spe"]),
                )
                for spread in cast(list[object], value["spreads"])
            )
            return cls(
                team=CanonicalTeam.from_dict(cast(Mapping[str, object], value["team"])),
                spreads=spreads,
                metadata=TeamMetadata.from_dict(cast(Mapping[str, object], value["metadata"])),
                spread_provenance=str(value["spread_provenance"]),
                imputer_version=int(cast(int, value["imputer_version"])),
                validator_version=str(value["validator_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid serialized team variant") from exc


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
    *, own_team: bool, game_number: int, event_index: int, evidence_game: int, evidence_event: int
) -> None:
    if min(event_index, evidence_game, evidence_event) < 0 or game_number < 1:
        raise ValueError("Game numbers must be positive and evidence cutoffs nonnegative")
    if not own_team and (evidence_game, evidence_event) > (game_number, event_index):
        raise ValueError("Opponent Stat Point evidence cannot come from the future")
