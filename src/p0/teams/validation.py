"""Offline admission through the pinned Showdown validator."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS
from p0.teams.team import TeamVariant


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    team_hash: str
    valid: bool
    packed_team: str | None
    problems: tuple[str, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def showdown_payload(variant: TeamVariant) -> str:
    pairs = sorted(
        zip(variant.team.members, variant.spreads, strict=True),
        key=lambda pair: pair[0].canonical().species,
    )
    team = []
    for member, spread in pairs:
        member = member.canonical()
        team.append(
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
        )
    return json.dumps({"format": FORMAT.battle_format, "team": team})


def validate_variant(
    variant: TeamVariant,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
) -> AdmissionResult:
    validator = repository_root / "scripts" / "validate_champions_team.js"
    try:
        process = runner(
            ["node", str(validator)],
            input=showdown_payload(variant),
            text=True,
            capture_output=True,
            cwd=repository_root,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Pinned Showdown validator timed out after {timeout:g}s") from exc
    if process.returncode:
        raise RuntimeError(f"Pinned Showdown validator failed: {process.stderr.strip()}")
    try:
        result = json.loads(process.stdout)
        return AdmissionResult(
            team_hash=variant.team.team_hash,
            valid=bool(result["valid"]),
            packed_team=result["packedTeam"],
            problems=tuple(result["problems"]),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Pinned Showdown validator returned a malformed response") from exc


def validate_many(
    variants: Sequence[TeamVariant],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
) -> tuple[AdmissionResult, ...]:
    return tuple(
        validate_variant(
            variant,
            runner=runner,
            timeout=timeout,
            repository_root=repository_root,
        )
        for variant in variants
    )
