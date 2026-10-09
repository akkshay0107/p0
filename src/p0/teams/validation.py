"""Offline admission through the pinned Showdown validator."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import orjson

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS
from p0.teams.stat_points import STAT_NAMES
from p0.teams.team import TeamRecord


class AdmissionResult(NamedTuple):
    team_key: str
    valid: bool
    packed_team: str | None
    problems: tuple[str, ...]


def _variant_dict(
    variant: TeamRecord,
    *,
    format_id: str = FORMAT.battle_format,
) -> dict[str, Any]:
    pairs = [
        (member.canonical(), spread)
        for member, spread in zip(variant.team.members, variant.spreads, strict=True)
    ]
    pairs.sort(key=lambda pair: pair[0].species)

    team = []
    for member, spread in pairs:
        team.append(
            {
                # An inferred canonical species ID is not a nickname. Sending it
                # as name makes Showdown reject long form IDs as nicknames.
                "name": "",
                "species": member.species,
                "item": member.item,
                "ability": member.ability,
                "moves": list(member.moves),
                "nature": member.nature,
                "evs": spread.as_dict(),
                "ivs": dict.fromkeys(STAT_NAMES, 31),
                "gender": member.gender,
                "level": member.level,
            }
        )

    return {"format": format_id, "team": team}


def showdown_payload(
    variant: TeamRecord,
    *,
    format_id: str = FORMAT.battle_format,
) -> str:
    return orjson.dumps(_variant_dict(variant, format_id=format_id)).decode("utf-8")


def validate_many_batched(
    variants: Sequence[TeamRecord],
    *,
    batch_size: int = 256,
    timeout: float = 60.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
    format_id: str = FORMAT.battle_format,
) -> tuple[AdmissionResult, ...]:
    """
    Validate multiple team variants using batched Node invocations.

    Arguments:
      variants: Sequence of team variants to validate against Champions rules.
      batch_size: Maximum number of variants sent per Node subprocess call.
      timeout: Maximum execution duration allowed per batch subprocess.
      repository_root: Root path where validation scripts are located.
      format_id: Target battle format ID.

    Returns:
      A tuple of admission results aligned exactly with input variants.
    """
    if not variants:
        return ()
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validator = repository_root / "scripts" / "validate_champions_batch.js"
    results: list[AdmissionResult] = []
    for offset in range(0, len(variants), batch_size):
        chunk = variants[offset : offset + batch_size]
        payload = orjson.dumps(
            [_variant_dict(variant, format_id=format_id) for variant in chunk]
        ).decode("utf-8")
        try:
            process = subprocess.run(
                ["node", str(validator)],
                input=payload,
                text=True,
                capture_output=True,
                cwd=repository_root,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Pinned Showdown batched validator timed out after {timeout:g}s"
            ) from exc
        if process.returncode:
            raise RuntimeError(
                f"Pinned Showdown batched validator failed: {process.stderr.strip()}"
            )
        try:
            parsed = orjson.loads(process.stdout)
            if not isinstance(parsed, list) or len(parsed) != len(chunk):
                raise ValueError("Response count mismatch")
            for variant, item in zip(chunk, parsed, strict=True):
                results.append(
                    AdmissionResult(
                        team_key=variant.team.team_key,
                        valid=bool(item["valid"]),
                        packed_team=item["packedTeam"],
                        problems=tuple(item["problems"]),
                    )
                )
        except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
            raise RuntimeError(
                "Pinned Showdown batched validator returned a malformed response"
            ) from exc
    return tuple(results)


# The default corpus validator; every validation goes through the batched path.
validate_many = validate_many_batched


def validate_variant(
    variant: TeamRecord,
    *,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
    format_id: str = FORMAT.battle_format,
) -> AdmissionResult:
    return validate_many(
        (variant,),
        timeout=timeout,
        repository_root=repository_root,
        format_id=format_id,
    )[0]
