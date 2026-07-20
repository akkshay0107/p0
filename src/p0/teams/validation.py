"""Offline admission through the pinned Showdown validator."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _variant_dict(variant: TeamVariant) -> dict[str, Any]:
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
    return {"format": FORMAT.battle_format, "team": team}


def showdown_payload(variant: TeamVariant) -> str:
    return json.dumps(_variant_dict(variant))


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


def validate_many_batched(
    variants: Sequence[TeamVariant],
    *,
    batch_size: int = 256,
    runner: Runner = subprocess.run,
    timeout: float = 60.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
) -> tuple[AdmissionResult, ...]:
    """Validate multiple team variants using batched Node invocations.

    Arguments:
      variants: Sequence of team variants to validate against Champions rules.
      batch_size: Maximum number of variants sent per Node subprocess call.
      runner: Subprocess invocation callable used to spawn Node.
      timeout: Maximum execution duration allowed per batch subprocess.
      repository_root: Root path where validation scripts are located.

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
        payload = json.dumps([_variant_dict(variant) for variant in chunk])
        try:
            process = runner(
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
            parsed = json.loads(process.stdout)
            if not isinstance(parsed, list) or len(parsed) != len(chunk):
                raise ValueError("Response count mismatch")
            for variant, item in zip(chunk, parsed, strict=True):
                results.append(
                    AdmissionResult(
                        team_hash=variant.team.team_hash,
                        valid=bool(item["valid"]),
                        packed_team=item["packedTeam"],
                        problems=tuple(item["problems"]),
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Pinned Showdown batched validator returned a malformed response"
            ) from exc
    return tuple(results)


def validate_many(
    variants: Sequence[TeamVariant],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
) -> tuple[AdmissionResult, ...]:
    if not variants:
        return ()
    if len(variants) == 1:
        return (
            validate_variant(
                variants[0],
                runner=runner,
                timeout=timeout,
                repository_root=repository_root,
            ),
        )
    return validate_many_batched(
        variants,
        runner=runner,
        timeout=timeout,
        repository_root=repository_root,
    )


class PersistentShowdownValidator:
    """Persistent Node subprocess context manager for continuous validation.

    Spawns a long-lived Node worker over stdio to validate large streams of
    teams without per-batch startup overhead.
    """

    def __init__(
        self,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        repository_root: Path = DEFAULT_PATHS.repository_root,
    ) -> None:
        self._popen_factory = popen_factory
        self._repository_root = repository_root
        self._process: Any = None

    def __enter__(self) -> PersistentShowdownValidator:
        validator = self._repository_root / "scripts" / "validate_champions_batch.js"
        self._process = self._popen_factory(
            ["node", str(validator), "--persistent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._repository_root,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                    self._process.stdin.flush()
                    self._process.stdin.close()
                self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.terminate()
                except Exception:
                    pass
            finally:
                if self._process is not None:
                    for stream in (self._process.stdout, self._process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except Exception:
                                pass
                self._process = None

    def validate_many(
        self,
        variants: Sequence[TeamVariant],
        *,
        batch_size: int = 256,
    ) -> tuple[AdmissionResult, ...]:
        """Validate variants through the active persistent worker.

        Arguments:
          variants: Sequence of team variants to validate.
          batch_size: Maximum number of variants sent per stdio batch request.

        Returns:
          A tuple of admission results aligned exactly with input variants.
        """
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("PersistentShowdownValidator process is not open")
        if not variants:
            return ()
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        results: list[AdmissionResult] = []
        for offset in range(0, len(variants), batch_size):
            chunk = variants[offset : offset + batch_size]
            payload = json.dumps({"batch": [_variant_dict(variant) for variant in chunk]})
            try:
                self._process.stdin.write(payload + "\n")
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("Persistent worker closed stdout unexpectedly")
                parsed = json.loads(line)
                if parsed.get("status") != "ok":
                    raise RuntimeError(f"Persistent worker error: {parsed.get('message')}")
                items = parsed.get("results")
                if not isinstance(items, list) or len(items) != len(chunk):
                    raise ValueError("Response count mismatch")
                for variant, item in zip(chunk, items, strict=True):
                    results.append(
                        AdmissionResult(
                            team_hash=variant.team.team_hash,
                            valid=bool(item["valid"]),
                            packed_team=item["packedTeam"],
                            problems=tuple(item["problems"]),
                        )
                    )
            except Exception as exc:
                raise RuntimeError(f"PersistentShowdownValidator failed: {exc}") from exc
        return tuple(results)
