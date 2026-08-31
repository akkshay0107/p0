"""Offline admission through the pinned Showdown validator."""

from __future__ import annotations

import selectors
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from queue import Empty, Queue
from typing import Any, NamedTuple

import orjson

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS
from p0.teams.team import TeamRecord


class AdmissionResult(NamedTuple):
    team_hash: str
    valid: bool
    packed_team: str | None
    problems: tuple[str, ...]


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _variant_dict(
    variant: TeamRecord,
    *,
    format_id: str = FORMAT.battle_format,
) -> dict[str, Any]:
    canonical_members = variant.team.canonical().members
    if variant.team.members == canonical_members:
        pairs = list(zip(variant.team.members, variant.spreads, strict=True))
    else:
        pairs = sorted(
            zip(variant.team.members, variant.spreads, strict=True),
            key=lambda pair: pair[0].canonical().species,
        )

    team = []
    for member, spread in pairs:
        canon_member = member.canonical()
        team.append(
            {
                # An inferred canonical species ID is not a nickname. Sending it
                # as name makes Showdown reject long form IDs as nicknames.
                "name": "",
                "species": canon_member.species,
                "item": canon_member.item,
                "ability": canon_member.ability,
                "moves": list(canon_member.moves),
                "nature": canon_member.nature,
                "evs": spread.as_dict(),
                "ivs": {name: 31 for name in ("hp", "atk", "def", "spa", "spd", "spe")},
                "gender": canon_member.gender,
                "level": canon_member.level,
            }
        )

    return {"format": format_id, "team": team}


def showdown_payload(
    variant: TeamRecord,
    *,
    format_id: str = FORMAT.battle_format,
) -> str:
    return orjson.dumps(_variant_dict(variant, format_id=format_id)).decode("utf-8")


def validate_variant(
    variant: TeamRecord,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
    format_id: str = FORMAT.battle_format,
) -> AdmissionResult:
    validator = repository_root / "scripts" / "validate_champions_team.js"
    try:
        process = runner(
            ["node", str(validator)],
            input=showdown_payload(variant, format_id=format_id),
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
        result = orjson.loads(process.stdout)
        return AdmissionResult(
            team_hash=variant.team.team_hash,
            valid=bool(result["valid"]),
            packed_team=result["packedTeam"],
            problems=tuple(result["problems"]),
        )
    except (KeyError, TypeError, orjson.JSONDecodeError) as exc:
        raise RuntimeError("Pinned Showdown validator returned a malformed response") from exc


def validate_many_batched(
    variants: Sequence[TeamRecord],
    *,
    batch_size: int = 256,
    runner: Runner = subprocess.run,
    timeout: float = 60.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
    format_id: str = FORMAT.battle_format,
) -> tuple[AdmissionResult, ...]:
    """
    Validate multiple team variants using batched Node invocations.

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
        payload = orjson.dumps(
            [_variant_dict(variant, format_id=format_id) for variant in chunk]
        ).decode("utf-8")
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
            parsed = orjson.loads(process.stdout)
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
        except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
            raise RuntimeError(
                "Pinned Showdown batched validator returned a malformed response"
            ) from exc
    return tuple(results)


def validate_many(
    variants: Sequence[TeamRecord],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 30.0,
    repository_root: Path = DEFAULT_PATHS.repository_root,
    format_id: str = FORMAT.battle_format,
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
                format_id=format_id,
            ),
        )
    return validate_many_batched(
        variants,
        runner=runner,
        timeout=timeout,
        repository_root=repository_root,
        format_id=format_id,
    )


class PersistentShowdownValidator:
    """
    Persistent Node subprocess context manager for continuous validation.

    Spawns a long-lived Node worker over stdio to validate large streams of
    teams without per-batch startup overhead.
    """

    def __init__(
        self,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        repository_root: Path = DEFAULT_PATHS.repository_root,
        request_timeout: float = 30.0,
        format_id: str = FORMAT.battle_format,
    ) -> None:
        self._popen_factory = popen_factory
        self._repository_root = repository_root
        self._process: Any = None
        self._request_timeout = request_timeout
        self._format_id = format_id

    def __enter__(self) -> PersistentShowdownValidator:
        validator = self._repository_root / "scripts" / "validate_champions_batch.js"
        self._process = self._popen_factory(
            ["node", str(validator), "--persistent"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Diagnostics are intentionally discarded here: the persistent
            # protocol is line-oriented and an undrained stderr pipe can block
            # the validator before it produces a response.
            stderr=subprocess.DEVNULL,
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
                    self._process.stdin.write(
                        orjson.dumps({"command": "stop"}).decode("utf-8") + "\n"
                    )
                    self._process.stdin.flush()
                    self._process.stdin.close()
                self._process.wait(timeout=2.0)
            except (BrokenPipeError, OSError, ValueError, subprocess.TimeoutExpired):
                try:
                    self._process.terminate()
                    self._process.wait(timeout=1.0)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    try:
                        self._process.kill()
                    except (OSError, ValueError):
                        pass
                    try:
                        self._process.wait(timeout=1.0)
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
            finally:
                if self._process is not None:
                    for stream in (self._process.stdout, self._process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except (OSError, RuntimeError, ValueError):
                                pass
                self._process = None

    def _readline(self) -> str:
        """Read one worker response with a bounded wait."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Persistent worker stdout is unavailable")
        stream = self._process.stdout
        try:
            selector = selectors.DefaultSelector()
            try:
                selector.register(stream, selectors.EVENT_READ)
                ready = selector.select(self._request_timeout)
            finally:
                selector.close()
            if not ready:
                raise TimeoutError(
                    f"Persistent validator response timed out after {self._request_timeout:g}s"
                )
            return stream.readline()
        except (AttributeError, OSError, TypeError, ValueError):
            # Test doubles and non-file streams may not expose a selectable fd.
            result: Queue[str | BaseException] = Queue(maxsize=1)

            def read() -> None:
                try:
                    result.put(stream.readline())
                # Preserve worker failures, including interpreter-level errors,
                # so the caller receives them instead of hanging on the queue.
                except BaseException as exc:
                    result.put(exc)

            threading.Thread(target=read, daemon=True).start()
            try:
                value = result.get(timeout=self._request_timeout)
            except Empty as exc:
                raise TimeoutError(
                    f"Persistent validator response timed out after {self._request_timeout:g}s"
                ) from exc
            if isinstance(value, BaseException):
                raise value
            return value

    def validate_many(
        self,
        variants: Sequence[TeamRecord],
        *,
        batch_size: int = 256,
    ) -> tuple[AdmissionResult, ...]:
        """
        Validate variants through the active persistent worker.

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
            poll = getattr(self._process, "poll", lambda: None)
            if poll() is not None:
                raise RuntimeError("Persistent validator exited before completing the request")
            chunk = variants[offset : offset + batch_size]
            payload = orjson.dumps(
                {"batch": [_variant_dict(variant, format_id=self._format_id) for variant in chunk]}
            ).decode("utf-8")
            try:
                self._process.stdin.write(payload + "\n")
                self._process.stdin.flush()
                line = self._readline()
                if not line:
                    raise RuntimeError("Persistent worker closed stdout unexpectedly")
                parsed = orjson.loads(line)
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
            except (
                BrokenPipeError,
                OSError,
                TimeoutError,
                TypeError,
                ValueError,
                orjson.JSONDecodeError,
                RuntimeError,
            ) as exc:
                raise RuntimeError(f"PersistentShowdownValidator failed: {exc}") from exc
        return tuple(results)
