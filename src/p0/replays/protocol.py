"""Parsing of public Showdown replay payloads into ordered pure records.

This module understands transport JSON and the line-oriented Showdown protocol,
but does not simulate a battle. Keeping parsing here makes reconstruction
reproducible from the immutable response bytes and gives malformed logs an
explicit error instead of silently dropping lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from p0.replays.schema import (
    GameEndReason,
    OTSData,
    ProtocolLine,
    ReplayMetadata,
    ReplayOutcome,
)


class ReplayParseError(ValueError):
    """Raised when a response is not a supported public replay payload."""


@dataclass(frozen=True, slots=True)
class ReplayDocument:
    """Normalized replay input consumed by grouping and reconstruction."""

    metadata: ReplayMetadata
    protocol_lines: tuple[ProtocolLine, ...]
    ots: tuple[OTSData, OTSData]
    outcome: ReplayOutcome
    raw_payload: bytes

    def __post_init__(self) -> None:
        for expected, line in enumerate(self.protocol_lines):
            if line.index != expected:
                raise ValueError("ReplayDocument protocol lines must be contiguous and ordered")
        if tuple(ots.player for ots in self.ots) != ("p1", "p2"):
            raise ValueError("ReplayDocument.ots must be ordered as p1 and p2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "protocol_lines": [line.to_dict() for line in self.protocol_lines],
            "ots": [ots.to_dict() for ots in self.ots],
            "outcome": self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayDocument:
        expected = {"metadata", "protocol_lines", "ots", "outcome"}
        if set(value) != expected:
            raise ValueError("Invalid ReplayDocument fields")
        ots = tuple(OTSData.from_dict(item) for item in value["ots"])
        if len(ots) != 2:
            raise ValueError("ReplayDocument.ots must contain both players")
        return cls(
            metadata=ReplayMetadata.from_dict(value["metadata"]),
            protocol_lines=tuple(ProtocolLine.from_dict(item) for item in value["protocol_lines"]),
            ots=(ots[0], ots[1]),
            outcome=ReplayOutcome.from_dict(value["outcome"]),
            raw_payload=b"",
        )


def _as_object(payload: bytes | str | Mapping[str, Any]) -> tuple[Mapping[str, Any], bytes]:
    if isinstance(payload, Mapping):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return payload, encoded
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayParseError("Replay response is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ReplayParseError("Replay response root must be a JSON object")
    return value, raw


def _timestamp(value: Any) -> str:
    if isinstance(value, (int, float)) and value >= 0:
        return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except ValueError as exc:
            raise ReplayParseError(f"Invalid upload timestamp {value!r}") from exc
    return "1970-01-01T00:00:00Z"


def _players(value: Mapping[str, Any]) -> tuple[str, str]:
    p1 = value.get("p1")
    p2 = value.get("p2")
    if isinstance(p1, str) and isinstance(p2, str) and p1 and p2:
        return p1, p2
    players = value.get("players")
    if (
        isinstance(players, list)
        and len(players) == 2
        and all(isinstance(item, str) for item in players)
    ):
        return players[0], players[1]
    raise ReplayParseError("Replay metadata must contain p1 and p2 players")


def _metadata(
    value: Mapping[str, Any], requested_id: str | None, format_id: str | None
) -> ReplayMetadata:
    replay_id = value.get("id", requested_id)
    if not isinstance(replay_id, str) or not replay_id:
        raise ReplayParseError("Replay metadata has no replay id")
    actual_format = value.get("format", format_id)
    if not isinstance(actual_format, str) or not actual_format:
        raise ReplayParseError("Replay metadata has no format id")
    room_id = str(value.get("roomid", value.get("room_id", replay_id)))
    parent = value.get("parent", value.get("parentid", value.get("parent_room", "")))
    winner = value.get("winner", "")
    if winner is None:
        winner = ""
    rating = value.get("rating")
    views = value.get("views")
    return ReplayMetadata(
        replay_id=replay_id,
        format_id=actual_format,
        player_names=_players(value),
        winner=str(winner),
        upload_time=_timestamp(value.get("uploadtime", value.get("upload_time"))),
        room_id=room_id,
        parent_room=str(parent),
        game_number=(None if value.get("game_number") is None else int(value["game_number"])),
        rating=(None if rating is None else int(rating)),
        views=(None if views is None else int(views)),
    )


def _protocol_lines(log: Any) -> tuple[ProtocolLine, ...]:
    if isinstance(log, list):
        if not all(isinstance(line, str) for line in log):
            raise ReplayParseError("Replay log arrays must contain strings")
        lines = log
    elif isinstance(log, str):
        lines = log.splitlines()
    else:
        raise ReplayParseError("Replay metadata has no string or array log")
    result: list[ProtocolLine] = []
    turn: int | None = None
    for index, line in enumerate(lines):
        if line == "":
            continue
        if not line.startswith("|"):
            raise ReplayParseError(f"Malformed protocol line {index}: {line!r}")
        parts = tuple(line.split("|"))
        if len(parts) < 2 or not parts[1]:
            raise ReplayParseError(f"Malformed protocol line {index}: {line!r}")
        if parts[1] == "turn":
            if len(parts) < 3 or not parts[2].isdigit():
                raise ReplayParseError(f"Invalid turn line {index}: {line!r}")
            turn = int(parts[2])
        result.append(ProtocolLine(len(result), line, parts, turn))
    if not result:
        raise ReplayParseError("Replay log contains no protocol lines")
    return tuple(result)


def _details_from_payload(payload: str) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = None
    details: dict[str, dict[str, Any]] = {}
    if isinstance(value, list):
        entries = value
    elif isinstance(value, Mapping):
        entries = value.get("team", value.get("pokemon", ()))
    else:
        entries = ()
        for packed_set in payload.split("]"):
            fields = packed_set.split("|")
            if len(fields) < 5 or not fields[0]:
                continue
            species = fields[1] or fields[0]
            details[species] = {
                "name": fields[0],
                "species": species,
                "item": fields[2],
                "ability": fields[3],
                "moves": tuple(move for move in fields[4].split(",") if move),
                "nature": fields[5] if len(fields) > 5 else "",
                "evs": fields[6] if len(fields) > 6 else "",
                "level": fields[10] if len(fields) > 10 and fields[10] else 100,
            }
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                name = entry.get("species", entry.get("name"))
                if isinstance(name, str) and name:
                    species = name.split(",", 1)[0].strip()
                    entry_details = dict(entry)
                    details[species] = entry_details
            elif isinstance(entry, str) and entry:
                species = entry.split(",", 1)[0].strip()
                details[species] = {"raw": entry}
    return details


def _ots(lines: tuple[ProtocolLine, ...]) -> tuple[OTSData, OTSData]:
    payloads = {"p1": [], "p2": []}
    species: dict[str, list[str]] = {"p1": [], "p2": []}
    details: dict[str, dict[str, Mapping[str, Any]]] = {"p1": {}, "p2": {}}
    for line in lines:
        if len(line.parts) < 3:
            continue
        tag = line.parts[1]
        if tag == "showteam" and len(line.parts) >= 4 and line.parts[2] in payloads:
            payload = "|".join(line.parts[3:])
            payloads[line.parts[2]].append(payload)
            payload_details = _details_from_payload(payload)
            species[line.parts[2]].extend(payload_details)
            details[line.parts[2]].update(payload_details)
        elif tag == "poke" and line.parts[2] in species and len(line.parts) >= 4:
            species[line.parts[2]].append(line.parts[3].split(",", 1)[0].strip())
    result = []
    for player in ("p1", "p2"):
        raw = "\n".join(payloads[player])
        result.append(
            OTSData(
                player=player,
                raw_payload=raw,
                revealed_species=tuple(dict.fromkeys(item for item in species[player] if item)),
                revealed_details=details[player],
            )
        )
    return result[0], result[1]


def _outcome(metadata: ReplayMetadata, lines: tuple[ProtocolLine, ...]) -> ReplayOutcome:
    winner = -1
    end_reason = GameEndReason.NORMAL
    terminal: int | None = None
    players = tuple(name.casefold() for name in metadata.player_names)
    for line in lines:
        if len(line.parts) < 3:
            continue
        tag = line.parts[1]
        if tag == "win":
            terminal = line.index
            winner_name = line.parts[2].strip().casefold()
            if winner_name in players:
                winner = players.index(winner_name)
        elif tag == "tie":
            terminal = line.index
        elif tag == "forfeit":
            terminal = line.index
            end_reason = GameEndReason.FORFEIT
        elif tag == "message" and "timeout" in line.parts[-1].casefold():
            end_reason = GameEndReason.TIMEOUT
            terminal = line.index
    turns = max((line.turn or 0 for line in lines), default=0)
    return ReplayOutcome(winner, end_reason, turns, terminal)


def _bestof_metadata(metadata: ReplayMetadata, lines: tuple[ProtocolLine, ...]) -> ReplayMetadata:
    parent = metadata.parent_room
    game_number = metadata.game_number
    for line in lines:
        if len(line.parts) < 4 or line.parts[1] not in {"uhtml", "uhtmlchange"}:
            continue
        if line.parts[2] != "bestof":
            continue
        html = "|".join(line.parts[3:])
        game_match = re.search(r"Game\s+(\d+)", html, flags=re.IGNORECASE)
        if game_match:
            game_number = int(game_match.group(1))
        href_match = re.search(r'href=["\']?/([^"\'>]+)', html)
        if href_match:
            candidate = href_match.group(1)
            parent = candidate.removeprefix("battle-")
    if parent == metadata.parent_room and game_number == metadata.game_number:
        return metadata
    return ReplayMetadata(
        replay_id=metadata.replay_id,
        format_id=metadata.format_id,
        player_names=metadata.player_names,
        winner=metadata.winner,
        upload_time=metadata.upload_time,
        room_id=metadata.room_id,
        parent_room=parent,
        game_number=game_number,
        rating=metadata.rating,
        views=metadata.views,
    )


def parse_replay_payload(
    payload: bytes | str | Mapping[str, Any],
    *,
    replay_id: str | None = None,
    format_id: str | None = None,
) -> ReplayDocument:
    """Parse one public replay response without applying future protocol lines."""
    value, raw = _as_object(payload)
    metadata = _metadata(value, replay_id, format_id)
    lines = _protocol_lines(value.get("log"))
    metadata = _bestof_metadata(metadata, lines)
    return ReplayDocument(metadata, lines, _ots(lines), _outcome(metadata, lines), raw)


parse_replay = parse_replay_payload
parse_protocol = _protocol_lines


__all__ = [
    "ReplayDocument",
    "ReplayParseError",
    "parse_protocol",
    "parse_replay",
    "parse_replay_payload",
]
