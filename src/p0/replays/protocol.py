"""
Parsing of public Showdown replay payloads into ordered pure records.

This module understands transport JSON and the line-oriented Showdown protocol,
but does not simulate a battle. Keeping parsing here makes reconstruction
reproducible from the immutable response bytes and gives malformed logs an
explicit error instead of silently dropping lines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

import orjson

from p0.replays.identity import (
    ReplayMemberId,
    ReplaySide,
    canonical_format_id,
    normalize_showdown_id,
)
from p0.replays.schema import (
    GameEndReason,
    OTSData,
    OTSMember,
    ProtocolLine,
    ReplayMetadata,
    ReplayOutcome,
)


class ReplayParseError(ValueError):
    """Raised when a response is not a supported public replay payload."""

    category = "INVALID_INPUT_CONTRACT"


class ReplayInputContractError(ReplayParseError):
    """Raised when transport, metadata, OTS, or terminal input is unusable."""


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
        if tuple(ots.side for ots in self.ots) != (ReplaySide.P1, ReplaySide.P2):
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
        try:
            encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        except (TypeError, ValueError) as exc:
            raise ReplayInputContractError("Replay payload contains non-JSON metadata") from exc
        return payload, encoded

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        value = orjson.loads(raw)
    except (UnicodeDecodeError, orjson.JSONDecodeError) as exc:
        raise ReplayParseError("Replay response is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ReplayParseError("Replay response root must be a JSON object")
    return value, raw


def _timestamp(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")

    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except ValueError as exc:
            raise ReplayParseError(f"Invalid upload timestamp {value!r}") from exc

    logging.getLogger(__name__).warning("replay has no upload timestamp; using epoch fallback")
    return "1970-01-01T00:00:00Z"  # fallback


def _players(value: Mapping[str, Any]) -> tuple[str, str]:
    p1 = value.get("p1")
    p2 = value.get("p2")
    if isinstance(p1, str) and isinstance(p2, str) and p1 and p2:
        return p1, p2

    players = value.get("players")
    if (
        isinstance(players, list)
        and len(players) == 2
        and all(isinstance(item, str) and item.strip() for item in players)
    ):
        return players[0], players[1]

    raise ReplayParseError("Replay metadata must contain p1 and p2 players")


def _metadata(
    value: Mapping[str, Any], requested_id: str | None, format_id: str | None
) -> ReplayMetadata:
    replay_id = value.get("id", requested_id)
    if not isinstance(replay_id, str) or not replay_id:
        raise ReplayParseError("Replay metadata has no replay id")

    actual_format = canonical_format_id(value, expected=format_id)
    if actual_format is None:
        raise ReplayParseError("Replay metadata has no format id")

    room_value = value.get("roomid", value.get("room_id"))
    room_id = room_value if isinstance(room_value, str) and room_value else replay_id
    parent_value = value.get("parent", value.get("parentid", value.get("parent_room")))
    if parent_value is not None and not isinstance(parent_value, str):
        raise ReplayParseError("Replay parent room must be a string when present")

    parent_room = "" if parent_value is None else parent_value
    winner = value.get("winner", "")
    if winner is None:
        winner = ""
    if not isinstance(winner, str):
        raise ReplayParseError("Replay winner must be a string when present")

    def _optional_int(field: str) -> int | None:
        candidate = value.get(field)
        if candidate in (None, ""):
            return None
        if isinstance(candidate, bool):
            raise ReplayInputContractError(f"Replay metadata {field} must be an integer")
        try:
            return int(candidate)
        except (TypeError, ValueError) as exc:
            raise ReplayInputContractError(f"Replay metadata {field} must be an integer") from exc

    return ReplayMetadata(
        replay_id=replay_id,
        format_id=actual_format,
        player_names=_players(value),
        winner=winner,
        upload_time=_timestamp(value.get("uploadtime", value.get("upload_time"))),
        room_id=room_id,
        parent_room=parent_room,
        game_number=_optional_int("game_number"),
        rating=_optional_int("rating"),
        views=_optional_int("views"),
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
    skipping_chat_response = False
    for index, line in enumerate(lines):
        if line == "":
            continue

        if not line.startswith("|"):
            # The replay endpoint occasionally emits the room's plain MESSAGE
            # text as an unframed line after a chat record.  It carries no
            # battle state and must not make an otherwise valid replay fail.
            if skipping_chat_response:
                skipping_chat_response = False
                continue
            raise ReplayParseError(f"Malformed protocol line {index}: {line!r}")

        parts_list = line.split("|")
        if len(parts_list) >= 2 and parts_list[1] in {"c", "chat", "c:", "chatmsg"}:
            skipping_chat_response = True
            continue

        skipping_chat_response = False
        if len(parts_list) >= 2 and parts_list[1] == "turn":
            if len(parts_list) < 3 or not parts_list[2].isdigit():
                raise ReplayParseError(f"Invalid turn line {index}: {line!r}")
            turn = int(parts_list[2])
        result.append(ProtocolLine(len(result), line, tuple(parts_list), turn))

    if not result:
        raise ReplayParseError("Replay log contains no protocol lines")

    return tuple(result)


def _moves(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(move for move in value.split(",") if move)
    if isinstance(value, (list, tuple)) and all(isinstance(move, str) and move for move in value):
        return tuple(value)
    raise ReplayParseError("OTS moves must be a comma-separated string or string array")


def _level(value: Any) -> int:
    if value is None or value == "":
        return 50
    normalized = value.lstrip("L") if isinstance(value, str) else value
    try:
        level = int(normalized)
    except (TypeError, ValueError) as exc:
        raise ReplayParseError(f"Invalid OTS level {value!r}") from exc
    if not 1 <= level <= 100:
        raise ReplayParseError(f"OTS level must be in [1, 100], received {level}")
    return level


def _serialized_evs(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    raise ReplayParseError("OTS EVs must be a string, object, or array")


def _optional_ots_string(entry: Mapping[str, Any], field: str) -> str:
    value = entry.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReplayParseError(f"OTS {field} must be a string when present")
    return value


def _parse_mapping_ots_member(
    side: ReplaySide,
    roster_index: int,
    entry: Mapping[str, Any],
) -> OTSMember:
    species_value = entry.get("species", entry.get("name"))
    if not isinstance(species_value, str) or not species_value.strip():
        raise ReplayParseError(f"OTS member {side.value}[{roster_index}] has no species")
    species = species_value.split(",", 1)[0].strip()
    nickname_value = entry.get("name")
    if nickname_value is not None and not isinstance(nickname_value, str):
        raise ReplayParseError("OTS name must be a string when present")
    nickname = nickname_value or species
    raw = orjson.dumps(entry, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return OTSMember(
        member_id=ReplayMemberId(side, roster_index),
        nickname=nickname,
        species=species,
        item=_optional_ots_string(entry, "item"),
        ability=_optional_ots_string(entry, "ability"),
        moves=_moves(entry.get("moves")),
        nature=_optional_ots_string(entry, "nature"),
        gender=_optional_ots_string(entry, "gender"),
        level=_level(entry.get("level")),
        evs=_serialized_evs(entry.get("evs")),
        raw_packed_set=raw,
    )


def _parse_packed_ots_member(
    side: ReplaySide,
    roster_index: int,
    packed_set: str,
) -> OTSMember:
    fields = packed_set.split("|")
    if len(fields) < 5 or not fields[0]:
        raise ReplayParseError(f"Malformed packed OTS member {side.value}[{roster_index}]")
    nickname = fields[0]
    species = fields[1] or nickname
    return OTSMember(
        member_id=ReplayMemberId(side, roster_index),
        nickname=nickname,
        species=species,
        item=fields[2],
        ability=fields[3],
        moves=_moves(fields[4]),
        nature=fields[5] if len(fields) > 5 else "",
        gender=fields[7] if len(fields) > 7 else "",
        level=_level(fields[10] if len(fields) > 10 else None),
        evs=fields[6] if len(fields) > 6 else "",
        raw_packed_set=packed_set,
    )


def _parse_showteam_payload(side: ReplaySide, payload: str) -> tuple[OTSMember, ...]:
    if not payload:
        raise ReplayParseError(f"Empty showteam payload for {side.value}")
    try:
        value = orjson.loads(payload)
    except orjson.JSONDecodeError:
        value = None

    if isinstance(value, Mapping):
        entries = value.get("team", value.get("pokemon"))
    else:
        entries = value

    if isinstance(entries, list):
        members = []
        for roster_index, entry in enumerate(entries):
            if isinstance(entry, Mapping):
                members.append(_parse_mapping_ots_member(side, roster_index, entry))
            elif isinstance(entry, str) and entry:
                species = entry.split(",", 1)[0].strip()
                members.append(
                    _parse_mapping_ots_member(
                        side,
                        roster_index,
                        {"name": species, "species": species, "raw": entry},
                    )
                )
            else:
                raise ReplayParseError(
                    f"OTS member {side.value}[{roster_index}] must be an object or string"
                )
    elif value is None:
        packed_sets = payload.split("]")
        members = [
            _parse_packed_ots_member(side, roster_index, packed_set)
            for roster_index, packed_set in enumerate(packed_sets)
            if packed_set
        ]
    else:
        raise ReplayParseError(f"Unsupported showteam payload for {side.value}")

    if not members:
        raise ReplayParseError(f"Showteam payload for {side.value} contains no members")
    if len(members) > 6:
        raise ReplayParseError(f"Showteam payload for {side.value} contains more than six members")
    return tuple(members)


def _ots(lines: tuple[ProtocolLine, ...]) -> tuple[OTSData, OTSData]:
    payloads: dict[ReplaySide, list[str]] = {ReplaySide.P1: [], ReplaySide.P2: []}
    for line in lines:
        if len(line.parts) < 4 or line.parts[1] != "showteam":
            continue
        try:
            side = ReplaySide(line.parts[2])
        except ValueError as exc:
            raise ReplayParseError(f"Invalid showteam side at line {line.index}") from exc
        payloads[side].append("|".join(line.parts[3:]))

    result: list[OTSData] = []
    for side in (ReplaySide.P1, ReplaySide.P2):
        if len(payloads[side]) > 1:
            raise ReplayParseError(
                f"Replay cannot contain repeated showteam payloads for {side.value}; "
                f"received {len(payloads[side])}"
            )
        if not payloads[side]:
            result.append(OTSData(side, "", ()))
            continue
        raw_payload = payloads[side][0]
        result.append(OTSData(side, raw_payload, _parse_showteam_payload(side, raw_payload)))

    return result[0], result[1]


def _outcome(metadata: ReplayMetadata, lines: tuple[ProtocolLine, ...]) -> ReplayOutcome:
    winner = -1
    end_reason = GameEndReason.NORMAL
    terminal: int | None = None
    players = tuple(normalize_showdown_id(name) for name in metadata.player_names)
    pending_message_reason: GameEndReason | None = None
    for line in lines:
        if len(line.parts) < 2:
            continue
        tag = line.parts[1]
        if tag in {"message", "-message"}:
            text = "|".join(line.parts[2:]).casefold()
            if (
                "timed out" in text
                or "timeout" in text
                or "inactive" in text
                or "inactivity" in text
            ):
                pending_message_reason = GameEndReason.TIMEOUT
            elif "forfeit" in text or "changing their name" in text or "inappropriate name" in text:
                pending_message_reason = GameEndReason.FORFEIT
            continue
        if tag == "win":
            if len(line.parts) != 3 or not line.parts[2]:
                raise ReplayInputContractError(f"Malformed win line at index {line.index}")
            terminal = line.index
            winner_name = normalize_showdown_id(line.parts[2])
            if winner_name in players:
                winner = players.index(winner_name)
            else:
                raise ReplayInputContractError(
                    f"Win line names an unknown player at index {line.index}"
                )
        elif tag == "tie":
            if len(line.parts) != 2:
                raise ReplayInputContractError(f"Malformed tie line at index {line.index}")
            terminal = line.index
        elif tag == "forfeit":
            if len(line.parts) != 3 or not line.parts[2]:
                raise ReplayInputContractError(f"Malformed forfeit line at index {line.index}")
            terminal = line.index
            end_reason = GameEndReason.FORFEIT
        if tag in {"win", "tie"} and pending_message_reason is not None:
            end_reason = pending_message_reason
    turns = max((line.turn or 0 for line in lines), default=0)
    if terminal is None:
        # Incomplete captures are retained for grouping/quarantine.  A
        # terminal marker is required before reconstruction, but its absence
        # is not malformed transport input.
        return ReplayOutcome(-1, GameEndReason.NORMAL, turns, None)
    if (
        winner < 0
        and end_reason is GameEndReason.NORMAL
        and any(line.parts[1] == "win" for line in lines if len(line.parts) > 1)
    ):
        raise ReplayInputContractError("Replay win outcome has no recognized winner")
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


__all__ = [
    "ReplayDocument",
    "ReplayParseError",
    "ReplayInputContractError",
    "parse_replay_payload",
]
