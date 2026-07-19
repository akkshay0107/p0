"""Deterministic grouping of public replay games into Bo3 series."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from p0.replays.protocol import ReplayDocument
from p0.replays.schema import (
    GroupingMethod,
    SeriesMembership,
    SeriesRecord,
)


@dataclass(frozen=True, slots=True)
class GroupingDiagnostic:
    code: str
    replay_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class GroupedSeries:
    record: SeriesRecord
    games: tuple[ReplayDocument, ...]
    memberships: tuple[SeriesMembership, ...]
    diagnostics: tuple[GroupingDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupingResult:
    series: tuple[GroupedSeries, ...]
    diagnostics: tuple[GroupingDiagnostic, ...]


def _canonical_players(document: ReplayDocument) -> tuple[str, str]:
    players = tuple(name.strip().casefold() for name in document.metadata.player_names)
    if len(set(players)) != 2 or not all(players):
        raise ValueError(f"Replay {document.metadata.replay_id} does not have two distinct players")
    return tuple(sorted(players))  # type: ignore[return-value]


def _roles(document: ReplayDocument, players: tuple[str, str]) -> tuple[int, int]:
    source = tuple(name.strip().casefold() for name in document.metadata.player_names)
    if set(source) != set(players):
        raise ValueError("Replay players do not match their grouping key")
    return source.index(players[0]), source.index(players[1])


def _parent_key(document: ReplayDocument) -> tuple[str, GroupingMethod]:
    parent = document.metadata.parent_room.strip()
    if parent:
        return parent, GroupingMethod.PARENT_ROOM
    match = re.match(r"^(.*?)(?:-game(?:-\d+)?)$", document.metadata.room_id)
    if match:
        return match.group(1), GroupingMethod.FALLBACK_SAME_PLAYERS
    return "", GroupingMethod.FALLBACK_SAME_PLAYERS


def _time(document: ReplayDocument) -> datetime:
    return datetime.fromisoformat(document.metadata.upload_time.replace("Z", "+00:00"))


def _team_hash(document: ReplayDocument, side: int) -> str:
    ots = document.ots[side]
    members = []
    for species in sorted(ots.revealed_species, key=str.casefold):
        details = ots.revealed_details.get(species, {})
        members.append(
            {
                "species": species.casefold(),
                "item": str(details.get("item", "")).casefold(),
                "ability": str(details.get("ability", "")).casefold(),
                "nature": str(details.get("nature", "")).casefold(),
                "moves": sorted(
                    str(move).casefold()
                    for move in details.get("moves", ())
                    if isinstance(move, str)
                ),
            }
        )
    payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _series_id(format_id: str, key: str, games: tuple[ReplayDocument, ...]) -> str:
    value = "\n".join((format_id, key, *(game.metadata.replay_id for game in games)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _make_group(
    documents: tuple[ReplayDocument, ...],
    *,
    key: str,
    method: GroupingMethod,
    diagnostic: GroupingDiagnostic | None = None,
) -> GroupedSeries:
    if not documents:
        raise ValueError("Cannot group an empty replay collection")
    format_id = documents[0].metadata.format_id
    players = _canonical_players(documents[0])
    if any(_canonical_players(document) != players for document in documents):
        raise ValueError("A series cannot contain games from different player pairs")
    ordered = tuple(
        sorted(
            documents,
            key=lambda document: (
                document.metadata.game_number is None,
                document.metadata.game_number or 0,
                _time(document),
                document.metadata.replay_id,
            ),
        )
    )
    score = [0, 0]
    for game in ordered:
        winner = game.outcome.winner
        if winner in (0, 1):
            source_role = winner
            score[_roles(game, players)[source_role]] += 1
    complete = max(score) == 2
    series_id = _series_id(format_id, key, ordered)
    memberships = tuple(
        SeriesMembership(
            series_id=series_id,
            replay_id=game.metadata.replay_id,
            game_number=index,
            canonical_player_roles=_roles(game, players),
            grouping_method=method,
            confidence=1.0 if method is GroupingMethod.PARENT_ROOM else 0.5,
            diagnostics=() if diagnostic is None else (diagnostic.code,),
        )
        for index, game in enumerate(ordered, 1)
    )
    first_roles = _roles(ordered[0], players)
    team_hashes = (_team_hash(ordered[0], first_roles[0]), _team_hash(ordered[0], first_roles[1]))
    conflicts = []
    for game in ordered[1:]:
        game_hashes = (
            _team_hash(game, _roles(game, players)[0]),
            _team_hash(game, _roles(game, players)[1]),
        )
        if game_hashes != team_hashes:
            conflicts.append(
                GroupingDiagnostic(
                    "team_identity_conflict",
                    tuple(item.metadata.replay_id for item in ordered),
                    "OTS team identity differs between games",
                )
            )
            break
    record = SeriesRecord(
        series_id=series_id,
        format_id=format_id,
        players=players,
        game_replay_ids=tuple(game.metadata.replay_id for game in ordered),
        game_player_roles=tuple(membership.canonical_player_roles for membership in memberships),
        team_hashes=team_hashes,
        is_complete=complete,
        score=(score[0], score[1]),
        grouping_method=method,
        grouping_confidence=1.0 if method is GroupingMethod.PARENT_ROOM else 0.5,
    )
    diagnostics = tuple(conflicts) + (() if diagnostic is None else (diagnostic,))
    return GroupedSeries(record, ordered, memberships, diagnostics)


def group_replays(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
    max_games: int = 3,
) -> GroupingResult:
    """Group compatible documents without merging conflicting player pairs."""
    if max_games < 1 or max_games > 3:
        raise ValueError("max_games must be between one and three")
    candidates = tuple(
        sorted(
            (
                document
                for document in documents
                if format_id is None or document.metadata.format_id == format_id
            ),
            key=lambda document: (_time(document), document.metadata.replay_id),
        )
    )
    diagnostics: list[GroupingDiagnostic] = []
    buckets: dict[tuple[str, tuple[str, str]], list[ReplayDocument]] = {}
    methods: dict[tuple[str, tuple[str, str]], GroupingMethod] = {}
    fallback_state: dict[tuple[str, str], tuple[datetime, int]] = {}
    for document in candidates:
        try:
            players = _canonical_players(document)
            key, method = _parent_key(document)
        except ValueError as exc:
            diagnostics.append(
                GroupingDiagnostic("invalid_players", (document.metadata.replay_id,), str(exc))
            )
            continue
        if not key:
            previous = fallback_state.get(players)
            if previous is None or (_time(document) - previous[0]).total_seconds() > 24 * 60 * 60:
                cluster = 0 if previous is None else previous[1] + 1
            else:
                cluster = previous[1]
            key = f"fallback:{players[0]}:{players[1]}:{cluster}"
            fallback_state[players] = (_time(document), cluster)
        bucket = (key, players)
        buckets.setdefault(bucket, []).append(document)
        methods[bucket] = method

    result: list[GroupedSeries] = []
    for bucket in sorted(buckets, key=lambda item: (item[0], item[1])):
        games = tuple(
            sorted(buckets[bucket], key=lambda item: (_time(item), item.metadata.replay_id))
        )
        method = methods[bucket]
        if method is GroupingMethod.FALLBACK_SAME_PLAYERS and len(games) > 1:
            chunks: list[list[ReplayDocument]] = [[]]
            previous_hashes: tuple[str, str] | None = None
            for game in games:
                roles = _roles(game, bucket[1])
                hashes = (_team_hash(game, roles[0]), _team_hash(game, roles[1]))
                if previous_hashes is not None and hashes != previous_hashes:
                    diagnostics.append(
                        GroupingDiagnostic(
                            "fallback_team_conflict",
                            tuple(item.metadata.replay_id for item in games),
                            "fallback grouping stopped at an OTS team identity conflict",
                        )
                    )
                    chunks.append([])
                chunks[-1].append(game)
                previous_hashes = hashes
            if len(chunks) > 1:
                for chunk_index, chunk in enumerate(chunks):
                    result.append(
                        _make_group(
                            tuple(chunk),
                            key=f"{bucket[0]}:{chunk_index}",
                            method=method,
                        )
                    )
                continue
        if len(games) > max_games:
            diagnostic = GroupingDiagnostic(
                "too_many_games",
                tuple(game.metadata.replay_id for game in games),
                f"group contains {len(games)} games; retained first {max_games}",
            )
            diagnostics.append(diagnostic)
            games = games[:max_games]
        if len(games) < 2:
            diagnostic = GroupingDiagnostic(
                "incomplete_series",
                tuple(game.metadata.replay_id for game in games),
                "fewer than two games were available",
            )
            diagnostics.append(diagnostic)
        if not any(game.outcome.winner in (0, 1) for game in games):
            diagnostics.append(
                GroupingDiagnostic(
                    "missing_outcome",
                    tuple(game.metadata.replay_id for game in games),
                    "no game has a public winner",
                )
            )
        group = _make_group(games, key=bucket[0], method=method, diagnostic=None)
        result.append(group)
    return GroupingResult(tuple(result), tuple(diagnostics))


__all__ = [
    "GroupedSeries",
    "GroupingDiagnostic",
    "GroupingResult",
    "group_replays",
]
