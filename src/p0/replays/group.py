"""Deterministic grouping of public replay games into Bo3 series."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from p0.replays.identity import normalize_showdown_id
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
    players = tuple(normalize_showdown_id(name) for name in document.metadata.player_names)
    if len(set(players)) != 2 or not all(players):
        raise ValueError(f"Replay {document.metadata.replay_id} does not have two distinct players")
    return tuple(sorted(players))  # type: ignore[return-value]


def _roles(document: ReplayDocument, players: tuple[str, str]) -> tuple[int, int]:
    source = tuple(normalize_showdown_id(name) for name in document.metadata.player_names)
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


def _series_id(format_id: str, key: str, players: tuple[str, str]) -> str:
    value = "\n".join((format_id, key, *players))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _ordered_games(documents: tuple[ReplayDocument, ...]) -> tuple[ReplayDocument, ...]:
    return tuple(
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


def _membership_numbers(games: tuple[ReplayDocument, ...]) -> tuple[int, ...]:
    return tuple(
        game.metadata.game_number if game.metadata.game_number is not None else index
        for index, game in enumerate(games, 1)
    )


def _numbering_diagnostics(
    games: tuple[ReplayDocument, ...],
    numbers: tuple[int, ...],
) -> tuple[GroupingDiagnostic, ...]:
    replay_ids = tuple(game.metadata.replay_id for game in games)
    explicit = tuple(game.metadata.game_number for game in games)
    diagnostics: list[GroupingDiagnostic] = []
    if any(number is None for number in explicit) and any(
        number is not None for number in explicit
    ):
        diagnostics.append(
            GroupingDiagnostic(
                "missing_game_number",
                replay_ids,
                "some games have authoritative numbers while others do not",
            )
        )
    if len(set(numbers)) != len(numbers):
        diagnostics.append(
            GroupingDiagnostic(
                "duplicate_game_number",
                replay_ids,
                f"series has duplicate game numbers: {numbers}",
            )
        )
    if tuple(sorted(set(numbers))) != tuple(range(1, len(games) + 1)):
        diagnostics.append(
            GroupingDiagnostic(
                "non_contiguous_game_numbers",
                replay_ids,
                f"series game numbers are not contiguous from one: {numbers}",
            )
        )
    return tuple(diagnostics)


def _series_score(
    games: tuple[ReplayDocument, ...],
    players: tuple[str, str],
) -> tuple[tuple[int, int], tuple[GroupingDiagnostic, ...]]:
    score = [0, 0]
    games_after_clinch: list[str] = []
    games_without_winner: list[str] = []
    clinched = False
    for game in games:
        if clinched:
            games_after_clinch.append(game.metadata.replay_id)
            continue
        winner = game.outcome.winner
        if winner in (0, 1):
            score[_roles(game, players)[winner]] += 1
            clinched = max(score) == 2
        else:
            games_without_winner.append(game.metadata.replay_id)
    diagnostics: list[GroupingDiagnostic] = []
    if games_without_winner:
        diagnostics.append(
            GroupingDiagnostic(
                "missing_outcome",
                tuple(game.metadata.replay_id for game in games),
                f"games have no public winner: {tuple(games_without_winner)}",
            )
        )
    if games_after_clinch:
        diagnostics.append(
            GroupingDiagnostic(
                "game_after_series_clinch",
                tuple(game.metadata.replay_id for game in games),
                f"games occur after the series was won: {tuple(games_after_clinch)}",
            )
        )
    return (score[0], score[1]), tuple(diagnostics)


def _make_group(
    documents: tuple[ReplayDocument, ...],
    *,
    key: str,
    method: GroupingMethod,
    diagnostics: tuple[GroupingDiagnostic, ...] = (),
) -> GroupedSeries:
    if not documents:
        raise ValueError("Cannot group an empty replay collection")
    format_id = documents[0].metadata.format_id
    players = _canonical_players(documents[0])
    if any(_canonical_players(document) != players for document in documents):
        raise ValueError("A series cannot contain games from different player pairs")
    ordered = _ordered_games(documents)
    numbers = _membership_numbers(ordered)
    numbering_diagnostics = _numbering_diagnostics(ordered, numbers)
    score, outcome_diagnostics = _series_score(ordered, players)
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
    complete = (
        max(score) == 2
        and not numbering_diagnostics
        and not outcome_diagnostics
        and not conflicts
        and not any(diagnostic.code == "too_many_games" for diagnostic in diagnostics)
    )
    series_id = _series_id(format_id, key, players)
    membership_diagnostics = tuple(
        diagnostic.code
        for diagnostic in (
            *diagnostics,
            *numbering_diagnostics,
            *outcome_diagnostics,
            *conflicts,
        )
    )
    memberships = tuple(
        SeriesMembership(
            series_id=series_id,
            replay_id=game.metadata.replay_id,
            game_number=game_number,
            canonical_player_roles=_roles(game, players),
            grouping_method=method,
            confidence=1.0 if method is GroupingMethod.PARENT_ROOM else 0.5,
            diagnostics=membership_diagnostics,
        )
        for game, game_number in zip(ordered, numbers, strict=True)
    )
    record = SeriesRecord(
        series_id=series_id,
        format_id=format_id,
        players=players,
        game_replay_ids=tuple(game.metadata.replay_id for game in ordered),
        game_player_roles=tuple(membership.canonical_player_roles for membership in memberships),
        team_hashes=team_hashes,
        is_complete=complete,
        score=score,
        grouping_method=method,
        grouping_confidence=1.0 if method is GroupingMethod.PARENT_ROOM else 0.5,
    )
    group_diagnostics = (
        *diagnostics,
        *numbering_diagnostics,
        *outcome_diagnostics,
        *conflicts,
    )
    return GroupedSeries(record, ordered, memberships, group_diagnostics)


def group_replays(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
    max_games: int = 3,
) -> GroupingResult:
    """Group compatible documents without merging conflicting player pairs.

    Arguments:
        documents: Parsed, model-agnostic replay documents.
        format_id: Optional exact format filter.
        max_games: Maximum number of games retained in one series.

    Returns:
        Deterministically ordered groups and quarantine diagnostics.
    """
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
    buckets: dict[tuple[str, str, tuple[str, str]], list[ReplayDocument]] = {}
    methods: dict[tuple[str, str, tuple[str, str]], GroupingMethod] = {}
    fallback_state: dict[tuple[str, tuple[str, str]], tuple[datetime, int]] = {}
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
            fallback_key = (document.metadata.format_id, players)
            previous = fallback_state.get(fallback_key)
            if previous is None or (_time(document) - previous[0]).total_seconds() > 24 * 60 * 60:
                cluster = 0 if previous is None else previous[1] + 1
            else:
                cluster = previous[1]
            key = f"fallback:{players[0]}:{players[1]}:{cluster}"
            fallback_state[fallback_key] = (_time(document), cluster)
        bucket = (document.metadata.format_id, key, players)
        buckets.setdefault(bucket, []).append(document)
        methods[bucket] = method

    result: list[GroupedSeries] = []
    for bucket in sorted(buckets):
        games = tuple(
            sorted(buckets[bucket], key=lambda item: (_time(item), item.metadata.replay_id))
        )
        method = methods[bucket]
        if method is GroupingMethod.FALLBACK_SAME_PLAYERS and len(games) > 1:
            chunks: list[list[ReplayDocument]] = [[]]
            previous_hashes: tuple[str, str] | None = None
            for game in games:
                roles = _roles(game, bucket[2])
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
                    group = _make_group(
                        tuple(chunk),
                        key=f"{bucket[1]}:{chunk_index}",
                        method=method,
                    )
                    result.append(group)
                    diagnostics.extend(group.diagnostics)
                    if not group.record.is_complete:
                        diagnostics.append(
                            GroupingDiagnostic(
                                "incomplete_series",
                                group.record.game_replay_ids,
                                "series does not contain a validated two-win result",
                            )
                        )
                continue
        group_diagnostics: list[GroupingDiagnostic] = []
        if len(games) > max_games:
            diagnostic = GroupingDiagnostic(
                "too_many_games",
                tuple(game.metadata.replay_id for game in games),
                f"group contains {len(games)} games; retained first {max_games}",
            )
            diagnostics.append(diagnostic)
            group_diagnostics.append(diagnostic)
            games = games[:max_games]
        group = _make_group(
            games,
            key=bucket[1],
            method=method,
            diagnostics=tuple(group_diagnostics),
        )
        result.append(group)
        diagnostics.extend(
            diagnostic for diagnostic in group.diagnostics if diagnostic not in group_diagnostics
        )
        if not group.record.is_complete:
            diagnostics.append(
                GroupingDiagnostic(
                    "incomplete_series",
                    group.record.game_replay_ids,
                    "series does not contain a validated two-win result",
                )
            )
    return GroupingResult(tuple(result), tuple(diagnostics))


def individual_games(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
) -> tuple[ReplayDocument, ...]:
    """Return deterministic, deduplicated games for model-agnostic Bo1 use."""
    selected: dict[str, ReplayDocument] = {}
    for document in documents:
        if format_id is not None and document.metadata.format_id != format_id:
            continue
        previous = selected.get(document.metadata.replay_id)
        if previous is not None and previous.raw_payload != document.raw_payload:
            raise ValueError(
                f"Conflicting payloads share replay id {document.metadata.replay_id!r}"
            )
        selected[document.metadata.replay_id] = document
    return tuple(
        sorted(
            selected.values(),
            key=lambda document: (_time(document), document.metadata.replay_id),
        )
    )


def validated_bo3_series(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
) -> tuple[GroupedSeries, ...]:
    """Return only explicit-parent series with complete, unambiguous metadata."""
    grouping = group_replays(documents, format_id=format_id)
    blocking = {
        "duplicate_game_number",
        "fallback_team_conflict",
        "game_after_series_clinch",
        "missing_game_number",
        "missing_outcome",
        "non_contiguous_game_numbers",
        "team_identity_conflict",
        "too_many_games",
    }
    return tuple(
        group
        for group in grouping.series
        if group.record.is_complete
        and group.record.grouping_method is GroupingMethod.PARENT_ROOM
        and all(game.metadata.game_number is not None for game in group.games)
        and not any(diagnostic.code in blocking for diagnostic in group.diagnostics)
    )


__all__ = [
    "GroupedSeries",
    "GroupingDiagnostic",
    "GroupingResult",
    "group_replays",
    "individual_games",
    "validated_bo3_series",
]
