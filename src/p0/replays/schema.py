"""Versioned replay intermediate representation and action-evidence labels.

The raw layer beneath this schema is deliberately schema-free: scraped replay
JSON is stored as verbatim immutable response bytes on disk
(artifacts/replays/raw/<format_id>/<replay_id>.json.gz) and is never wrapped
in a versioned record. The only structured raw-layer artifact is an
append-only fetch index of FetchIndexEntry lines used for resume and
deduplication. Every IR record here is derived from those raw bytes and fully
regenerable, so bumping REPLAY_IR_SCHEMA_VERSION means re-running the parser
over the cache, never re-scraping.

The IR is also independent of the tensor observation schema: records store
raw protocol text and action ids from the closed 49-action contract, never
tensors or vocabulary ids, so vocabulary or observation changes never require
re-parsing protocol structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Mapping

from p0.battle.actions import ACT_SIZE

REPLAY_IR_SCHEMA_VERSION = 1


class GroupingMethod(IntEnum):
    UNSPECIFIED = 0
    PARENT_ROOM = 1
    FALLBACK_SAME_PLAYERS = 2


class GameEndReason(IntEnum):
    UNSPECIFIED = 0
    NORMAL = 1
    FORFEIT = 2
    TIMEOUT = 3


class DecisionType(IntEnum):
    UNSPECIFIED = 0
    TEAM_PREVIEW = 1
    TURN = 2
    FORCED_SWITCH = 3
    PIVOT_SWITCH = 4
    FORCED_PASS = 5


class LabelKind(IntEnum):
    UNSPECIFIED = 0
    EXACT = 1
    PARTIAL = 2
    UNKNOWN = 3


class MaskProvenance(IntEnum):
    UNSPECIFIED = 0
    CONSERVATIVE_RECONSTRUCTED = 1
    ORACLE_REQUEST = 2


def _require_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


def _require_ir_schema(value: Any, owner: str) -> None:
    if value != REPLAY_IR_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {owner} ir_schema {value!r}; expected {REPLAY_IR_SCHEMA_VERSION}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_iso_timestamp(value: str, owner: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{owner} must be an ISO-8601 timestamp") from exc


@dataclass(frozen=True, slots=True)
class FetchIndexEntry:
    """One line of the append-only raw-cache fetch index.

    Knows nothing about replay content, so it survives every IR schema change.
    """

    replay_id: str
    format_id: str
    source_url: str
    fetched_at: str
    http_status: int
    content_sha256: str
    byte_size: int

    _FIELDS = frozenset(
        {
            "replay_id",
            "format_id",
            "source_url",
            "fetched_at",
            "http_status",
            "content_sha256",
            "byte_size",
        }
    )

    def __post_init__(self) -> None:
        if not self.replay_id or not self.format_id or not self.source_url:
            raise ValueError("Fetch index entries require replay_id, format_id, and source_url")
        _require_iso_timestamp(self.fetched_at, "FetchIndexEntry.fetched_at")
        if type(self.http_status) is not int or not 100 <= self.http_status <= 599:
            raise ValueError("FetchIndexEntry.http_status must be an HTTP status code")
        if not _is_sha256(self.content_sha256):
            raise ValueError("FetchIndexEntry.content_sha256 must be a lowercase SHA-256 digest")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("FetchIndexEntry.byte_size must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "format_id": self.format_id,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at,
            "http_status": self.http_status,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FetchIndexEntry:
        _require_fields(value, cls._FIELDS, "FetchIndexEntry")
        return cls(
            replay_id=str(value["replay_id"]),
            format_id=str(value["format_id"]),
            source_url=str(value["source_url"]),
            fetched_at=str(value["fetched_at"]),
            http_status=int(value["http_status"]),
            content_sha256=str(value["content_sha256"]),
            byte_size=int(value["byte_size"]),
        )


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    """Reconstructed joint-action supervision for one decision.

    Candidates are explicit joint pairs at this layer; the flat
    values-plus-offsets ragged encoding exists only in compiled tensor shards.
    One representation covers all label kinds: EXACT stores exactly one
    candidate, PARTIAL two or more, UNKNOWN none.
    """

    label_kind: LabelKind
    candidates: tuple[tuple[int, int], ...]
    confidence: float
    mask_provenance: MaskProvenance
    tags: tuple[str, ...] = ()

    _FIELDS = frozenset({"label_kind", "candidates", "confidence", "mask_provenance", "tags"})

    def __post_init__(self) -> None:
        if self.label_kind is LabelKind.UNSPECIFIED:
            raise ValueError("ActionEvidence.label_kind must be EXACT, PARTIAL, or UNKNOWN")
        if self.mask_provenance is MaskProvenance.UNSPECIFIED:
            raise ValueError("ActionEvidence.mask_provenance must be specified")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ActionEvidence.confidence must be in [0, 1]")
        expected = {LabelKind.EXACT: "exactly one candidate", LabelKind.PARTIAL: "two or more"}
        if self.label_kind is LabelKind.EXACT and len(self.candidates) != 1:
            raise ValueError(f"EXACT evidence requires {expected[LabelKind.EXACT]}")
        if self.label_kind is LabelKind.PARTIAL and len(self.candidates) < 2:
            raise ValueError(f"PARTIAL evidence requires {expected[LabelKind.PARTIAL]} candidates")
        if self.label_kind is LabelKind.UNKNOWN and self.candidates:
            raise ValueError("UNKNOWN evidence must carry no candidates")
        seen: set[tuple[int, int]] = set()
        for candidate in self.candidates:
            if len(candidate) != 2:
                raise ValueError("Candidates must be joint action pairs")
            for action in candidate:
                if type(action) is not int or not 0 <= action < ACT_SIZE:
                    raise ValueError(f"Candidate action id {action!r} outside [0, {ACT_SIZE})")
            if candidate in seen:
                raise ValueError(f"Duplicate candidate joint action {candidate}")
            seen.add(candidate)

    @property
    def exact_action(self) -> tuple[int, int]:
        if self.label_kind is not LabelKind.EXACT:
            raise ValueError("exact_action is only defined for EXACT evidence")
        return self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_kind": int(self.label_kind),
            "candidates": [list(candidate) for candidate in self.candidates],
            "confidence": self.confidence,
            "mask_provenance": int(self.mask_provenance),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionEvidence:
        _require_fields(value, cls._FIELDS, "ActionEvidence")
        try:
            return cls(
                label_kind=LabelKind(value["label_kind"]),
                candidates=tuple(
                    (int(candidate[0]), int(candidate[1])) for candidate in value["candidates"]
                ),
                confidence=float(value["confidence"]),
                mask_provenance=MaskProvenance(value["mask_provenance"]),
                tags=tuple(str(tag) for tag in value["tags"]),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid serialized ActionEvidence: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One inferred decision request and its attached evidence.

    Line indices bound the execution segment in the owning GameRecord's
    protocol_lines: the observation is captured before pre_line_index and the
    evidence derives from lines [pre_line_index, post_line_index).
    """

    decision_index: int
    player: int
    decision_type: DecisionType
    pre_line_index: int
    post_line_index: int
    evidence: ActionEvidence

    _FIELDS = frozenset(
        {
            "decision_index",
            "player",
            "decision_type",
            "pre_line_index",
            "post_line_index",
            "evidence",
        }
    )

    def __post_init__(self) -> None:
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("DecisionRecord.decision_index must be a nonnegative integer")
        if self.player not in (0, 1):
            raise ValueError("DecisionRecord.player must be 0 or 1")
        if self.decision_type is DecisionType.UNSPECIFIED:
            raise ValueError("DecisionRecord.decision_type must be specified")
        if not 0 <= self.pre_line_index <= self.post_line_index:
            raise ValueError("DecisionRecord line indices must satisfy 0 <= pre <= post")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "player": self.player,
            "decision_type": int(self.decision_type),
            "pre_line_index": self.pre_line_index,
            "post_line_index": self.post_line_index,
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionRecord:
        _require_fields(value, cls._FIELDS, "DecisionRecord")
        return cls(
            decision_index=int(value["decision_index"]),
            player=int(value["player"]),
            decision_type=DecisionType(value["decision_type"]),
            pre_line_index=int(value["pre_line_index"]),
            post_line_index=int(value["post_line_index"]),
            evidence=ActionEvidence.from_dict(value["evidence"]),
        )


@dataclass(frozen=True, slots=True)
class ReplayDiagnostics:
    """Parser and reconstruction counters kept alongside the derived records.

    Counter keys mirror EVENT_DIAGNOSTICS (oov_ids, missing_pre_hp,
    grounding_misses) plus reconstruction-specific counts, so ambiguity and
    loss masking stay visible in every compile report.
    """

    counters: Mapping[str, int]
    parse_errors: tuple[str, ...] = ()

    _FIELDS = frozenset({"counters", "parse_errors"})

    def __post_init__(self) -> None:
        for key, count in self.counters.items():
            if not isinstance(key, str) or type(count) is not int or count < 0:
                raise ValueError("Diagnostics counters must map strings to nonnegative integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": {key: self.counters[key] for key in sorted(self.counters)},
            "parse_errors": list(self.parse_errors),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayDiagnostics:
        _require_fields(value, cls._FIELDS, "ReplayDiagnostics")
        counters = value["counters"]
        if not isinstance(counters, Mapping):
            raise ValueError("ReplayDiagnostics.counters must be a JSON object")
        return cls(
            counters={str(key): int(count) for key, count in counters.items()},
            parse_errors=tuple(str(item) for item in value["parse_errors"]),
        )


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One game of a series with raw protocol lines and derived decisions."""

    game_id: str
    series_id: str
    game_number: int
    protocol_lines: tuple[str, ...]
    ots_payloads: tuple[str, str]
    winner: int
    end_reason: GameEndReason
    turns: int
    decisions: tuple[DecisionRecord, ...]
    diagnostics: ReplayDiagnostics
    ir_schema: int = REPLAY_IR_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "game_id",
            "series_id",
            "game_number",
            "protocol_lines",
            "ots_payloads",
            "winner",
            "end_reason",
            "turns",
            "decisions",
            "diagnostics",
            "ir_schema",
        }
    )

    def __post_init__(self) -> None:
        _require_ir_schema(self.ir_schema, "GameRecord")
        if not self.game_id or not self.series_id:
            raise ValueError("GameRecord requires game_id and series_id")
        if type(self.game_number) is not int or self.game_number < 1:
            raise ValueError("GameRecord.game_number must be a positive integer")
        if len(self.ots_payloads) != 2:
            raise ValueError("GameRecord.ots_payloads must hold both players' sheets")
        if self.winner not in (-1, 0, 1):
            raise ValueError("GameRecord.winner must be 0, 1, or -1 for no result")
        if self.end_reason is GameEndReason.UNSPECIFIED:
            raise ValueError("GameRecord.end_reason must be specified")
        if type(self.turns) is not int or self.turns < 0:
            raise ValueError("GameRecord.turns must be a nonnegative integer")
        line_count = len(self.protocol_lines)
        previous_index = -1
        for decision in self.decisions:
            if decision.decision_index <= previous_index:
                raise ValueError("GameRecord decisions must have ascending decision_index")
            previous_index = decision.decision_index
            if decision.post_line_index > line_count:
                raise ValueError(
                    f"Decision {decision.decision_index} boundary exceeds protocol length"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "series_id": self.series_id,
            "game_number": self.game_number,
            "protocol_lines": list(self.protocol_lines),
            "ots_payloads": list(self.ots_payloads),
            "winner": self.winner,
            "end_reason": int(self.end_reason),
            "turns": self.turns,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "diagnostics": self.diagnostics.to_dict(),
            "ir_schema": self.ir_schema,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GameRecord:
        _require_fields(value, cls._FIELDS, "GameRecord")
        _require_ir_schema(value["ir_schema"], "GameRecord")
        ots = tuple(str(item) for item in value["ots_payloads"])
        if len(ots) != 2:
            raise ValueError("GameRecord.ots_payloads must hold both players' sheets")
        return cls(
            game_id=str(value["game_id"]),
            series_id=str(value["series_id"]),
            game_number=int(value["game_number"]),
            protocol_lines=tuple(str(line) for line in value["protocol_lines"]),
            ots_payloads=(ots[0], ots[1]),
            winner=int(value["winner"]),
            end_reason=GameEndReason(value["end_reason"]),
            turns=int(value["turns"]),
            decisions=tuple(DecisionRecord.from_dict(item) for item in value["decisions"]),
            diagnostics=ReplayDiagnostics.from_dict(value["diagnostics"]),
            ir_schema=int(value["ir_schema"]),
        )


@dataclass(frozen=True, slots=True)
class SeriesRecord:
    """A grouped Bo3 series with ordered games and canonical player identity.

    team_hashes are CanonicalTeam.team_hash values (order- and
    spelling-independent), tying the series to corpus team identity.
    game_player_roles maps each game's p1/p2 to the canonical player index.
    """

    series_id: str
    format_id: str
    players: tuple[str, str]
    game_replay_ids: tuple[str, ...]
    game_player_roles: tuple[tuple[int, int], ...]
    team_hashes: tuple[str, str]
    is_complete: bool
    score: tuple[int, int]
    grouping_method: GroupingMethod
    grouping_confidence: float
    ir_schema: int = REPLAY_IR_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "series_id",
            "format_id",
            "players",
            "game_replay_ids",
            "game_player_roles",
            "team_hashes",
            "is_complete",
            "score",
            "grouping_method",
            "grouping_confidence",
            "ir_schema",
        }
    )

    def __post_init__(self) -> None:
        _require_ir_schema(self.ir_schema, "SeriesRecord")
        if not self.series_id or not self.format_id:
            raise ValueError("SeriesRecord requires series_id and format_id")
        if len(self.players) != 2 or not all(self.players):
            raise ValueError("SeriesRecord.players must name both players")
        if not 1 <= len(self.game_replay_ids) <= 3 or not all(self.game_replay_ids):
            raise ValueError("SeriesRecord.game_replay_ids must hold one to three replay ids")
        if len(self.game_player_roles) != len(self.game_replay_ids):
            raise ValueError("SeriesRecord requires one role mapping per game")
        for roles in self.game_player_roles:
            if sorted(roles) != [0, 1]:
                raise ValueError("Each game's roles must be a permutation of (0, 1)")
        if len(self.team_hashes) != 2 or not all(_is_sha256(digest) for digest in self.team_hashes):
            raise ValueError("SeriesRecord.team_hashes must be two lowercase SHA-256 digests")
        if len(self.score) != 2 or any(type(wins) is not int or wins < 0 for wins in self.score):
            raise ValueError("SeriesRecord.score must be two nonnegative win counts")
        if sum(self.score) > len(self.game_replay_ids) or max(self.score) > 2:
            raise ValueError("SeriesRecord.score is inconsistent with the game count")
        if self.is_complete and max(self.score) != 2:
            raise ValueError("A complete Bo3 series requires a player with two wins")
        if self.grouping_method is GroupingMethod.UNSPECIFIED:
            raise ValueError("SeriesRecord.grouping_method must be specified")
        if not 0.0 <= self.grouping_confidence <= 1.0:
            raise ValueError("SeriesRecord.grouping_confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "format_id": self.format_id,
            "players": list(self.players),
            "game_replay_ids": list(self.game_replay_ids),
            "game_player_roles": [list(roles) for roles in self.game_player_roles],
            "team_hashes": list(self.team_hashes),
            "is_complete": self.is_complete,
            "score": list(self.score),
            "grouping_method": int(self.grouping_method),
            "grouping_confidence": self.grouping_confidence,
            "ir_schema": self.ir_schema,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SeriesRecord:
        _require_fields(value, cls._FIELDS, "SeriesRecord")
        _require_ir_schema(value["ir_schema"], "SeriesRecord")
        try:
            players = tuple(str(name) for name in value["players"])
            team_hashes = tuple(str(digest) for digest in value["team_hashes"])
            score = tuple(int(wins) for wins in value["score"])
            return cls(
                series_id=str(value["series_id"]),
                format_id=str(value["format_id"]),
                players=(players[0], players[1]),
                game_replay_ids=tuple(str(item) for item in value["game_replay_ids"]),
                game_player_roles=tuple(
                    (int(roles[0]), int(roles[1])) for roles in value["game_player_roles"]
                ),
                team_hashes=(team_hashes[0], team_hashes[1]),
                is_complete=bool(value["is_complete"]),
                score=(score[0], score[1]),
                grouping_method=GroupingMethod(value["grouping_method"]),
                grouping_confidence=float(value["grouping_confidence"]),
                ir_schema=int(value["ir_schema"]),
            )
        except (IndexError, TypeError) as exc:
            raise ValueError(f"Invalid serialized SeriesRecord: {exc}") from exc
