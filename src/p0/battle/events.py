"""Ordered battle event records addressed by the four active battlefield positions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

from p0.model.architecture_contract import RAW_EVENT_COUNT

if TYPE_CHECKING:
    from p0.model.tokenizer import PokemonTokenizer

LOGGER = logging.getLogger(__name__)


class EventKind(IntEnum):
    NONE = 0  # padding row
    MOVE = 1
    SWITCH = 2
    FAINT = 3
    CANT = 4
    DAMAGE = 5
    HEAL = 6
    BOOST = 7
    CRIT = 8
    FAIL = 9
    ACTIVATE = 10  # a move effect such as Protect or Wide Guard blocked or triggered
    ITEM = 11
    SWAP = 12


class EventPosition(IntEnum):
    OWN_LEFT = 0
    OWN_RIGHT = 1
    OPPONENT_LEFT = 2
    OPPONENT_RIGHT = 3
    NONE = 4


class EventDetail(IntEnum):
    NONE = 0
    ATK = 1
    DEF = 2
    SPA = 3
    SPD = 4
    SPE = 5
    ACCURACY = 6
    EVASION = 7
    FLINCH = 8
    ITEM_REVEALED = 9
    ITEM_REMOVED = 10


SPATIAL_SLOT_COUNT = 4
MAX_EVENT_RECORDS = RAW_EVENT_COUNT
EVENT_CATEGORICAL_WIDTH = 5  # [kind, source, target, move_id, detail]
EVENT_NUMERICAL_WIDTH = 2  # [amount, amount_known]
NUM_EVENT_KINDS = len(EventKind)
NUM_EVENT_POSITIONS = len(EventPosition)
NUM_EVENT_DETAILS = len(EventDetail)

MAX_BOOST_STAGES = 6.0
_HP_STRIP_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ% "
_STAT_DETAILS = {
    "atk": EventDetail.ATK,
    "def": EventDetail.DEF,
    "spa": EventDetail.SPA,
    "spd": EventDetail.SPD,
    "spe": EventDetail.SPE,
    "accuracy": EventDetail.ACCURACY,
    "evasion": EventDetail.EVASION,
}
# Showdown prints "-ability|holder|Name|boost" before the stat changes an ability
# causes (Intimidate, Moxie). These tags continue that ability's attribution.
_ABILITY_CHAIN_TAGS = frozenset({"-ability", "-boost", "-unboost", "-fail", "-immune"})


class EventRecord(NamedTuple):
    """
    One observed event. source is the position that acted or caused it, and
    target is the position it affected; either may be NONE.
    """

    kind: int
    source: int = EventPosition.NONE
    target: int = EventPosition.NONE
    move_id: int = 0
    detail: int = EventDetail.NONE
    amount: float = 0.0
    amount_known: float = 0.0


def get_hp_fraction(hp_status: str) -> float:
    """Extract float HP fraction from a Showdown HP status string."""
    hp_part = hp_status.split(" ", 1)[0]
    if "/" not in hp_part:
        return 0.0
    try:
        num, den_str = hp_part.split("/")
        den_clean = den_str.strip(_HP_STRIP_CHARS)
        return float(num) / float(den_clean)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _position(endpoint: str, player_role: str) -> int:
    """Map a Showdown entity string (e.g. 'p1a: Pikachu') to a perspective position."""
    if len(endpoint) < 3 or endpoint[0] != "p":
        return EventPosition.NONE
    player_num = endpoint[1]
    slot_letter = endpoint[2].lower()
    if player_num not in ("1", "2") or slot_letter not in ("a", "b"):
        return EventPosition.NONE

    side_offset = 0 if endpoint.startswith(player_role) else 2
    return side_offset + (0 if slot_letter == "a" else 1)


def _of_position(parts: Sequence[str], player_role: str) -> int | None:
    """Return the position named by a trailing '[of]' tag, if the line has one."""
    for part in parts[3:]:
        if part.startswith("[of] "):
            return _position(part.removeprefix("[of] "), player_role)
    return None


def _move_id(resolver: PokemonTokenizer, name: str) -> int:
    move_id, _ = resolver.resolve("moves", name)
    return move_id


class SpatialEventRecorder:
    """Collect ordered event records until the owning player consumes them at a decision."""

    __slots__ = (
        "player_role",
        "records",
        "dropped",
        "consumed",
        "_move_source",
        "_ability_source",
    )

    def __init__(self, player_role: str = "p1") -> None:
        self.player_role = player_role
        self.records: list[EventRecord] = []
        self.dropped = 0
        # True from a decision until the next protocol line; a decision requested
        # in between retries a rejected choice and observes the same interval.
        self.consumed = False
        self._move_source = EventPosition.NONE
        self._ability_source: int | None = None

    def pending(self) -> tuple[EventRecord, ...]:
        """Return the records of the interval that the next decision observes."""
        return tuple(self.records)

    def consume(self) -> tuple[EventRecord, ...]:
        """
        Close the interval that the owning player's decision observed and return its records.

        The records are cleared when the next protocol line arrives, not here, so a
        retried decision observes and consumes the same interval again.
        """
        if self.dropped:
            LOGGER.warning(
                "Dropped %d event records beyond the %d-record capacity for %s",
                self.dropped,
                MAX_EVENT_RECORDS,
                self.player_role,
            )
        self.dropped = 0
        self.consumed = True
        return tuple(self.records)

    def _add(self, record: EventRecord) -> None:
        # Keep the earliest records: they hold action order and direct move outcomes.
        if len(self.records) == MAX_EVENT_RECORDS:
            self.dropped += 1
            return
        self.records.append(record)

    def apply_line(
        self,
        parts: Sequence[str],
        resolver: PokemonTokenizer,
        hp_for: Callable[[str], float | None],
    ) -> None:
        """
        Append the records one protocol line produces, if any.

        Arguments:
          parts: the split protocol line, starting with the empty leading field
          resolver: vocabulary used to encode move names
          hp_for: HP fraction of an identifier before this line, or None if unknown

        Returns:
          None; records accumulate until consume is called.
        """
        if len(parts) < 2:
            return

        if self.consumed:
            self.records.clear()
            self.consumed = False

        tag = parts[1]
        role = self.player_role
        if tag not in _ABILITY_CHAIN_TAGS:
            self._ability_source = None

        if tag in ("turn", "upkeep"):
            self._move_source = EventPosition.NONE

        elif tag == "move" and len(parts) >= 4:
            actor = _position(parts[2], role)
            self._move_source = actor
            target = _position(parts[4], role) if len(parts) >= 5 else EventPosition.NONE
            self._add(EventRecord(EventKind.MOVE, actor, target, _move_id(resolver, parts[3])))

        elif tag in ("switch", "drag") and len(parts) >= 3:
            self._move_source = EventPosition.NONE
            self._add(EventRecord(EventKind.SWITCH, target=_position(parts[2], role)))

        elif tag == "swap" and len(parts) >= 4:
            source = _position(parts[2], role)
            if source == EventPosition.NONE or parts[3] not in ("0", "1"):
                target = EventPosition.NONE
            else:
                side_offset = 0 if source < EventPosition.OPPONENT_LEFT else 2
                target = side_offset + int(parts[3])
            self._add(EventRecord(EventKind.SWAP, source, target))

        elif tag == "faint" and len(parts) >= 3:
            self._add(EventRecord(EventKind.FAINT, target=_position(parts[2], role)))

        elif tag == "cant" and len(parts) >= 3:
            attempted = _move_id(resolver, parts[4]) if len(parts) >= 5 else 0
            detail = (
                EventDetail.FLINCH if len(parts) >= 4 and parts[3] == "flinch" else EventDetail.NONE
            )
            self._add(
                EventRecord(
                    EventKind.CANT, _position(parts[2], role), move_id=attempted, detail=detail
                )
            )

        elif tag in ("-damage", "-heal") and len(parts) >= 4:
            of_source = _of_position(parts, role)
            if of_source is not None:
                source = of_source
            elif any(part.startswith("[from]") for part in parts[4:]):
                source = EventPosition.NONE
            else:
                source = self._move_source

            pre_hp = hp_for(parts[2])
            amount = 0.0 if pre_hp is None else get_hp_fraction(parts[3]) - pre_hp
            kind = EventKind.DAMAGE if tag == "-damage" else EventKind.HEAL
            self._add(
                EventRecord(
                    kind,
                    source,
                    _position(parts[2], role),
                    amount=amount,
                    amount_known=float(pre_hp is not None),
                )
            )

        elif tag in ("-boost", "-unboost") and len(parts) >= 5:
            target = _position(parts[2], role)
            of_source = _of_position(parts, role)
            if of_source is not None:
                source = of_source
            elif self._ability_source is not None:
                source = self._ability_source
            elif any(part.startswith("[from]") for part in parts[5:]):
                source = target
            else:
                source = self._move_source

            try:
                stages = int(parts[4]) / MAX_BOOST_STAGES
            except ValueError:
                return
            self._add(
                EventRecord(
                    EventKind.BOOST,
                    source,
                    target,
                    detail=_STAT_DETAILS.get(parts[3], EventDetail.NONE),
                    amount=stages if tag == "-boost" else -stages,
                    amount_known=1.0,
                )
            )

        elif tag == "-ability" and len(parts) >= 3:
            self._ability_source = _position(parts[2], role)

        elif tag == "-crit" and len(parts) >= 3:
            self._add(EventRecord(EventKind.CRIT, self._move_source, _position(parts[2], role)))

        elif tag == "-miss" and len(parts) >= 3:
            target = _position(parts[3], role) if len(parts) >= 4 else EventPosition.NONE
            self._add(EventRecord(EventKind.FAIL, _position(parts[2], role), target))

        elif tag in ("-fail", "-immune") and len(parts) >= 3:
            self._add(EventRecord(EventKind.FAIL, self._move_source, _position(parts[2], role)))

        elif tag == "-activate" and len(parts) >= 4 and parts[3].startswith("move: "):
            self._add(
                EventRecord(
                    EventKind.ACTIVATE,
                    self._move_source,
                    _position(parts[2], role),
                    _move_id(resolver, parts[3].removeprefix("move: ")),
                )
            )

        elif tag in ("-item", "-enditem") and len(parts) >= 3:
            detail = EventDetail.ITEM_REVEALED if tag == "-item" else EventDetail.ITEM_REMOVED
            self._add(EventRecord(EventKind.ITEM, target=_position(parts[2], role), detail=detail))


__all__ = [
    "EVENT_CATEGORICAL_WIDTH",
    "EVENT_NUMERICAL_WIDTH",
    "MAX_BOOST_STAGES",
    "MAX_EVENT_RECORDS",
    "NUM_EVENT_DETAILS",
    "NUM_EVENT_KINDS",
    "NUM_EVENT_POSITIONS",
    "SPATIAL_SLOT_COUNT",
    "EventDetail",
    "EventKind",
    "EventPosition",
    "EventRecord",
    "SpatialEventRecorder",
    "get_hp_fraction",
]
