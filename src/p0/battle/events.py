"""Ordered battle event records with explicit participants and reported effect identities."""

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
    ABILITY = 13
    CONDITION_SET = 14
    CONDITION_END = 15
    SIDE_CONDITIONS_SWAPPED = 16


class EventPosition(IntEnum):
    OWN_LEFT = 0
    OWN_RIGHT = 1
    OPPONENT_LEFT = 2
    OPPONENT_RIGHT = 3
    NONE = 4
    OWN_SIDE = 5
    OPPONENT_SIDE = 6
    FIELD = 7
    BOTH_SIDES = 8


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
    ITEM_PRESENT = 9
    ITEM_ENDED = 10
    ITEM_EATEN = 11
    ITEM_USED = 12
    ITEM_REMOVED_BY_MOVE = 13
    ITEM_STOLEN_AND_EATEN = 14
    CANT_PARALYSIS = 15
    CANT_SLEEP = 16
    CANT_FREEZE = 17
    CANT_RECHARGE = 18
    CANT_NO_PP = 19
    CANT_ATTRACT = 20
    CANT_DISABLE = 21
    CANT_FOCUS_PUNCH = 22
    CANT_SHELL_TRAP = 23
    CANT_GRAVITY = 24
    CANT_HEAL_BLOCK = 25
    CANT_IMPRISON = 26
    CANT_TAUNT = 27
    CANT_THROAT_CHOP = 28
    CANT_ABILITY = 29
    CANT_OTHER = 30
    ABILITY_REPORTED = 31
    ABILITY_ACTIVATED = 32
    MISS = 33
    IMMUNE = 34


class EffectNamespace(IntEnum):
    NONE = 0
    POKEMON = 1
    SIDE = 2
    FIELD = 3
    WEATHER = 4


SPATIAL_SLOT_COUNT = 4
MAX_EVENT_RECORDS = RAW_EVENT_COUNT
EVENT_CATEGORICAL_WIDTH = 9
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
# Ability reports can precede several stat changes or failures. Only these tags
# continue the reported holder and ability; any other tag clears both.
_ABILITY_CHAIN_TAGS = frozenset({"-ability", "-boost", "-unboost", "-fail", "-immune"})
_CANT_DETAILS = {
    "par": EventDetail.CANT_PARALYSIS,
    "slp": EventDetail.CANT_SLEEP,
    "frz": EventDetail.CANT_FREEZE,
    "flinch": EventDetail.FLINCH,
    "recharge": EventDetail.CANT_RECHARGE,
    "nopp": EventDetail.CANT_NO_PP,
    "attract": EventDetail.CANT_ATTRACT,
    "disable": EventDetail.CANT_DISABLE,
    "focus punch": EventDetail.CANT_FOCUS_PUNCH,
    "shell trap": EventDetail.CANT_SHELL_TRAP,
    "move: gravity": EventDetail.CANT_GRAVITY,
    "move: heal block": EventDetail.CANT_HEAL_BLOCK,
    "move: imprison": EventDetail.CANT_IMPRISON,
    "move: taunt": EventDetail.CANT_TAUNT,
    "move: throat chop": EventDetail.CANT_THROAT_CHOP,
}
_BLOCKING_ABILITIES = frozenset({"armor tail", "damp", "dazzling", "queenly majesty"})
_ITEM_RECIPIENT_CAUSES = frozenset(
    {
        "move: Thief",
        "move: Covet",
        "move: Recycle",
        "move: Fling",
        "move: G-Max Replenish",
        "ability: Magician",
        "ability: Pickpocket",
        "ability: Harvest",
        "ability: Pickup",
    }
)
_ACTOR_FIRST_ACTIVATIONS = frozenset(
    {
        "Guard Split",
        "Power Split",
        "Lock-On",
        "Mind Reader",
        "Skill Swap",
        "Snatch",
    }
)
_CONDITION_TAGS = {
    "-weather": (EffectNamespace.WEATHER, "weathers"),
    "-fieldstart": (EffectNamespace.FIELD, "fields"),
    "-fieldend": (EffectNamespace.FIELD, "fields"),
    "-sidestart": (EffectNamespace.SIDE, "side_conditions"),
    "-sideend": (EffectNamespace.SIDE, "side_conditions"),
}


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
    item_id: int = 0
    ability_id: int = 0
    condition_namespace: int = EffectNamespace.NONE
    condition_id: int = 0
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


def _tag_value(parts: Sequence[str], prefix: str) -> str:
    for part in parts[2:]:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _identity(resolver: PokemonTokenizer, table: str, name: str) -> int:
    identity = resolver.effect_id_for(table, name)
    if name and not identity:
        LOGGER.warning("Unknown event %s name: %s", table, name)
    return identity


def _cause_ids(resolver: PokemonTokenizer, cause: str) -> tuple[int, int, int]:
    move = _identity(resolver, "moves", cause) if cause.startswith("move: ") else 0
    item = _identity(resolver, "items", cause) if cause.startswith("item: ") else 0
    ability = _identity(resolver, "abilities", cause) if cause.startswith("ability: ") else 0
    return move, item, ability


class SpatialEventRecorder:
    """Collect ordered event records until the owning player consumes them at a decision."""

    __slots__ = (
        "player_role",
        "records",
        "dropped",
        "consumed",
        "_move_source",
        "_ability_source",
        "_ability_id",
    )

    def __init__(self, player_role: str = "p1") -> None:
        self.player_role = player_role
        self.records: list[EventRecord] = []
        self.dropped = 0
        # True from a decision until the next protocol line; a decision requested
        # in between retries a rejected choice and observes the same interval.
        self.consumed = False
        self._move_source = EventPosition.NONE
        self._ability_source = EventPosition.NONE
        self._ability_id = 0

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
          resolver: vocabulary used to encode reported identities
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
            self._ability_source = EventPosition.NONE
            self._ability_id = 0

        cause = _tag_value(parts, "[from] ")
        of_source = _position(_tag_value(parts, "[of] "), role)
        move_id, item_id, ability_id = _cause_ids(resolver, cause)

        if tag in ("turn", "upkeep"):
            self._move_source = EventPosition.NONE

        elif tag == "move" and len(parts) >= 4:
            actor = _position(parts[2], role)
            self._move_source = actor
            target = _position(parts[4], role) if len(parts) >= 5 else EventPosition.NONE
            self._add(
                EventRecord(EventKind.MOVE, actor, target, _identity(resolver, "moves", parts[3]))
            )

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
            target = _position(parts[2], role)
            source = EventPosition.NONE
            reason = parts[3].lower() if len(parts) >= 4 else ""
            attempted = _identity(resolver, "moves", parts[4]) if len(parts) >= 5 else 0
            detail = _CANT_DETAILS.get(reason, EventDetail.CANT_OTHER)
            if reason.startswith("ability: "):
                ability_name = reason.removeprefix("ability: ")
                ability_id = _identity(resolver, "abilities", ability_name)
                if ability_name in _BLOCKING_ABILITIES:
                    detail = EventDetail.CANT_ABILITY
                    source, target = target, of_source
                elif ability_name == "truant":
                    detail = EventDetail.CANT_ABILITY
                    source = target
            if detail == EventDetail.CANT_OTHER:
                LOGGER.warning("Unknown cant reason: %s", reason)
            self._add(
                EventRecord(
                    EventKind.CANT,
                    source,
                    target,
                    attempted,
                    detail,
                    ability_id=ability_id,
                )
            )

        elif tag in ("-damage", "-heal") and len(parts) >= 4:
            target = _position(parts[2], role)
            if of_source != EventPosition.NONE:
                source = of_source
            elif cause.startswith(("ability: ", "item: ")):
                source = target
            else:
                source = EventPosition.NONE if cause else self._move_source

            pre_hp = hp_for(parts[2])
            amount = 0.0 if pre_hp is None else get_hp_fraction(parts[3]) - pre_hp
            kind = EventKind.DAMAGE if tag == "-damage" else EventKind.HEAL
            self._add(
                EventRecord(
                    kind,
                    source,
                    target,
                    move_id=move_id,
                    item_id=item_id,
                    ability_id=ability_id,
                    amount=amount,
                    amount_known=float(pre_hp is not None),
                )
            )

        elif tag in ("-boost", "-unboost") and len(parts) >= 5:
            target = _position(parts[2], role)
            if of_source != EventPosition.NONE:
                source = of_source
            elif cause:
                source = target if cause.startswith(("item: ", "ability: ")) else EventPosition.NONE
            elif self._ability_source != EventPosition.NONE:
                source = self._ability_source
            else:
                source = self._move_source
            if not cause:
                ability_id = self._ability_id

            try:
                stages = int(parts[4]) / MAX_BOOST_STAGES
            except ValueError:
                return
            self._add(
                EventRecord(
                    EventKind.BOOST,
                    source,
                    target,
                    move_id=move_id,
                    item_id=item_id,
                    ability_id=ability_id,
                    detail=_STAT_DETAILS.get(parts[3], EventDetail.NONE),
                    amount=stages if tag == "-boost" else -stages,
                    amount_known=1.0,
                )
            )

        elif tag == "-ability" and len(parts) >= 4:
            holder = _position(parts[2], role)
            ability_id = _identity(resolver, "abilities", parts[3])
            self._ability_source = holder
            self._ability_id = ability_id
            self._add(
                EventRecord(
                    EventKind.ABILITY,
                    holder,
                    detail=EventDetail.ABILITY_REPORTED,
                    ability_id=ability_id,
                )
            )

        elif tag == "-crit" and len(parts) >= 3:
            self._add(EventRecord(EventKind.CRIT, self._move_source, _position(parts[2], role)))

        elif tag == "-miss" and len(parts) >= 3:
            target = _position(parts[3], role) if len(parts) >= 4 else EventPosition.NONE
            self._add(
                EventRecord(
                    EventKind.FAIL, _position(parts[2], role), target, detail=EventDetail.MISS
                )
            )

        elif tag in ("-fail", "-immune") and len(parts) >= 3:
            target = _position(parts[2], role)
            source = self._move_source
            if of_source != EventPosition.NONE:
                source = of_source
            elif cause:
                source = target if cause.startswith(("item: ", "ability: ")) else EventPosition.NONE
            elif self._ability_source != EventPosition.NONE:
                source = self._ability_source
            if not cause:
                ability_id = self._ability_id
            self._add(
                EventRecord(
                    EventKind.FAIL,
                    source,
                    target,
                    move_id,
                    EventDetail.IMMUNE if tag == "-immune" else EventDetail.NONE,
                    item_id=item_id,
                    ability_id=ability_id,
                )
            )

        elif tag == "-activate" and len(parts) >= 4:
            participant = _position(parts[2], role)
            effect = parts[3]
            if effect.startswith("ability: "):
                self._add(
                    EventRecord(
                        EventKind.ABILITY,
                        participant,
                        of_source,
                        detail=EventDetail.ABILITY_ACTIVATED,
                        ability_id=_identity(resolver, "abilities", effect),
                    )
                )
            elif effect.startswith("move: "):
                source, target = self._move_source, participant
                if effect.removeprefix("move: ") in _ACTOR_FIRST_ACTIVATIONS:
                    source, target = participant, of_source
                elif of_source != EventPosition.NONE:
                    source = of_source
                self._add(
                    EventRecord(
                        EventKind.ACTIVATE,
                        source,
                        target,
                        _identity(resolver, "moves", effect),
                    )
                )

        elif tag in ("-item", "-enditem") and len(parts) >= 4:
            target = _position(parts[2], role)
            source = of_source
            item_id = _identity(resolver, "items", parts[3])
            if tag == "-item":
                detail = EventDetail.ITEM_PRESENT
                if cause in _ITEM_RECIPIENT_CAUSES:
                    source = target
            elif cause == "stealeat":
                detail = EventDetail.ITEM_STOLEN_AND_EATEN
                move_id = _identity(resolver, "moves", _tag_value(parts, "[move] "))
            elif "[eat]" in parts:
                detail = EventDetail.ITEM_EATEN
                source = target
            elif "[weaken]" in parts or cause == "gem":
                detail = EventDetail.ITEM_USED
                source = target
            elif cause.startswith("move: "):
                detail = EventDetail.ITEM_REMOVED_BY_MOVE
                if cause == "move: Fling":
                    source = target
            else:
                detail = EventDetail.ITEM_ENDED
                # Pickpocket's end line names only the donor, not the ability holder.
                if cause == "ability: Pickpocket" or not cause:
                    source = EventPosition.NONE
            self._add(
                EventRecord(
                    EventKind.ITEM,
                    source,
                    target,
                    move_id,
                    detail,
                    item_id,
                    ability_id,
                )
            )

        elif tag in _CONDITION_TAGS and len(parts) >= 3:
            namespace, table = _CONDITION_TAGS[tag]
            if tag == "-weather" and "[upkeep]" in parts:
                return
            name = parts[2]
            target = EventPosition.FIELD
            if namespace == EffectNamespace.SIDE:
                if len(parts) < 4:
                    return
                target = (
                    EventPosition.OWN_SIDE if name.startswith(role) else EventPosition.OPPONENT_SIDE
                )
                name = parts[3]
            cleared = tag == "-weather" and name == "none"
            kind = (
                EventKind.CONDITION_END
                if tag.endswith("end") or cleared
                else EventKind.CONDITION_SET
            )
            self._add(
                EventRecord(
                    kind,
                    of_source,
                    target,
                    move_id,
                    item_id=item_id,
                    ability_id=ability_id,
                    condition_namespace=namespace,
                    condition_id=0 if cleared else _identity(resolver, table, name),
                )
            )

        elif tag == "-swapsideconditions":
            self._add(
                EventRecord(EventKind.SIDE_CONDITIONS_SWAPPED, target=EventPosition.BOTH_SIDES)
            )


__all__ = [
    "EVENT_CATEGORICAL_WIDTH",
    "EVENT_NUMERICAL_WIDTH",
    "MAX_BOOST_STAGES",
    "MAX_EVENT_RECORDS",
    "NUM_EVENT_DETAILS",
    "NUM_EVENT_KINDS",
    "NUM_EVENT_POSITIONS",
    "SPATIAL_SLOT_COUNT",
    "EffectNamespace",
    "EventDetail",
    "EventKind",
    "EventPosition",
    "EventRecord",
    "SpatialEventRecorder",
    "get_hp_fraction",
]
