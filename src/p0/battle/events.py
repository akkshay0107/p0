"""Pure protocol-event values and parser entry point.

This module defines battle event data structures, event priorities for truncation,
and high-performance line-by-line parsing of raw Showdown protocol logs into structured
BattleEvent objects for model consumption.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import NamedTuple, Protocol


class EventTypeId(IntEnum):
    NONE = 0
    MOVE = 1
    SWITCH_IN = 2
    FAINT = 3
    DAMAGE = 4
    HEAL = 5
    STATUS_SET = 6
    STATUS_CURE = 7
    BOOST = 8
    UNBOOST = 9
    ITEM_END = 10
    ITEM_REVEAL = 11
    WEATHER_START = 12
    FIELD_START = 13
    SIDE_START = 14
    BLOCKED = 15
    CRIT = 16
    MEGA = 17
    FAILED = 18
    WEATHER_END = 19
    FIELD_END = 20
    SIDE_END = 21
    EFFECT_START = 22
    EFFECT_END = 23
    ABILITY = 24
    ITEM_TRANSFER = 25
    FORME_CHANGE = 26
    DRAG = 27
    SWAP = 28
    MISS = 29
    IMMUNE = 30
    CANT = 31
    PREPARE = 32
    SINGLEMOVE = 33
    BOOST_SET = 34
    BOOST_CLEAR = 35
    BOOST_SWAP = 36
    BOOST_INVERT = 37
    BOOST_COPY = 38
    TRANSFORM = 39
    ABILITY_END = 40
    ACTIVATE = 41
    FIELD_ACTIVATE = 42
    NO_TARGET = 43


EVENT_TYPE_COUNT = max(EventTypeId) + 1

# Counters for silent event-pipeline degradations include out of vocabulary ids,
# missing pre-HP values, and grounding misses. Reset with clear.
EVENT_DIAGNOSTICS: Counter[str] = Counter()

# Mirrors tokenizer.Resolution.OOV without importing the model layer.
_RESOLUTION_OOV = "oov"


class RawBattleEvent(NamedTuple):
    message: tuple[str, ...]
    pre_hp: float | None = None


class BattleEvent(NamedTuple):
    event_type: EventTypeId
    entity_id: str | None
    target_id: str | None = None
    move_id: int = 0
    item_id: int = 0
    status_id: int = 0
    effect_id: int = 0
    ability_id: int = 0
    flags: int = 0
    value: float = 0.0
    order: int = 0


# Structural / state-defining transitions survive truncation first: they are
# low-frequency and carry information no state token fully reconstructs.
HIGH_PRIORITY_EVENTS = frozenset(
    {
        EventTypeId.MOVE,
        EventTypeId.SWITCH_IN,
        EventTypeId.DRAG,
        EventTypeId.FAINT,
        EventTypeId.ITEM_END,
        EventTypeId.ITEM_TRANSFER,
        EventTypeId.STATUS_SET,
        EventTypeId.STATUS_CURE,
        EventTypeId.MEGA,
        EventTypeId.FORME_CHANGE,
        EventTypeId.TRANSFORM,
        EventTypeId.WEATHER_START,
        EventTypeId.WEATHER_END,
        EventTypeId.FIELD_START,
        EventTypeId.FIELD_END,
        EventTypeId.SIDE_START,
        EventTypeId.SIDE_END,
        EventTypeId.EFFECT_START,
        EventTypeId.EFFECT_END,
        EventTypeId.ABILITY,
        EventTypeId.ABILITY_END,
        EventTypeId.CANT,
        EventTypeId.SINGLEMOVE,
        EventTypeId.PREPARE,
        EventTypeId.BOOST_SET,
    }
)

# Routine numeric deltas: informative but partly recoverable from state tokens.
MEDIUM_PRIORITY_EVENTS = frozenset(
    {
        EventTypeId.BOOST,
        EventTypeId.UNBOOST,
        EventTypeId.DAMAGE,
        EventTypeId.HEAL,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_SWAP,
        EventTypeId.BOOST_INVERT,
        EventTypeId.BOOST_COPY,
        EventTypeId.ACTIVATE,
        EventTypeId.BLOCKED,
    }
)

# Protocol status codes; the status vocab table is keyed by these raw codes.
STATUS_CODES = frozenset({"brn", "frz", "par", "psn", "slp", "tox"})

PROTECT_EFFECTS = (
    "move: Protect",
    "move: Detect",
    "move: Spiky Shield",
    "move: Baneful Bunker",
    "move: King's Shield",
    "move: Obstruct",
)


class EventResolver(Protocol):
    """Vocabulary resolution protocol mapping identifiers to integer IDs."""

    def id_for(self, table: str, name: str | None) -> int:
        """Fetch the exact ID for a name within a vocabulary table."""
        ...

    def effect_id_for(self, table: str, name: str | None) -> int:
        """Fetch the effect ID for a name within a vocabulary table."""
        ...

    def resolve(self, table: str, name: str | None) -> tuple[int, str]:
        """Resolve a name, returning both the ID and the resolution type."""
        ...


_PRE_HP_TAGS = frozenset({"-damage", "-heal"})


def build_raw_event(
    split_message: Sequence[str],
    pre_hp_for: Callable[[str], float | None],
) -> RawBattleEvent:
    """Shared raw-line producer for live capture and replay reconstruction.

    Both producers must snapshot the entity's HP before the line is applied,
    so damage/heal deltas are computed against identical baselines in training
    and replay. Keep every raw-line -> RawBattleEvent rule in this function.
    """
    pre_hp = None
    if len(split_message) > 2 and split_message[1] in _PRE_HP_TAGS:
        pre_hp = pre_hp_for(split_message[2])

    return RawBattleEvent(tuple(split_message), pre_hp)


def get_hp_fraction(hp_status: str) -> float:
    """Extract float HP fraction from a Showdown HP status string."""
    hp_part = hp_status.split(" ", 1)[0]

    if "/" not in hp_part:
        return 0.0

    try:
        num, den_str = hp_part.split("/")
        den_clean = "".join(c for c in den_str if c.isdigit() or c == ".")
        return float(num) / float(den_clean)
    except (ValueError, ZeroDivisionError):
        return 0.0


_priority_list = [0] * EVENT_TYPE_COUNT
for _ev in HIGH_PRIORITY_EVENTS:
    _priority_list[_ev] = 2
for _ev in MEDIUM_PRIORITY_EVENTS:
    _priority_list[_ev] = 1
_PRIORITY_MAP = tuple(_priority_list)


def _sort_priority_key(event: BattleEvent) -> tuple[int, int]:
    """Provide a sort key to rank high-priority events earlier for truncation."""
    return (-_PRIORITY_MAP[event.event_type], event.order)


def _sort_order_key(event: BattleEvent) -> int:
    """Provide a sort key to restore original event order after truncation."""
    return event.order


def truncate_events(events: list[BattleEvent], limit: int = 24) -> list[BattleEvent]:
    """Truncate event sequence to limit while preserving high-priority events."""
    if len(events) <= limit:
        return events

    selected = sorted(events, key=_sort_priority_key)[:limit]
    return sorted(selected, key=_sort_order_key)


def _resolve_id(resolver: EventResolver, table: str, name: str | None) -> int:
    """Resolve an identifier, updating diagnostics upon Out-Of-Vocabulary matches."""
    resolved_id, resolution = resolver.resolve(table, name)
    if resolution == _RESOLUTION_OOV:
        EVENT_DIAGNOSTICS["oov_ids"] += 1
    return resolved_id


def _resolve_effect(resolver: EventResolver, table: str, name: str) -> int:
    """Strip prefixes from effect names and resolve their identifiers."""
    _, separator, remainder = name.partition(":")
    return _resolve_id(resolver, table, remainder if separator else name)


def parse_events(
    raw_events: list[RawBattleEvent],
    resolver: EventResolver,
) -> list[BattleEvent]:
    """Parse raw Showdown protocol event lines into structured BattleEvents.

    Arguments:
      raw_events: sequence of raw battle event line records to parse
      resolver: vocabulary event resolver mapping string identifiers to integers

    Returns:
      list of parsed BattleEvent objects in order of occurrence
    """
    events: list[BattleEvent] = []
    last_attacker: str | None = None

    for raw_event in raw_events:
        message = raw_event.message
        if len(message) < 2:
            continue

        tag = message[1]
        order = len(events)

        if tag == "move" and len(message) >= 4:
            last_attacker = message[2]
            generated = any(part.startswith("[from]") for part in message[5:])
            events.append(
                BattleEvent(
                    EventTypeId.MOVE,
                    last_attacker,
                    target_id=message[4] if len(message) >= 5 else None,
                    move_id=_resolve_id(resolver, "moves", message[3]),
                    flags=4 if generated else 0,
                    order=order,
                )
            )

        elif tag in ("switch", "drag") and len(message) >= 5:
            events.append(
                BattleEvent(
                    EventTypeId.DRAG if tag == "drag" else EventTypeId.SWITCH_IN,
                    message[2],
                    order=order,
                )
            )

        elif tag == "swap" and len(message) >= 3:
            events.append(BattleEvent(EventTypeId.SWAP, message[2], order=order))

        elif tag == "faint" and len(message) >= 3:
            events.append(BattleEvent(EventTypeId.FAINT, message[2], order=order))

        elif tag in ("-damage", "-heal") and len(message) >= 4:
            new_hp = get_hp_fraction(message[3])
            if raw_event.pre_hp is None:
                EVENT_DIAGNOSTICS["missing_pre_hp"] += 1
            value = 0.0 if raw_event.pre_hp is None else new_hp - raw_event.pre_hp
            events.append(
                BattleEvent(
                    EventTypeId.DAMAGE if tag == "-damage" else EventTypeId.HEAL,
                    message[2],
                    value=value,
                    order=order,
                )
            )

        elif tag in ("-boost", "-unboost") and len(message) >= 5:
            amount = int(message[4]) / 6.0
            events.append(
                BattleEvent(
                    EventTypeId.BOOST if tag == "-boost" else EventTypeId.UNBOOST,
                    message[2],
                    value=amount if tag == "-boost" else -amount,
                    order=order,
                )
            )

        elif tag in ("-status", "-curestatus") and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.STATUS_SET if tag == "-status" else EventTypeId.STATUS_CURE,
                    message[2],
                    status_id=_resolve_id(resolver, "status", message[3]),
                    order=order,
                )
            )

        elif tag in ("-enditem", "-item") and len(message) >= 4:
            transferred = tag == "-item" and any(
                "move: trick" in part.lower() or "move: switcheroo" in part.lower()
                for part in message[4:]
            )
            event_type = (
                EventTypeId.ITEM_TRANSFER
                if transferred
                else EventTypeId.ITEM_END
                if tag == "-enditem"
                else EventTypeId.ITEM_REVEAL
            )
            events.append(
                BattleEvent(
                    event_type,
                    message[2],
                    item_id=_resolve_id(resolver, "items", message[3]),
                    order=order,
                )
            )

        elif tag == "-ability" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.ABILITY,
                    message[2],
                    ability_id=_resolve_id(resolver, "abilities", message[3]),
                    order=order,
                )
            )

        elif tag == "-weather" and len(message) >= 3:
            if any("[upkeep]" in part for part in message[3:]):
                continue
            if message[2] == "none":
                events.append(BattleEvent(EventTypeId.WEATHER_END, None, order=order))
            else:
                events.append(
                    BattleEvent(
                        EventTypeId.WEATHER_START,
                        None,
                        effect_id=_resolve_effect(resolver, "weathers", message[2]),
                        order=order,
                    )
                )

        elif tag in ("-fieldstart", "-fieldend") and len(message) >= 3:
            events.append(
                BattleEvent(
                    EventTypeId.FIELD_START if tag == "-fieldstart" else EventTypeId.FIELD_END,
                    None,
                    effect_id=_resolve_effect(resolver, "fields", message[2]),
                    order=order,
                )
            )

        elif tag in ("-sidestart", "-sideend") and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.SIDE_START if tag == "-sidestart" else EventTypeId.SIDE_END,
                    message[2],
                    effect_id=_resolve_effect(resolver, "side_conditions", message[3]),
                    order=order,
                )
            )

        elif tag in ("-start", "-end") and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.EFFECT_START if tag == "-start" else EventTypeId.EFFECT_END,
                    message[2],
                    effect_id=_resolve_effect(resolver, "volatiles", message[3]),
                    order=order,
                )
            )

        elif tag in ("-formechange", "detailschange") and len(message) >= 4:
            events.append(BattleEvent(EventTypeId.FORME_CHANGE, message[2], order=order))

        elif tag == "-fail":
            events.append(
                BattleEvent(
                    EventTypeId.FAILED,
                    last_attacker or (message[2] if len(message) >= 3 else None),
                    order=order,
                )
            )

        elif tag in ("-immune", "-miss"):
            named = message[2] if len(message) >= 3 else None
            if tag == "-miss":
                source = named or last_attacker
                target = message[3] if len(message) >= 4 else None
                flags = 2
            else:
                source = last_attacker
                target = named
                flags = 1
            events.append(
                BattleEvent(
                    EventTypeId.BLOCKED,
                    source,
                    target_id=target,
                    flags=flags,
                    order=order,
                )
            )

        elif tag == "-activate" and len(message) >= 4:
            effect = message[3]
            if effect.startswith(PROTECT_EFFECTS):
                events.append(
                    BattleEvent(
                        EventTypeId.BLOCKED,
                        last_attacker,
                        target_id=message[2],
                        order=order,
                    )
                )
            else:
                kind, separator, name = effect.partition(":")
                kind = kind.strip().lower() if separator else ""
                ability_id = 0
                item_id = 0
                effect_id = 0
                if kind == "ability":
                    ability_id = _resolve_id(resolver, "abilities", name)
                elif kind == "item":
                    item_id = _resolve_id(resolver, "items", name)
                else:
                    effect_id = _resolve_effect(resolver, "volatiles", effect)

                events.append(
                    BattleEvent(
                        EventTypeId.ACTIVATE,
                        message[2],
                        ability_id=ability_id,
                        item_id=item_id,
                        effect_id=effect_id,
                        order=order,
                    )
                )

        elif tag == "-crit" and len(message) >= 3:
            events.append(BattleEvent(EventTypeId.CRIT, message[2], order=order))

        elif tag == "-mega" and len(message) >= 3:
            events.append(BattleEvent(EventTypeId.MEGA, message[2], order=order))

        elif tag == "cant" and len(message) >= 4:
            reason = message[3]
            is_status = reason in STATUS_CODES
            events.append(
                BattleEvent(
                    EventTypeId.CANT,
                    message[2],
                    status_id=_resolve_id(resolver, "status", reason) if is_status else 0,
                    effect_id=0 if is_status else _resolve_effect(resolver, "volatiles", reason),
                    move_id=_resolve_id(resolver, "moves", message[4]) if len(message) >= 5 else 0,
                    order=order,
                )
            )

        elif tag == "-prepare" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.PREPARE,
                    message[2],
                    target_id=message[4] if len(message) >= 5 else None,
                    move_id=_resolve_id(resolver, "moves", message[3]),
                    order=order,
                )
            )

        elif tag == "-singlemove" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.SINGLEMOVE,
                    message[2],
                    effect_id=_resolve_effect(resolver, "volatiles", message[3]),
                    order=order,
                )
            )

        elif tag == "-setboost" and len(message) >= 5:
            events.append(
                BattleEvent(
                    EventTypeId.BOOST_SET,
                    message[2],
                    value=int(message[4]) / 6.0,
                    order=order,
                )
            )

        elif tag in ("-clearboost", "-clearnegativeboost", "-clearallboost"):
            events.append(
                BattleEvent(
                    EventTypeId.BOOST_CLEAR,
                    message[2] if len(message) >= 3 else None,
                    flags=1 if tag == "-clearnegativeboost" else 0,
                    order=order,
                )
            )

        elif tag == "-swapboost" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.BOOST_SWAP,
                    message[2],
                    target_id=message[3],
                    order=order,
                )
            )

        elif tag == "-invertboost" and len(message) >= 3:
            events.append(BattleEvent(EventTypeId.BOOST_INVERT, message[2], order=order))

        elif tag == "-copyboost" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.BOOST_COPY,
                    message[2],
                    target_id=message[3],
                    order=order,
                )
            )

        elif tag == "-transform" and len(message) >= 4:
            events.append(
                BattleEvent(
                    EventTypeId.TRANSFORM,
                    message[2],
                    target_id=message[3],
                    order=order,
                )
            )

        elif tag == "-endability" and len(message) >= 3:
            events.append(
                BattleEvent(
                    EventTypeId.ABILITY_END,
                    message[2],
                    ability_id=_resolve_id(resolver, "abilities", message[3])
                    if len(message) >= 4
                    else 0,
                    order=order,
                )
            )

        elif tag == "-fieldactivate" and len(message) >= 3:
            events.append(
                BattleEvent(
                    EventTypeId.FIELD_ACTIVATE,
                    None,
                    effect_id=_resolve_effect(resolver, "fields", message[2]),
                    order=order,
                )
            )

        elif tag == "-notarget":
            events.append(
                BattleEvent(
                    EventTypeId.NO_TARGET,
                    message[2] if len(message) >= 3 else last_attacker,
                    order=order,
                )
            )

    return events


__all__ = [
    "EVENT_DIAGNOSTICS",
    "EVENT_TYPE_COUNT",
    "BattleEvent",
    "EventTypeId",
    "RawBattleEvent",
    "build_raw_event",
    "parse_events",
    "truncate_events",
]
