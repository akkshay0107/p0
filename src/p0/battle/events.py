"""Pure protocol-event values and parser entry point."""

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


# Counters for silent event-pipeline degradations: "oov_ids", "missing_pre_hp",
# "grounding_misses". Reset with .clear().
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
    def id_for(self, table: str, name: str | None) -> int: ...

    def effect_id_for(self, table: str, name: str | None) -> int: ...

    def resolve(self, table: str, name: str | None) -> tuple[int, str]: ...


_PRE_HP_TAGS = frozenset({"-damage", "-heal"})


def build_raw_event(
    split_message: Sequence[str],
    pre_hp_for: Callable[[str], float | None],
) -> RawBattleEvent:
    """Shared raw-line producer for live capture and replay reconstruction.

    Both producers must snapshot the entity's HP *before* the line is applied,
    so damage/heal deltas are computed against identical baselines in training
    and replay. Keep every raw-line -> RawBattleEvent rule in this function.
    """
    pre_hp = None
    if len(split_message) > 2 and split_message[1] in _PRE_HP_TAGS:
        pre_hp = pre_hp_for(split_message[2])
    return RawBattleEvent(tuple(split_message), pre_hp)


def get_hp_fraction(hp_status: str) -> float:
    hp_status = hp_status.split()[0]
    if hp_status == "0" or "/" not in hp_status:
        return 0.0
    try:
        numerator, denominator = hp_status.split("/")
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _event_priority(event: BattleEvent) -> int:
    if event.event_type in HIGH_PRIORITY_EVENTS:
        return 2
    if event.event_type in MEDIUM_PRIORITY_EVENTS:
        return 1
    return 0


def truncate_events(events: list[BattleEvent], limit: int = 24) -> list[BattleEvent]:
    if len(events) <= limit:
        return events
    selected = sorted(events, key=lambda event: (-_event_priority(event), event.order))[:limit]
    return sorted(selected, key=lambda event: event.order)


def parse_events(
    raw_events: list[RawBattleEvent],
    resolver: EventResolver,
) -> list[BattleEvent]:
    events: list[BattleEvent] = []
    last_attacker: str | None = None

    def event(
        event_type: EventTypeId,
        entity_id: str | None,
        order: int,
        *,
        target_id: str | None = None,
        move_id: int = 0,
        item_id: int = 0,
        status_id: int = 0,
        effect_id: int = 0,
        ability_id: int = 0,
        flags: int = 0,
        value: float = 0.0,
    ) -> BattleEvent:
        return BattleEvent(
            event_type,
            entity_id,
            target_id,
            move_id,
            item_id,
            status_id,
            effect_id,
            ability_id,
            flags,
            value,
            order,
        )

    def resolve_id(table: str, name: str | None) -> int:
        resolved_id, resolution = resolver.resolve(table, name)
        if resolution == _RESOLUTION_OOV:
            EVENT_DIAGNOSTICS["oov_ids"] += 1
        return resolved_id

    def resolve_effect(table: str, name: str) -> int:
        _, separator, remainder = name.partition(":")
        return resolve_id(table, remainder if separator else name)

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
                event(
                    EventTypeId.MOVE,
                    last_attacker,
                    order,
                    target_id=message[4] if len(message) >= 5 else None,
                    move_id=resolve_id("moves", message[3]),
                    flags=4 if generated else 0,
                )
            )
        elif tag in ("switch", "drag") and len(message) >= 5:
            events.append(
                event(
                    EventTypeId.DRAG if tag == "drag" else EventTypeId.SWITCH_IN,
                    message[2],
                    order,
                )
            )
        elif tag == "swap" and len(message) >= 3:
            events.append(event(EventTypeId.SWAP, message[2], order))
        elif tag == "faint" and len(message) >= 3:
            events.append(event(EventTypeId.FAINT, message[2], order))
        elif tag in ("-damage", "-heal") and len(message) >= 4:
            new_hp = get_hp_fraction(message[3])
            if raw_event.pre_hp is None:
                EVENT_DIAGNOSTICS["missing_pre_hp"] += 1
            value = 0.0 if raw_event.pre_hp is None else new_hp - raw_event.pre_hp
            events.append(
                event(
                    EventTypeId.DAMAGE if tag == "-damage" else EventTypeId.HEAL,
                    message[2],
                    order,
                    value=value,
                )
            )
        elif tag in ("-boost", "-unboost") and len(message) >= 5:
            amount = int(message[4]) / 6.0
            events.append(
                event(
                    EventTypeId.BOOST if tag == "-boost" else EventTypeId.UNBOOST,
                    message[2],
                    order,
                    value=amount if tag == "-boost" else -amount,
                )
            )
        elif tag in ("-status", "-curestatus") and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.STATUS_SET if tag == "-status" else EventTypeId.STATUS_CURE,
                    message[2],
                    order,
                    status_id=resolve_id("status", message[3]),
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
                event(
                    event_type,
                    message[2],
                    order,
                    item_id=resolve_id("items", message[3]),
                )
            )
        elif tag == "-ability" and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.ABILITY,
                    message[2],
                    order,
                    ability_id=resolve_id("abilities", message[3]),
                )
            )
        elif tag == "-weather" and len(message) >= 3:
            if any("[upkeep]" in part for part in message[3:]):
                # Showdown re-emits the active weather every turn; an upkeep
                # line is not a state transition and must not burn a slot.
                continue
            if message[2] == "none":
                events.append(event(EventTypeId.WEATHER_END, None, order))
            else:
                events.append(
                    event(
                        EventTypeId.WEATHER_START,
                        None,
                        order,
                        effect_id=resolve_effect("weathers", message[2]),
                    )
                )
        elif tag in ("-fieldstart", "-fieldend") and len(message) >= 3:
            events.append(
                event(
                    EventTypeId.FIELD_START if tag == "-fieldstart" else EventTypeId.FIELD_END,
                    None,
                    order,
                    effect_id=resolve_effect("fields", message[2]),
                )
            )
        elif tag in ("-sidestart", "-sideend") and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.SIDE_START if tag == "-sidestart" else EventTypeId.SIDE_END,
                    message[2],
                    order,
                    effect_id=resolve_effect("side_conditions", message[3]),
                )
            )
        elif tag in ("-start", "-end") and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.EFFECT_START if tag == "-start" else EventTypeId.EFFECT_END,
                    message[2],
                    order,
                    effect_id=resolve_effect("volatiles", message[3]),
                )
            )
        elif tag in ("-formechange", "detailschange") and len(message) >= 4:
            events.append(event(EventTypeId.FORME_CHANGE, message[2], order))
        elif tag == "-fail":
            events.append(
                event(
                    EventTypeId.FAILED,
                    last_attacker or (message[2] if len(message) >= 3 else None),
                    order,
                )
            )
        elif tag in ("-immune", "-miss"):
            # -immune names the immune (blocked-on) Pokemon; -miss names the
            # source and, when present, the missed target. Keep both endpoints.
            named = message[2] if len(message) >= 3 else None
            if tag == "-miss":
                source = named or last_attacker
                target = message[3] if len(message) >= 4 else None
                flags = 2
            else:
                source = last_attacker
                target = named
                flags = 1
            events.append(event(EventTypeId.BLOCKED, source, order, target_id=target, flags=flags))
        elif tag == "-activate" and len(message) >= 4:
            effect = message[3]
            if effect.startswith(PROTECT_EFFECTS):
                events.append(
                    event(EventTypeId.BLOCKED, last_attacker, order, target_id=message[2])
                )
            else:
                kind, separator, name = effect.partition(":")
                kind = kind.strip().lower() if separator else ""
                activation = event(EventTypeId.ACTIVATE, message[2], order)
                if kind == "ability":
                    activation = activation._replace(ability_id=resolve_id("abilities", name))
                elif kind == "item":
                    activation = activation._replace(item_id=resolve_id("items", name))
                else:
                    activation = activation._replace(effect_id=resolve_effect("volatiles", effect))
                events.append(activation)
        elif tag == "-crit" and len(message) >= 3:
            events.append(event(EventTypeId.CRIT, message[2], order))
        elif tag == "-mega" and len(message) >= 3:
            events.append(event(EventTypeId.MEGA, message[2], order))
        elif tag == "cant" and len(message) >= 4:
            # "Fully paralyzed / flinched / asleep / taunted out of the move".
            # Flinch has no -start/-end line, so this is its only representation.
            reason = message[3]
            is_status = reason in STATUS_CODES
            events.append(
                event(
                    EventTypeId.CANT,
                    message[2],
                    order,
                    status_id=resolve_id("status", reason) if is_status else 0,
                    effect_id=0 if is_status else resolve_effect("volatiles", reason),
                    move_id=resolve_id("moves", message[4]) if len(message) >= 5 else 0,
                )
            )
        elif tag == "-prepare" and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.PREPARE,
                    message[2],
                    order,
                    target_id=message[4] if len(message) >= 5 else None,
                    move_id=resolve_id("moves", message[3]),
                )
            )
        elif tag == "-singlemove" and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.SINGLEMOVE,
                    message[2],
                    order,
                    effect_id=resolve_effect("volatiles", message[3]),
                )
            )
        elif tag == "-setboost" and len(message) >= 5:
            events.append(
                event(EventTypeId.BOOST_SET, message[2], order, value=int(message[4]) / 6.0)
            )
        elif tag in ("-clearboost", "-clearnegativeboost", "-clearallboost"):
            events.append(
                event(
                    EventTypeId.BOOST_CLEAR,
                    message[2] if len(message) >= 3 else None,
                    order,
                    flags=1 if tag == "-clearnegativeboost" else 0,
                )
            )
        elif tag == "-swapboost" and len(message) >= 4:
            events.append(event(EventTypeId.BOOST_SWAP, message[2], order, target_id=message[3]))
        elif tag == "-invertboost" and len(message) >= 3:
            events.append(event(EventTypeId.BOOST_INVERT, message[2], order))
        elif tag == "-copyboost" and len(message) >= 4:
            events.append(event(EventTypeId.BOOST_COPY, message[2], order, target_id=message[3]))
        elif tag == "-transform" and len(message) >= 4:
            events.append(event(EventTypeId.TRANSFORM, message[2], order, target_id=message[3]))
        elif tag == "-endability" and len(message) >= 3:
            events.append(
                event(
                    EventTypeId.ABILITY_END,
                    message[2],
                    order,
                    ability_id=resolve_id("abilities", message[3]) if len(message) >= 4 else 0,
                )
            )
        elif tag == "-fieldactivate" and len(message) >= 3:
            events.append(
                event(
                    EventTypeId.FIELD_ACTIVATE,
                    None,
                    order,
                    effect_id=resolve_effect("fields", message[2]),
                )
            )
        elif tag == "-notarget":
            events.append(
                event(
                    EventTypeId.NO_TARGET,
                    message[2] if len(message) >= 3 else last_attacker,
                    order,
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
