"""Pure protocol-event values and parser entry point."""

from __future__ import annotations

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


EVENT_TYPE_COUNT = max(EventTypeId) + 1


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


HIGH_PRIORITY_EVENTS = frozenset(
    {
        EventTypeId.MOVE,
        EventTypeId.SWITCH_IN,
        EventTypeId.FAINT,
        EventTypeId.ITEM_END,
        EventTypeId.STATUS_SET,
        EventTypeId.STATUS_CURE,
    }
)
MEDIUM_PRIORITY_EVENTS = frozenset({EventTypeId.BOOST, EventTypeId.UNBOOST, EventTypeId.DAMAGE})
STATUS_NAMES = {
    "brn": "burn",
    "frz": "freeze",
    "par": "paralysis",
    "psn": "poison",
    "slp": "sleep",
    "tox": "toxic",
}
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
                    move_id=resolver.id_for("moves", message[3]),
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
                    status_id=resolver.id_for("status", STATUS_NAMES.get(message[3], message[3])),
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
                    item_id=resolver.id_for("items", message[3]),
                )
            )
        elif tag == "-ability" and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.ABILITY,
                    message[2],
                    order,
                    ability_id=resolver.id_for("abilities", message[3]),
                )
            )
        elif tag == "-weather" and len(message) >= 3:
            if message[2] == "none":
                events.append(event(EventTypeId.WEATHER_END, None, order))
            else:
                events.append(
                    event(
                        EventTypeId.WEATHER_START,
                        None,
                        order,
                        effect_id=resolver.effect_id_for("weathers", message[2]),
                    )
                )
        elif tag in ("-fieldstart", "-fieldend") and len(message) >= 3:
            events.append(
                event(
                    EventTypeId.FIELD_START if tag == "-fieldstart" else EventTypeId.FIELD_END,
                    None,
                    order,
                    effect_id=resolver.effect_id_for("fields", message[2]),
                )
            )
        elif tag in ("-sidestart", "-sideend") and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.SIDE_START if tag == "-sidestart" else EventTypeId.SIDE_END,
                    message[2],
                    order,
                    effect_id=resolver.effect_id_for("side_conditions", message[3]),
                )
            )
        elif tag in ("-start", "-end") and len(message) >= 4:
            events.append(
                event(
                    EventTypeId.EFFECT_START if tag == "-start" else EventTypeId.EFFECT_END,
                    message[2],
                    order,
                    effect_id=resolver.effect_id_for("volatiles", message[3]),
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
            events.append(
                event(
                    EventTypeId.BLOCKED,
                    last_attacker or (message[2] if len(message) >= 3 else None),
                    order,
                    flags=1 if tag == "-immune" else 2,
                )
            )
        elif tag == "-activate" and len(message) >= 4:
            if message[3].startswith(PROTECT_EFFECTS):
                events.append(event(EventTypeId.BLOCKED, last_attacker or message[2], order))
        elif tag == "-crit" and len(message) >= 3:
            events.append(event(EventTypeId.CRIT, message[2], order))
        elif tag == "-mega" and len(message) >= 3:
            events.append(event(EventTypeId.MEGA, message[2], order))
    return events


__all__ = [
    "EVENT_TYPE_COUNT",
    "BattleEvent",
    "EventTypeId",
    "RawBattleEvent",
    "parse_events",
    "truncate_events",
]
