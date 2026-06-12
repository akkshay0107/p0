from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple
from weakref import WeakKeyDictionary, finalize

from poke_env.battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon

from src.model.structured_observation import EVENT_COUNT
from src.model.tokenizer import tokenizer


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


EVENT_TYPE_COUNT = max(EventTypeId) + 1


class RawBattleEvent(NamedTuple):
    message: tuple[str, ...]
    pre_hp: float | None = None


class BattleEvent(NamedTuple):
    event_type: EventTypeId
    entity_id: str | None
    move_id: int = 0
    item_id: int = 0
    status_id: int = 0
    value: float = 0.0
    order: int = 0


STATUS_NAMES = {
    "brn": "burn",
    "frz": "freeze",
    "par": "paralysis",
    "psn": "poison",
    "slp": "sleep",
    "tox": "toxic",
}

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
MEDIUM_PRIORITY_EVENTS = frozenset(
    {
        EventTypeId.BOOST,
        EventTypeId.UNBOOST,
        EventTypeId.DAMAGE,
    }
)
PROTECT_EFFECTS = (
    "move: Protect",
    "move: Detect",
    "move: Spiky Shield",
    "move: Baneful Bunker",
    "move: King's Shield",
    "move: Obstruct",
)

_raw_event_buffers: WeakKeyDictionary[object, list[RawBattleEvent]] = WeakKeyDictionary()
_pokemon_last_moves: dict[int, str] = {}
_battle_pokemon_ids: dict[int, set[int]] = {}
_registered_battles: set[int] = set()


def _cleanup_battle(battle_id: int) -> None:
    _registered_battles.discard(battle_id)
    pokemon_ids = _battle_pokemon_ids.pop(battle_id, None)
    if pokemon_ids:
        for pid in pokemon_ids:
            _pokemon_last_moves.pop(pid, None)


def _get_pokemon_safely(battle: DoubleBattle, identifier: str) -> Pokemon | None:
    try:
        return battle.get_pokemon(identifier)
    except (AssertionError, IndexError, KeyError, ValueError):
        return None


def _clear_last_move(pokemon: Pokemon) -> None:
    _pokemon_last_moves.pop(id(pokemon), None)


def _get_last_move(pokemon: Pokemon) -> str | None:
    return _pokemon_last_moves.get(id(pokemon), None)


def _set_last_move(battle_id: int, pokemon_id: int, move: str) -> None:
    _pokemon_last_moves[pokemon_id] = move
    _battle_pokemon_ids.setdefault(battle_id, set()).add(pokemon_id)


_original_switch_out = Pokemon.switch_out


def _patched_switch_out(self: Pokemon, fields):
    _clear_last_move(self)
    _original_switch_out(self, fields)


Pokemon.switch_out = _patched_switch_out

_original_parse_message = DoubleBattle.parse_message


def _patched_parse_message(self: DoubleBattle, split_message: list[str]):
    if id(self) not in _registered_battles:
        _registered_battles.add(id(self))
        finalize(self, _cleanup_battle, id(self))

    raw_events = _raw_event_buffers.setdefault(self, [])

    pre_hp = None
    if len(split_message) > 2 and split_message[1] in ("-damage", "-heal"):
        pokemon = _get_pokemon_safely(self, split_message[2])
        if pokemon is not None:
            pre_hp = pokemon.current_hp_fraction

    if len(split_message) > 3 and split_message[1] == "move":
        pokemon = _get_pokemon_safely(self, split_message[2])
        if pokemon is not None:
            _set_last_move(id(self), id(pokemon), split_message[3])

    raw_events.append(RawBattleEvent(tuple(split_message), pre_hp))
    _original_parse_message(self, split_message)


DoubleBattle.parse_message = _patched_parse_message


class EventCollector:
    @staticmethod
    def last_move(pokemon: Pokemon) -> str | None:
        return _get_last_move(pokemon)

    @staticmethod
    def set_raw_events(battle: object, raw_events: list[RawBattleEvent]) -> None:
        _raw_event_buffers[battle] = raw_events

    @staticmethod
    def get_hp_fraction(hp_status: str) -> float:
        hp_status = hp_status.split()[0]
        if hp_status == "0":
            return 0.0
        if "/" not in hp_status:
            return 0.0
        try:
            numerator, denominator = hp_status.split("/")
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _priority(event: BattleEvent) -> int:
        if event.event_type in HIGH_PRIORITY_EVENTS:
            return 2
        if event.event_type in MEDIUM_PRIORITY_EVENTS:
            return 1
        return 0

    @classmethod
    def truncate_events(
        cls,
        events: list[BattleEvent],
        limit: int = EVENT_COUNT,
    ) -> list[BattleEvent]:
        if len(events) <= limit:
            return events
        selected = sorted(events, key=lambda event: (-cls._priority(event), event.order))[:limit]
        return sorted(selected, key=lambda event: event.order)

    @staticmethod
    def _status_id(status: str) -> int:
        status_name = STATUS_NAMES.get(status, status)
        return tokenizer.id_for("status", status_name)

    @staticmethod
    def _event(
        event_type: EventTypeId,
        entity_id: str | None,
        order: int,
        *,
        move_id: int = 0,
        item_id: int = 0,
        status_id: int = 0,
        value: float = 0.0,
    ) -> BattleEvent:
        return BattleEvent(
            event_type=event_type,
            entity_id=entity_id,
            move_id=move_id,
            item_id=item_id,
            status_id=status_id,
            value=value,
            order=order,
        )

    @classmethod
    def consume_events(cls, battle: object) -> list[BattleEvent]:
        raw_events = _raw_event_buffers.pop(battle, [])
        return cls.parse_events(raw_events)

    @classmethod
    def parse_events(cls, raw_events: list[RawBattleEvent]) -> list[BattleEvent]:
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
                events.append(
                    cls._event(
                        EventTypeId.MOVE,
                        last_attacker,
                        order,
                        move_id=tokenizer.id_for("moves", message[3]),
                    )
                )
            elif tag in ("switch", "drag") and len(message) >= 5:
                events.append(cls._event(EventTypeId.SWITCH_IN, message[2], order))
            elif tag == "faint" and len(message) >= 3:
                events.append(cls._event(EventTypeId.FAINT, message[2], order))
            elif tag in ("-damage", "-heal") and len(message) >= 4:
                new_hp = cls.get_hp_fraction(message[3])
                if raw_event.pre_hp is None:
                    value = 0.0
                else:
                    value = new_hp - raw_event.pre_hp
                event_type = EventTypeId.DAMAGE if tag == "-damage" else EventTypeId.HEAL
                events.append(cls._event(event_type, message[2], order, value=value))
            elif tag in ("-boost", "-unboost") and len(message) >= 5:
                amount = int(message[4]) / 6.0
                if tag == "-unboost":
                    amount = -amount
                event_type = EventTypeId.BOOST if tag == "-boost" else EventTypeId.UNBOOST
                events.append(cls._event(event_type, message[2], order, value=amount))
            elif tag in ("-status", "-curestatus") and len(message) >= 4:
                event_type = EventTypeId.STATUS_SET if tag == "-status" else EventTypeId.STATUS_CURE
                events.append(
                    cls._event(
                        event_type,
                        message[2],
                        order,
                        status_id=cls._status_id(message[3]),
                    )
                )
            elif tag in ("-enditem", "-item") and len(message) >= 4:
                event_type = EventTypeId.ITEM_END if tag == "-enditem" else EventTypeId.ITEM_REVEAL
                events.append(
                    cls._event(
                        event_type,
                        message[2],
                        order,
                        item_id=tokenizer.id_for("items", message[3]),
                    )
                )
            elif tag == "-weather" and len(message) >= 3 and message[2] != "none":
                events.append(cls._event(EventTypeId.WEATHER_START, None, order))
            elif tag == "-fieldstart" and len(message) >= 3:
                events.append(cls._event(EventTypeId.FIELD_START, None, order))
            elif tag == "-sidestart" and len(message) >= 4:
                events.append(cls._event(EventTypeId.SIDE_START, message[2], order))
            elif tag == "-fail":
                entity_id = last_attacker or (message[2] if len(message) >= 3 else None)
                events.append(cls._event(EventTypeId.FAILED, entity_id, order))
            elif tag in ("-immune", "-miss"):
                entity_id = last_attacker or (message[2] if len(message) >= 3 else None)
                events.append(cls._event(EventTypeId.BLOCKED, entity_id, order))
            elif tag == "-activate" and len(message) >= 4:
                if message[3].startswith(PROTECT_EFFECTS):
                    events.append(
                        cls._event(EventTypeId.BLOCKED, last_attacker or message[2], order)
                    )
            elif tag == "-crit" and len(message) >= 3:
                events.append(cls._event(EventTypeId.CRIT, message[2], order))
            elif tag == "-mega" and len(message) >= 3:
                events.append(cls._event(EventTypeId.MEGA, message[2], order))

        return events
