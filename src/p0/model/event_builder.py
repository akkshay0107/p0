"""Temporary compatibility facade for the pure event parser and live capture."""

from p0.battle.events import (
    EVENT_TYPE_COUNT,
    BattleEvent,
    EventTypeId,
    ProtocolEventParser,
    RawBattleEvent,
)
from p0.model.tokenizer import tokenizer

__all__ = [
    "EVENT_TYPE_COUNT",
    "BattleEvent",
    "EventCollector",
    "EventTypeId",
    "ProtocolEventParser",
    "RawBattleEvent",
]


class EventCollector:
    @staticmethod
    def last_move(pokemon: object) -> str | None:
        from p0.runtime.live_event_capture import last_move

        return last_move(pokemon)

    @staticmethod
    def set_raw_events(battle: object, raw_events: list[RawBattleEvent]) -> None:
        from p0.runtime.live_event_capture import set_raw_events

        set_raw_events(battle, raw_events)

    get_hp_fraction = staticmethod(ProtocolEventParser.get_hp_fraction)

    @staticmethod
    def truncate_events(events: list[BattleEvent], limit: int = 24) -> list[BattleEvent]:
        return ProtocolEventParser.truncate_events(events, limit)

    @staticmethod
    def consume_events(battle: object) -> list[BattleEvent]:
        from p0.runtime.live_event_capture import consume_raw_events

        return EventCollector.parse_events(consume_raw_events(battle))

    @staticmethod
    def parse_events(raw_events: list[RawBattleEvent]) -> list[BattleEvent]:
        return ProtocolEventParser.parse_events(raw_events, tokenizer)
