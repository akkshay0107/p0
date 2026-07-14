import logging

from poke_env.battle import DoubleBattle

from p0.battle.events import (
    BattleEvent,
    EventTypeId,
    RawBattleEvent,
    truncate_events,
)
from p0.battle.events import (
    parse_events as parse_protocol_events,
)
from p0.model.tokenizer import tokenizer
from p0.runtime.live_event_capture import consume_raw_events, set_raw_events


def parse_events(raw_events: list[RawBattleEvent]) -> list[BattleEvent]:
    return parse_protocol_events(raw_events, tokenizer)


def test_parse_events_returns_typed_events_in_protocol_order():
    raw_events = [
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"), pre_hp=1.0),
        RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
        RawBattleEvent(("", "faint", "p2a: Charizard")),
    ]

    events = parse_events(raw_events)

    assert all(isinstance(event, BattleEvent) for event in events)
    assert [event.event_type for event in events] == [
        EventTypeId.MOVE,
        EventTypeId.DAMAGE,
        EventTypeId.STATUS_SET,
        EventTypeId.FAINT,
    ]
    assert [event.order for event in events] == [0, 1, 2, 3]
    assert events[0].entity_id == "p1a: Pikachu"
    assert events[1].entity_id == "p2a: Charizard"
    assert events[1].value == -0.25


def test_parse_events_uses_zero_when_previous_hp_is_unknown():
    events = parse_events([RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"))])

    assert events[0].value == 0.0


def test_parse_events_distinguishes_failed_and_blocked_moves():
    events = parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "-immune", "p2a: Charizard")),
            RawBattleEvent(("", "-fail", "p1a: Pikachu")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.MOVE,
        EventTypeId.BLOCKED,
        EventTypeId.FAILED,
    ]


def test_truncate_events_keeps_priority_and_original_order():
    events = [BattleEvent(EventTypeId.DAMAGE, "p1a: Pikachu", order=order) for order in range(24)]
    events.append(BattleEvent(EventTypeId.MOVE, "p2a: Charizard", order=24))

    selected = truncate_events(events)

    assert len(selected) == 24
    assert [event.order for event in selected] == [*range(23), 24]
    assert selected[-1].event_type == EventTypeId.MOVE


def test_consume_events_clears_buffer_immediately():
    battle = DoubleBattle("events", "player", logging.getLogger(__name__), 9)
    set_raw_events(
        battle,
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"))],
    )

    first = parse_events(consume_raw_events(battle))
    second = parse_events(consume_raw_events(battle))

    assert len(first) == 1
    assert second == []
