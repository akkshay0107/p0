from p0.model.event_builder import (
    BattleEvent,
    EventCollector,
    EventTypeId,
    RawBattleEvent,
)


class MockBattle:
    pass


def test_parse_events_returns_typed_events_in_protocol_order():
    raw_events = [
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"), pre_hp=1.0),
        RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
        RawBattleEvent(("", "faint", "p2a: Charizard")),
    ]

    events = EventCollector.parse_events(raw_events)

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
    events = EventCollector.parse_events(
        [RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"))]
    )

    assert events[0].value == 0.0


def test_parse_events_distinguishes_failed_and_blocked_moves():
    events = EventCollector.parse_events(
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

    selected = EventCollector.truncate_events(events)

    assert len(selected) == 24
    assert [event.order for event in selected] == [*range(23), 24]
    assert selected[-1].event_type == EventTypeId.MOVE


def test_consume_events_clears_buffer_immediately():
    battle = MockBattle()
    EventCollector.set_raw_events(
        battle,
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"))],
    )

    first = EventCollector.consume_events(battle)  # type: ignore
    second = EventCollector.consume_events(battle)  # type: ignore

    assert len(first) == 1
    assert second == []
