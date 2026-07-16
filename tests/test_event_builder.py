import logging

from poke_env.battle import DoubleBattle

from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    BattleEvent,
    EventTypeId,
    RawBattleEvent,
    build_raw_event,
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


def test_status_codes_resolve_against_vocab():
    events = parse_events(
        [
            RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
            RawBattleEvent(("", "-curestatus", "p2a: Charizard", "par")),
        ]
    )

    assert events[0].status_id == tokenizer.id_for("status", "par") > 0
    assert events[1].status_id == events[0].status_id


def test_cant_prepare_and_singlemove():
    events = parse_events(
        [
            RawBattleEvent(("", "cant", "p1a: Pikachu", "flinch")),
            RawBattleEvent(("", "cant", "p1a: Pikachu", "slp", "Thunderbolt")),
            RawBattleEvent(("", "-prepare", "p1a: Charizard", "Fly", "p2a: Venusaur")),
            RawBattleEvent(("", "-singlemove", "p1a: Gengar", "Destiny Bond")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.CANT,
        EventTypeId.CANT,
        EventTypeId.PREPARE,
        EventTypeId.SINGLEMOVE,
    ]
    assert events[0].effect_id == tokenizer.id_for("volatiles", "flinch") > 0
    assert events[1].status_id == tokenizer.id_for("status", "slp") > 0
    assert events[1].move_id == tokenizer.id_for("moves", "thunderbolt") > 0
    assert events[2].move_id == tokenizer.id_for("moves", "fly") > 0
    assert events[2].target_id == "p2a: Venusaur"
    assert events[3].effect_id == tokenizer.id_for("volatiles", "destinybond") > 0


def test_boost_manipulation_family():
    events = parse_events(
        [
            RawBattleEvent(
                ("", "-setboost", "p1a: Azumarill", "atk", "6", "[from] move: Belly Drum")
            ),
            RawBattleEvent(("", "-clearboost", "p1a: Azumarill")),
            RawBattleEvent(("", "-clearnegativeboost", "p1a: Azumarill")),
            RawBattleEvent(("", "-clearallboost")),
            RawBattleEvent(("", "-swapboost", "p1a: Malamar", "p2a: Incineroar", "atk, def")),
            RawBattleEvent(("", "-invertboost", "p2a: Incineroar")),
            RawBattleEvent(("", "-copyboost", "p1a: Ditto", "p2a: Dragapult")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.BOOST_SET,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_SWAP,
        EventTypeId.BOOST_INVERT,
        EventTypeId.BOOST_COPY,
    ]
    assert events[0].value == 1.0
    assert events[1].flags == 0
    assert events[2].flags == 1
    assert events[3].entity_id is None
    assert events[4].target_id == "p2a: Incineroar"
    assert events[6].target_id == "p2a: Dragapult"


def test_transform_endability_activate_notarget():
    events = parse_events(
        [
            RawBattleEvent(("", "-transform", "p1a: Ditto", "p2a: Dragapult")),
            RawBattleEvent(("", "-endability", "p2a: Incineroar", "Intimidate")),
            RawBattleEvent(("", "-activate", "p1a: Dondozo", "move: Substitute", "[damage]")),
            RawBattleEvent(("", "-activate", "p2a: Incineroar", "ability: Intimidate")),
            RawBattleEvent(("", "-fieldactivate", "move: Perish Song")),
            RawBattleEvent(("", "-notarget", "p1a: Pikachu")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.TRANSFORM,
        EventTypeId.ABILITY_END,
        EventTypeId.ACTIVATE,
        EventTypeId.ACTIVATE,
        EventTypeId.FIELD_ACTIVATE,
        EventTypeId.NO_TARGET,
    ]
    assert events[0].target_id == "p2a: Dragapult"
    assert events[1].ability_id == tokenizer.id_for("abilities", "intimidate") > 0
    assert events[2].effect_id == tokenizer.id_for("volatiles", "substitute") > 0
    assert events[3].ability_id == tokenizer.id_for("abilities", "intimidate") > 0
    assert events[5].entity_id == "p1a: Pikachu"


def test_blocked_keeps_both_endpoints():
    events = parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "-activate", "p2a: Charizard", "move: Protect")),
            RawBattleEvent(("", "-immune", "p2a: Charizard")),
            RawBattleEvent(("", "-miss", "p1a: Pikachu", "p2a: Charizard")),
        ]
    )

    protect, immune, miss = events[1], events[2], events[3]
    assert protect.event_type == EventTypeId.BLOCKED
    assert (protect.entity_id, protect.target_id) == ("p1a: Pikachu", "p2a: Charizard")
    assert (immune.entity_id, immune.target_id, immune.flags) == (
        "p1a: Pikachu",
        "p2a: Charizard",
        1,
    )
    assert (miss.entity_id, miss.target_id, miss.flags) == (
        "p1a: Pikachu",
        "p2a: Charizard",
        2,
    )


def test_weather_upkeep_is_skipped():
    events = parse_events(
        [
            RawBattleEvent(("", "-weather", "SunnyDay")),
            RawBattleEvent(("", "-weather", "SunnyDay", "[upkeep]")),
            RawBattleEvent(("", "-weather", "none")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.WEATHER_START,
        EventTypeId.WEATHER_END,
    ]
    assert events[0].effect_id == tokenizer.id_for("weathers", "sunnyday") > 0


def test_diagnostics_count_oov_and_missing_pre_hp():
    EVENT_DIAGNOSTICS.clear()

    parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Not A Real Move", "p2a: Charizard")),
            RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100")),
        ]
    )

    assert EVENT_DIAGNOSTICS == {"oov_ids": 1, "missing_pre_hp": 1}
    EVENT_DIAGNOSTICS.clear()


def test_truncation_keeps_structural_events():
    events = [BattleEvent(EventTypeId.DAMAGE, "p1a: Pikachu", order=i) for i in range(64)]
    events.extend(
        [
            BattleEvent(EventTypeId.WEATHER_START, None, order=64),
            BattleEvent(EventTypeId.SIDE_START, "p2", order=65),
            BattleEvent(EventTypeId.DRAG, "p1a: Pikachu", order=66),
            BattleEvent(EventTypeId.MEGA, "p1a: Pikachu", order=67),
            BattleEvent(EventTypeId.EFFECT_START, "p2a: Gengar", order=68),
        ]
    )

    selected = truncate_events(events, limit=64)

    kept_types = {event.event_type for event in selected}
    assert {
        EventTypeId.WEATHER_START,
        EventTypeId.SIDE_START,
        EventTypeId.DRAG,
        EventTypeId.MEGA,
        EventTypeId.EFFECT_START,
    } <= kept_types
    orders = [event.order for event in selected]
    assert orders == sorted(orders)


def test_raw_event_pre_hp_snapshot():
    def pre_hp_for(identifier: str) -> float | None:
        assert identifier == "p2a: Charizard"
        return 0.75

    damage = build_raw_event(["", "-damage", "p2a: Charizard", "50/100"], pre_hp_for)
    move = build_raw_event(["", "move", "p1a: Pikachu", "Thunderbolt"], pre_hp_for)

    assert damage.pre_hp == 0.75
    assert damage.message == ("", "-damage", "p2a: Charizard", "50/100")
    assert move.pre_hp is None


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
