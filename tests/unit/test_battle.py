from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from p0.battle.actions import (
    ACT_SIZE,
    ActionKind,
    SlotAction,
    canonical_team_actions,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    PROTECT_EFFECTS,
    BattleEvent,
    EventTypeId,
    RawBattleEvent,
    _resolve_effect,
    get_hp_fraction,
    parse_events,
    truncate_events,
)
from p0.battle.legality import (
    DecisionView,
    SlotDecision,
    action_mask,
    apply_joint_constraints,
    legal_actions,
    second_action_mask,
    validate_joint_action,
)
from p0.battle.series import SeriesPerspectiveKey
from p0.format_config import ACTION_CONTRACT
from p0.model.structured_observation import EVENT_COUNT
from p0.model.tokenizer import tokenizer


def test_action_contract_round_trips_ids_and_describes_canonical_ranges() -> None:
    assert [encode_action(decode_action(action)) for action in range(ACT_SIZE)] == list(
        range(ACT_SIZE)
    )
    assert ACTION_CONTRACT["action_count"] == ACT_SIZE
    ranges = ACTION_CONTRACT["ranges"]
    assert [(entry["start"], entry["end"]) for entry in ranges] == [
        (0, 1),
        (1, 7),
        (7, 27),
        (27, 47),
        (47, 48),
        (48, 49),
    ]
    assert [decode_action(index).kind.name.lower() for index in (0, 1, 7, 27, 47, 48)] == [
        "pass",
        "switch",
        "move",
        "move",
        "forced_move",
        "forced_move",
    ]
    actions = {
        encode_team_pair(first, second)
        for first in range(6)
        for second in range(6)
        if first != second
    }
    assert len(actions) == 30
    assert all(decode_team_pair(action)[0] != decode_team_pair(action)[1] for action in actions)
    assert team_selection(1, 8)[:4] == (0, 1, 2, 3)


def test_scalar_joint_constraints_match_policy_vectorization() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(
                switch_slots=(2, 3),
                move_targets=((-2, 1, 2), (0,), (), (1,)),
                can_mega=True,
            ),
            SlotDecision(
                switch_slots=(2, 4),
                move_targets=((-1, 1), (2,), (0,), ()),
                can_mega=True,
            ),
        )
    )
    first_legal = legal_actions(view, 0)
    for first in first_legal:
        joint_mask = second_action_mask(view, first)
        for second in range(ACT_SIZE):
            assert bool(joint_mask[second]) == validate_joint_action(view, first, second)


def test_series_perspective_key_validation() -> None:
    key0 = SeriesPerspectiveKey("series-1", 0)
    key1 = SeriesPerspectiveKey("series-1", 1)
    assert key0.series_id == "series-1" and key0.canonical_player == 0
    assert key1.canonical_player == 1

    with pytest.raises(ValueError, match="non-empty"):
        SeriesPerspectiveKey("", 0)

    with pytest.raises(ValueError, match="0 or 1"):
        SeriesPerspectiveKey("series-1", 2)
    with pytest.raises(ValueError, match="0 or 1"):
        SeriesPerspectiveKey("series-1", -1)
    with pytest.raises(ValueError, match="0 or 1"):
        SeriesPerspectiveKey("series-1", cast(Any, "0"))


def test_action_encoding_and_decoding_boundary_errors() -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 49\)"):
        decode_action(-1)
    with pytest.raises(ValueError, match=r"must be in \[0, 49\)"):
        decode_action(49)
    with pytest.raises(ValueError, match=r"must be in \[0, 49\)"):
        decode_action(100)

    with pytest.raises(ValueError, match="Invalid switch slot"):
        encode_action(SlotAction(ActionKind.SWITCH, switch_slot=-1))
    with pytest.raises(ValueError, match="Invalid switch slot"):
        encode_action(SlotAction(ActionKind.SWITCH, switch_slot=6))

    with pytest.raises(ValueError, match="Invalid move slot"):
        encode_action(SlotAction(ActionKind.MOVE, move_slot=-1, target=0))
    with pytest.raises(ValueError, match="Invalid move slot"):
        encode_action(SlotAction(ActionKind.MOVE, move_slot=4, target=0))

    with pytest.raises(ValueError, match="Invalid move target"):
        encode_action(SlotAction(ActionKind.MOVE, move_slot=0, target=-3))
    with pytest.raises(ValueError, match="Invalid move target"):
        encode_action(SlotAction(ActionKind.MOVE, move_slot=0, target=3))

    with pytest.raises(ValueError, match="Unsupported action kind"):
        encode_action(SlotAction(cast(Any, 99)))


def test_team_preview_bounds_and_validation_errors() -> None:
    with pytest.raises(ValueError, match="outside the roster"):
        encode_team_pair(-1, 0)
    with pytest.raises(ValueError, match="outside the roster"):
        encode_team_pair(0, 6)
    with pytest.raises(ValueError, match="must be distinct"):
        encode_team_pair(2, 2)

    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(-1)
    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(36)
    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(0)
    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(7)

    with pytest.raises(ValueError, match="four distinct members"):
        canonical_team_actions((0, 0, 0, 0, 0, 0))


def test_double_force_switch_with_single_available_switch() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(switch_slots=(2,), force_switch=True),
            SlotDecision(switch_slots=(2,), force_switch=True),
        )
    )
    assert legal_actions(view, 0) == (3, 0)
    assert legal_actions(view, 1) == (3, 0)


def test_apply_joint_constraints_fallback_and_error_resilience() -> None:
    preview = DecisionView(slots=(SlotDecision(), SlotDecision()), team_preview=True)
    mask = np.ones(ACT_SIZE, dtype=np.bool_)
    apply_joint_constraints(mask, preview, first=-1)
    assert not mask.any()

    view = DecisionView(
        slots=(
            SlotDecision(switch_slots=(2,)),
            SlotDecision(switch_slots=(2,)),
        )
    )
    slot1_mask = np.zeros(ACT_SIZE, dtype=np.bool_)
    slot1_mask[3] = True
    apply_joint_constraints(slot1_mask, view, first=3)
    assert slot1_mask[0]
    assert not slot1_mask[3]


def test_event_resolver_prefix_and_protect_activations() -> None:
    assert _resolve_effect(tokenizer, "moves", "move: Taunt") == tokenizer.id_for("moves", "taunt")
    assert _resolve_effect(tokenizer, "moves", "Taunt") == tokenizer.id_for("moves", "taunt")

    for protect_effect in PROTECT_EFFECTS:
        events = parse_events(
            [
                RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
                RawBattleEvent(("", "-activate", "p2a: Charizard", protect_effect)),
            ],
            tokenizer,
        )
        assert len(events) == 2
        assert events[1].event_type is EventTypeId.BLOCKED
        assert events[1].entity_id == "p1a: Pikachu"
        assert events[1].target_id == "p2a: Charizard"


def _struggle_battle(move_id: str, can_mega: bool) -> DecisionView:
    return DecisionView(
        slots=(
            SlotDecision(
                forced_move=True,
                can_mega=can_mega,
            ),
            SlotDecision(),
        )
    )


def test_struggle_env_roundtrip() -> None:
    view = _struggle_battle("struggle", can_mega=False)
    assert list(legal_actions(view, 0)) == [48]

    mega_view = _struggle_battle("recharge", can_mega=True)
    mask = list(legal_actions(mega_view, 0))
    assert 48 in mask
    assert 47 in mask


def test_action_ids_cover_all_boundary_categories() -> None:
    expected = {
        0: SlotAction(ActionKind.PASS),
        1: SlotAction(ActionKind.SWITCH, switch_slot=0),
        6: SlotAction(ActionKind.SWITCH, switch_slot=5),
        7: SlotAction(ActionKind.MOVE, move_slot=0, target=-2),
        11: SlotAction(ActionKind.MOVE, move_slot=0, target=2),
        26: SlotAction(ActionKind.MOVE, move_slot=3, target=2),
        27: SlotAction(ActionKind.MOVE, move_slot=0, target=-2, mega=True),
        46: SlotAction(ActionKind.MOVE, move_slot=3, target=2, mega=True),
        47: SlotAction(ActionKind.FORCED_MOVE, mega=True),
        48: SlotAction(ActionKind.FORCED_MOVE),
    }
    for action_id, semantic in expected.items():
        assert decode_action(action_id) == semantic
        assert encode_action(semantic) == action_id


def test_team_preview_pairs_and_joint_constraints_preserve_uniqueness() -> None:
    pairs = tuple((first, second) for first in range(6) for second in range(6) if first != second)
    assert tuple(decode_team_pair(encode_team_pair(*pair)) for pair in pairs) == pairs
    selection = (5, 4, 3, 2, 0, 1)
    lead, back = canonical_team_actions(selection)
    assert team_selection(lead, back) == selection

    preview = DecisionView(slots=(SlotDecision(), SlotDecision()), team_preview=True)
    assert legal_actions(preview, 0) == tuple(encode_team_pair(*pair) for pair in pairs)
    assert validate_joint_action(preview, 1, 15)
    assert not validate_joint_action(preview, 1, 14)

    regular = DecisionView(
        slots=(
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
        )
    )
    assert validate_joint_action(regular, 7, 11)
    assert not validate_joint_action(regular, 1, 1)
    assert not validate_joint_action(regular, 27, 31)

    forced = DecisionView(
        slots=(
            SlotDecision(forced_move=True, can_mega=True),
            SlotDecision(forced_move=True, can_mega=True),
        )
    )
    assert legal_actions(forced, 0) == (48, 47)


def test_unknown_legality_masks_are_supersets_of_the_proven_mask() -> None:
    proven = SlotDecision(switch_slots=(2,), move_targets=((-2, 1), (), (), ()), can_mega=True)
    unknown = SlotDecision(
        switch_slots=(2,),
        move_targets=((-2, 1), (), (), ()),
        can_mega=True,
        legality_known=False,
    )
    proven_view = DecisionView(slots=(proven, proven))
    unknown_view = DecisionView(slots=(unknown, unknown))

    proven_mask = action_mask(proven_view)
    unknown_mask = action_mask(unknown_view)

    assert np.all(unknown_mask >= proven_mask)
    assert set(legal_actions(unknown_view, 0)) - set(legal_actions(proven_view, 0)) == {0, 47, 48}


def _local_parse_events(raw_events: list[RawBattleEvent]) -> list[BattleEvent]:
    return parse_events(raw_events, tokenizer)


def test_hp_fraction_accepts_showdown_status_suffixes() -> None:
    assert get_hp_fraction("50/100g") == 0.5
    assert get_hp_fraction("20/100y") == 0.2
    assert get_hp_fraction("0 fnt") == 0.0


def test_parse_events_returns_typed_events_in_protocol_order() -> None:
    raw_events = [
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(
            ("", "-damage", "p2a: Charizard", "50/100", "[from] item: Life Orb"), pre_hp=0.75
        ),
        RawBattleEvent(("", "faint", "p2a: Charizard")),
        RawBattleEvent(("", "switch", "p2a: Blastoise", "Blastoise, L50, M", "100/100")),
        RawBattleEvent(("", "-mega", "p1a: Pikachu", "Pikachuite")),
        RawBattleEvent(("", "-status", "p1a: Pikachu", "par")),
        RawBattleEvent(
            ("", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2a: Blastoise")
        ),
        RawBattleEvent(("", "-fieldstart", "move: Electric Terrain")),
    ]
    events = _local_parse_events(raw_events)
    assert len(events) == 8
    assert events[0].event_type == EventTypeId.MOVE
    assert events[1].event_type == EventTypeId.DAMAGE
    assert events[1].value == pytest.approx(-0.25)
    assert events[2].event_type == EventTypeId.FAINT
    assert events[3].event_type == EventTypeId.SWITCH_IN
    assert events[4].event_type == EventTypeId.MEGA
    assert events[5].event_type == EventTypeId.STATUS_SET
    assert events[6].event_type == EventTypeId.WEATHER_START
    assert events[7].event_type == EventTypeId.FIELD_START


def test_parse_events_distinguishes_failed_and_blocked_moves() -> None:
    raw_events = [
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(("", "-immune", "p2a: Charizard")),
        RawBattleEvent(("", "-fail", "p1a: Pikachu")),
    ]
    events = _local_parse_events(raw_events)
    assert [event.event_type for event in events] == [
        EventTypeId.MOVE,
        EventTypeId.BLOCKED,
        EventTypeId.FAILED,
    ]


def test_parse_events_resets_last_attacker_at_turn_boundaries() -> None:
    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "turn", "2")),
            RawBattleEvent(("", "-fail", "p2a: Charizard")),
        ]
    )
    assert events[-1].event_type is EventTypeId.FAILED
    assert events[-1].entity_id == "p2a: Charizard"


def test_parse_events_clears_inferred_sources_at_upkeep_boundaries() -> None:
    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "upkeep")),
            RawBattleEvent(("", "-immune", "p2a: Charizard")),
            RawBattleEvent(("", "-activate", "p2a: Charizard", "move: Protect")),
        ]
    )
    immune, protect = events[-2:]
    assert (immune.event_type, immune.entity_id, immune.target_id) == (
        EventTypeId.BLOCKED,
        None,
        "p2a: Charizard",
    )
    assert (protect.event_type, protect.entity_id, protect.target_id) == (
        EventTypeId.BLOCKED,
        None,
        "p2a: Charizard",
    )


def test_ability_field_and_move_evidence() -> None:
    events = _local_parse_events(
        [
            RawBattleEvent(
                ("", "move", "p1a: Mew", "Metronome", "p2a: Gengar", "[from] ability: Dancer")
            ),
            RawBattleEvent(("", "-ability", "p2a: Gengar", "Cursed Body")),
            RawBattleEvent(("", "-fieldstart", "move: Trick Room")),
            RawBattleEvent(("", "-fieldend", "move: Trick Room")),
        ]
    )
    assert events[0].target_id == "p2a: Gengar"
    assert events[0].flags & 4
    assert events[1].event_type == EventTypeId.ABILITY
    assert events[1].ability_id > 0
    assert [event.event_type for event in events[2:]] == [
        EventTypeId.FIELD_START,
        EventTypeId.FIELD_END,
    ]
    assert events[2].effect_id > 0


def test_status_codes_resolve_against_vocab() -> None:
    events = _local_parse_events(
        [
            RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
            RawBattleEvent(("", "-curestatus", "p2a: Charizard", "par")),
        ]
    )
    assert events[0].status_id == tokenizer.id_for("status", "par") > 0
    assert events[1].status_id == events[0].status_id


def test_cant_prepare_and_singlemove() -> None:
    events = _local_parse_events(
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


def test_boost_manipulation_family() -> None:
    events = _local_parse_events(
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


def test_transform_endability_activate_notarget() -> None:
    events = _local_parse_events(
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


def test_blocked_keeps_both_endpoints() -> None:
    events = _local_parse_events(
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


def test_weather_upkeep_is_skipped() -> None:
    events = _local_parse_events(
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


def test_diagnostics_count_oov_and_missing_pre_hp() -> None:
    EVENT_DIAGNOSTICS.clear()
    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Not A Real Move", "p2a: Charizard")),
            RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100")),
        ]
    )
    assert EVENT_DIAGNOSTICS == {"oov_ids": 1, "missing_pre_hp": 1}
    assert events[1].value == 0.0
    EVENT_DIAGNOSTICS.clear()


def test_truncation_keeps_structural_events() -> None:
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
    selected = truncate_events(events, limit=24)
    assert len(selected) == 24
    kept_types = {e.event_type for e in selected}
    assert {
        EventTypeId.WEATHER_START,
        EventTypeId.SIDE_START,
        EventTypeId.DRAG,
        EventTypeId.MEGA,
        EventTypeId.EFFECT_START,
    } <= kept_types
    orders = [event.order for event in selected]
    assert orders == sorted(orders)


def test_protocol_event_stream_matches_showdown_golden_types() -> None:
    from tests.unit.replay_fixtures import GOLDEN_EVENT_TYPES, golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    events = parse_events(list(golden_raw_events()), tokenizer)
    assert tuple(event.event_type for event in events) == GOLDEN_EVENT_TYPES
    assert events[0].entity_id == "p1a: Pikachu"
    damage = next(event for event in events if event.event_type is EventTypeId.DAMAGE)
    heal = next(event for event in events if event.event_type is EventTypeId.HEAL)
    unboost = next(event for event in events if event.event_type is EventTypeId.UNBOOST)
    assert damage.value == pytest.approx(-0.25)
    assert heal.value == pytest.approx(0.15)
    assert unboost.value == pytest.approx(-1 / 6)
    assert any(event.event_type is EventTypeId.MEGA for event in events)
    assert EVENT_DIAGNOSTICS["oov_ids"] >= 1


def test_event_truncation_keeps_priority_events_and_protocol_order() -> None:
    from tests.unit.replay_fixtures import golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    events = parse_events(list(golden_raw_events()), tokenizer)
    truncated = truncate_events(events, limit=12)

    assert len(truncated) == 12
    assert [event.order for event in truncated] == sorted(event.order for event in truncated)
    assert any(event.event_type is EventTypeId.MOVE for event in truncated)
    assert any(event.event_type is EventTypeId.SWITCH_IN for event in truncated)
    assert any(event.event_type is EventTypeId.FAINT for event in truncated)


def test_malformed_and_incomplete_protocol_lines_are_diagnosed_without_fabrication() -> None:
    from tests.unit.replay_fixtures import GOLDEN_EVENT_TYPES, golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    raw_events = list(golden_raw_events())
    raw_events.extend(
        (
            RawBattleEvent(("",)),
            RawBattleEvent(("", "chat", "ignored")),
            RawBattleEvent(("", "switch", "p1a: Pikachu")),
            RawBattleEvent(("", "-damage", "p2a: Charizard", "50/100")),
            RawBattleEvent(("", "-status", "p2a: Charizard")),
            RawBattleEvent(("", "move", "p1a: Pikachu")),
        )
    )
    events = parse_events(raw_events, tokenizer)
    assert len(events) == len(GOLDEN_EVENT_TYPES) + 1
    assert EVENT_DIAGNOSTICS["oov_ids"] >= 1
    assert EVENT_DIAGNOSTICS["missing_pre_hp"] == 1
    damage = events[-1]
    assert damage.event_type is EventTypeId.DAMAGE
    assert damage.value == 0.0


def test_event_truncation_handles_below_equal_and_above_capacity_limits() -> None:
    from tests.unit.replay_fixtures import golden_raw_events

    events = parse_events(list(golden_raw_events()) * 2, tokenizer)
    assert len(events) > EVENT_COUNT
    for limit in (EVENT_COUNT - 1, EVENT_COUNT, EVENT_COUNT + 1):
        truncated = truncate_events(events, limit=limit)
        assert len(truncated) == limit
        assert [event.order for event in truncated] == sorted(event.order for event in truncated)


def test_protocol_parser_accepts_an_injected_resource_resolver() -> None:
    resolver = SimpleNamespace(
        resolve=lambda table, name: (
            (17, "exact") if table == "moves" and name == "Tackle" else (0, "exact")
        ),
    )
    events = parse_events(
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Tackle", "p2a: Eevee"))],
        cast(Any, resolver),
    )
    assert events[0].event_type is EventTypeId.MOVE
    assert events[0].move_id == 17
