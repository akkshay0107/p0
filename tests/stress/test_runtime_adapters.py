from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from poke_env.battle import DoubleBattle

from p0.runtime.poke_env_action_adapter import (
    action_to_order,
    action_to_single_order,
    order_to_action,
    single_order_to_action,
)
from p0.runtime.poke_env_battle_adapter import battle_view, current_battle_view, decision_view


class _BattleState:
    pass


def _battle(*, teampreview: bool = False, forced: bool = False) -> DoubleBattle:
    active = SimpleNamespace(
        moves={"tackle": SimpleNamespace(id="tackle")},
        fainted=False,
        base_species="Pikachu",
    )
    team = {
        f"p1: Species-{index}": SimpleNamespace(base_species=f"Species-{index}")
        for index in range(6)
    }
    battle = _BattleState()
    for name, value in {
        "player_username": "player",
        "battle_tag": "stress-battle",
        "teampreview": teampreview,
        "team": team,
        "active_pokemon": [active, None],
        "opponent_active_pokemon": [None, None],
        "available_moves": [[SimpleNamespace(id="struggle" if forced else "tackle")], []],
        "available_switches": [[], []],
        "valid_orders": [[], []],
        "can_mega_evolve": [False, False],
        "force_switch": [False, False],
        "trapped": [False, False],
        "maybe_trapped": [False, False],
        "_wait": False,
        "player_role": "p1",
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "turn": 1,
        "used_mega_evolve": False,
        "opponent_used_mega_evolve": False,
        "get_possible_showdown_targets": lambda move, pokemon: [0],
    }.items():
        setattr(battle, name, value)
    return cast(DoubleBattle, battle)


@pytest.mark.stress
def test_runtime_action_adapters_round_trip_control_move_and_preview_orders() -> None:
    battle = _battle()
    for action in (0, 7):
        order = action_to_single_order(action, battle, fake=True, position=0)
        assert int(single_order_to_action(order, battle, fake=True, position=0)) == action
    forced_battle = _battle(forced=True)
    forced_order = action_to_single_order(48, forced_battle, fake=True, position=0)
    assert int(single_order_to_action(forced_order, forced_battle, fake=True, position=0)) == 48

    assert np.array_equal(
        order_to_action(action_to_order(np.array([-2, -2]), battle), battle), [-2, -2]
    )
    assert np.array_equal(
        order_to_action(action_to_order(np.array([-1, -1]), battle), battle), [-1, -1]
    )

    preview = _battle(teampreview=True)
    selected = np.array([1, 15], dtype=np.int64)
    preview_order = action_to_order(selected, preview)
    assert np.array_equal(order_to_action(preview_order, preview), selected)


@pytest.mark.stress
def test_battle_view_cache_refreshes_decisions_without_replacing_facade() -> None:
    battle = _battle()
    first = current_battle_view(battle)
    first_decision = first.decision
    assert first is current_battle_view(battle)
    assert first_decision is first.decision
    battle._wait = True
    refreshed = battle_view(battle)
    assert refreshed is first
    assert refreshed.decision is not first_decision
    assert refreshed.decision.wait is True
    assert decision_view(battle) == refreshed.decision
