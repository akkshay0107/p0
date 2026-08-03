from __future__ import annotations

import pytest

from tests.stress._helpers import capture_showdown_decisions, stress_count


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_observed_showdown_orders_round_trip_to_recorded_actions(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=stress_count("P0_STRESS_ACTION_GAMES", 2),
    )
    assert decisions

    for decision in decisions:
        assert decision.chosen_action in decision.legal_joint_actions
        assert decision.chosen_order


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_showdown_order_sets_remain_nonempty_across_many_requests(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=stress_count("P0_STRESS_ACTION_GAMES", 2),
    )
    assert decisions
    assert all(decision.legal_joint_actions for decision in decisions)
