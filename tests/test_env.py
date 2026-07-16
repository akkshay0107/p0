from types import SimpleNamespace
from typing import cast

import pytest
from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import PassBattleOrder

from p0.runtime.poke_env_action_adapter import action_to_single_order


def test_action_validation_rejects_orders_outside_battle_order_space():
    valid_order = PassBattleOrder()
    battle = cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            valid_orders=([valid_order], []),
        ),
    )

    order = action_to_single_order(
        0,
        battle,
        fake=False,
        position=0,
    )
    assert str(order) == str(valid_order)

    battle.valid_orders[0].clear()
    with pytest.raises(ValueError, match="not in action space"):
        action_to_single_order(
            0,
            battle,
            fake=False,
            position=0,
        )
