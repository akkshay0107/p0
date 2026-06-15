from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import PassBattleOrder

from src.env import MegaEnv


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

    order = MegaEnv._action_to_order_individual(
        np.int64(0),
        battle,
        fake=False,
        pos=0,
    )
    assert str(order) == str(valid_order)

    battle.valid_orders[0].clear()
    with pytest.raises(ValueError, match="not in action space"):
        MegaEnv._action_to_order_individual(
            np.int64(0),
            battle,
            fake=False,
            pos=0,
        )
