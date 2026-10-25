"""Tests for translating live request legality into policy masks."""

from __future__ import annotations

import logging

import pytest
from poke_env.battle import DoubleBattle

from p0.battle.legality import legal_actions
from p0.runtime.poke_env_battle_adapter import decision_view

LOGGER = logging.getLogger(__name__)


class TestPokeEnvBattleAdapter:
    @pytest.mark.parametrize(
        ("trapping", "can_switch"),
        (
            ({}, True),
            ({"maybeTrapped": True}, True),
            ({"trapped": True}, False),
            ({"maybeTrapped": True, "trapped": True}, False),
        ),
    )
    def test_only_confirmed_trapping_blocks_switches(
        self, trapping: dict[str, bool], can_switch: bool
    ) -> None:
        battle = DoubleBattle("trapping", "Alice", LOGGER, gen=9)
        move = {"move": "Protect", "id": "protect", "pp": 8, "maxpp": 8, "target": "self"}
        pokemon = [
            {
                "ident": f"p1: {species}",
                "details": f"{species}, L50",
                "condition": "100/100",
                "active": index < 2,
                "moves": ["protect"],
                "baseAbility": ability,
                "item": "",
            }
            for index, (species, ability) in enumerate(
                (("Incineroar", "intimidate"), ("Raichu", "static"), ("Charizard", "blaze"))
            )
        ]
        battle.parse_request(
            {
                "active": [{"moves": [move]}, {"moves": [move], **trapping}],
                "side": {"name": "Alice", "id": "p1", "pokemon": pokemon},
            }
        )

        # Charizard is roster slot 2: its switch action is 2 + 1 = 3.
        assert (3 in legal_actions(decision_view(battle), 1)) is can_switch
