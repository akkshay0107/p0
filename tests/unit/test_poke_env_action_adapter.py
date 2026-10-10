"""Tests for converting policy actions of a called move request to orders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
from poke_env.battle import DoubleBattle

from p0.battle.legality import legal_actions
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import battle_view, decision_view


class TestPokeEnvActionAdapter:
    def test_called_dive_continuation_keeps_target_choices(self) -> None:
        request_path = Path(__file__).parents[1] / "fixtures" / "called_move_request.json"
        request = json.loads(request_path.read_text())
        battle = DoubleBattle("called-dive", "StressRandomB", logging.getLogger("test"), gen=9)
        lines = (
            "|player|p1|StressRandomA|1",
            "|player|p2|StressRandomB|1",
            "|switch|p1a: Blastoise|Blastoise, L50|100/100",
            "|switch|p1b: Pikachu|Pikachu, L50|100/100",
        )
        for line in lines:
            battle.parse_message(cast(list[str], line.split("|")))
        battle.parse_request(request)
        battle.parse_message(
            cast(list[str], "|-prepare|p2a: Espeon|Dive|p1a: Blastoise".split("|"))
        )

        active = battle.active_pokemon[0]
        assert active is not None
        assert tuple(active.moves) == (
            "drainingkiss",
            "charm",
            "imprison",
            "copycat",
        )
        assert tuple(move.id for move in battle.available_moves[0]) == ("dive",)
        assert tuple(battle_view(battle).active_pokemon[0].moves) == ("dive",)
        first_actions = legal_actions(decision_view(battle), 0)
        assert set(first_actions) == {7, 10, 11}
        assert legal_actions(decision_view(battle), 1) == (0,)
        # Actions 7, 10 and 11 are move slot 0 aimed at the ally and at each opponent.
        orders = {
            action: action_to_order(np.array([action, 0], dtype=np.int64), battle).message
            for action in first_actions
        }
        assert orders == {
            7: "/choose move dive -2, pass",
            10: "/choose move dive 1, pass",
            11: "/choose move dive 2, pass",
        }
