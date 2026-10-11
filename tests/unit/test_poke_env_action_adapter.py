"""Tests for converting policy actions of a called move request to orders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
from poke_env.battle import DoubleBattle

from p0.battle.actions import FORCED_ACTION
from p0.battle.legality import legal_actions
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import battle_view, decision_view
from p0.runtime.poke_env_patches import locked_choice


class TestPokeEnvActionAdapter:
    def test_called_dive_continuation_is_a_forced_move(self) -> None:
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
        # Showdown gives the locked Blastoise no choice, so its only action is the forced move.
        assert legal_actions(decision_view(battle), 0) == (FORCED_ACTION,)
        assert legal_actions(decision_view(battle), 1) == (0,)
        order = action_to_order(np.array([FORCED_ACTION, 0], dtype=np.int64), battle)
        assert locked_choice(battle, order.message) == "/choose move 1, pass"

    def test_locked_own_move_is_a_forced_move(self) -> None:
        battle = DoubleBattle("locked-uproar", "Alice", logging.getLogger("test"), gen=9)
        for line in (
            "|player|p1|Alice|1",
            "|player|p2|Bob|1",
            "|switch|p1a: Rotom|Rotom-Wash, L50|157/157",
            "|switch|p1b: Houndstone|Houndstone, L50, M|179/179",
            "|switch|p2a: Blissey|Blissey, L50, F|100/100",
            "|switch|p2b: Snorlax|Snorlax, L50, F|100/100",
        ):
            battle.parse_message(cast(list[str], line.split("|")))
        battle.parse_request(
            {
                "active": [
                    {"moves": [_request_move("Rest", "self", disabled=False)]},
                    # Showdown's request entry for a Pokemon on a later turn of Uproar.
                    {"moves": [{"move": "Uproar", "id": "uproar"}], "trapped": True},
                ],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Rotom", "Rotom-Wash, L50", "157/157", ["rest"]),
                        _request_pokemon(
                            "p1: Houndstone", "Houndstone, L50, M", "179/179", ["facade", "uproar"]
                        ),
                    ],
                },
            }
        )

        assert legal_actions(decision_view(battle), 1) == (FORCED_ACTION,)
        order = action_to_order(np.array([9, FORCED_ACTION], dtype=np.int64), battle)
        assert locked_choice(battle, order.message) == "/choose move rest, move 1"

    def test_sole_enabled_move_with_a_target_is_not_forced(self) -> None:
        battle = DoubleBattle("choice-lock", "Alice", logging.getLogger("test"), gen=9)
        for line in (
            "|player|p1|Alice|1",
            "|player|p2|Bob|1",
            "|switch|p1a: Rotom|Rotom-Wash, L50|157/157",
            "|switch|p1b: Houndstone|Houndstone, L50, M|179/179",
            "|switch|p2a: Blissey|Blissey, L50, F|100/100",
            "|switch|p2b: Snorlax|Snorlax, L50, F|100/100",
        ):
            battle.parse_message(cast(list[str], line.split("|")))
        # A Choice item disables every other move but still lets the player aim.
        battle.parse_request(
            {
                "active": [
                    {"moves": [_request_move("Rest", "self", disabled=False)]},
                    {
                        "moves": [
                            _request_move("Facade", "normal", disabled=False),
                            _request_move("Uproar", "randomNormal", disabled=True),
                        ]
                    },
                ],
                "side": {
                    "name": "Alice",
                    "id": "p1",
                    "pokemon": [
                        _request_pokemon("p1: Rotom", "Rotom-Wash, L50", "157/157", ["rest"]),
                        _request_pokemon(
                            "p1: Houndstone", "Houndstone, L50, M", "179/179", ["facade", "uproar"]
                        ),
                    ],
                },
            }
        )

        # Facade is move slot 0: aimed at the ally in the first slot (8) and each opponent (10, 11).
        assert legal_actions(decision_view(battle), 1) == (8, 10, 11)


def _request_move(name: str, target: str, *, disabled: bool) -> dict[str, object]:
    return {
        "move": name,
        "id": name.replace(" ", "").lower(),
        "pp": 8,
        "maxpp": 8,
        "target": target,
        "disabled": disabled,
    }


def _request_pokemon(
    ident: str, details: str, condition: str, moves: list[str]
) -> dict[str, object]:
    return {
        "ident": ident,
        "details": details,
        "condition": condition,
        "active": True,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": moves,
        "baseAbility": "noability",
        "item": "",
        "pokeball": "pokeball",
        "ability": "noability",
        "commanding": False,
        "reviving": False,
    }
