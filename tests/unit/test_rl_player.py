"""Tests for the live policy player's per-battle and per-series memory."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import cast

import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.battle import DoubleBattle

from p0.battle.events import EventKind, EventPosition
from p0.battle.legality import GAME_END_DECISION, action_mask
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view
from tests.team_fixtures import DEFAULT_TEST_TEAM

REQUEST_PATH = Path(__file__).parents[1] / "fixtures" / "called_move_request.json"


@pytest.fixture
def player():
    policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources()).eval()
    rl_player = RLPlayer(
        policy,
        observation_builder=ObservationBuilder(policy.resources),
        team_rng=random.Random(0),
        team=DEFAULT_TEST_TEAM,
        account_configuration=AccountConfiguration("StressRandomB", None),
        start_listening=False,
    )
    try:
        yield rl_player
    finally:
        poke_env_patches.uninstall_for_tests()


def _battle(tag: str, *, with_request: bool) -> DoubleBattle:
    """Build a live battle for the fixture request's side, parsed through the capture patch."""
    battle = DoubleBattle(tag, "StressRandomB", logging.getLogger("test"), gen=9)
    # poke-env's player sets this when it creates a battle from a challenge.
    battle.opponent_username = "StressRandomA"
    lines = (
        "|player|p1|StressRandomA|1",
        "|player|p2|StressRandomB|1",
        "|switch|p1a: Blastoise|Blastoise, L50|100/100",
        "|switch|p1b: Pikachu|Pikachu, L50|100/100",
    )
    for line in lines:
        battle.parse_message(cast(list[str], line.split("|")))
    if with_request:
        battle.parse_request(json.loads(REQUEST_PATH.read_text()))
    return battle


class TestRLPlayerMemory:
    def test_rejected_choice_retry_keeps_one_history_entry_and_the_same_events(
        self, player: RLPlayer
    ) -> None:
        battle = _battle("battle-retry", with_request=True)

        first_events = battle_view(battle).spatial_events
        player.choose_move(battle)
        # After an [Invalid choice] error poke-env asks for the same request again
        # before any battle line arrives.
        retried_events = battle_view(battle).spatial_events
        player.choose_move(battle)
        after_retry = int(player.memory_inputs(battle).history_mask.sum())
        battle.parse_message(["", "turn", "2"])
        player.choose_move(battle)
        after_next_decision = int(player.memory_inputs(battle).history_mask.sum())

        assert len(first_events) == 2
        assert retried_events == first_events
        assert after_retry == 1
        assert after_next_decision == 2

    def test_final_exchange_joins_the_series_memory_of_the_next_game(
        self, player: RLPlayer
    ) -> None:
        policy = player.policy
        battle = _battle("battle-final", with_request=True)
        player.choose_move(battle)
        memory = player.memory_inputs(battle)
        decision_tokens = memory.history_tokens[memory.history_mask]
        battle.parse_message(["", "faint", "p1a: Blastoise"])
        battle.won_by("StressRandomB")

        final_events = battle_view(battle).spatial_events
        with torch.no_grad():
            final_token = policy.encode(
                player.get_observation(battle).unsqueeze(0),
                torch.from_numpy(action_mask(GAME_END_DECISION)).unsqueeze(0),
            ).local_history_token
            with_final = policy.series(
                torch.cat((decision_tokens, final_token)).unsqueeze(0),
                torch.ones((1, 2), dtype=torch.bool),
            )[0]
            decisions_only = policy.series(
                decision_tokens.unsqueeze(0), torch.ones((1, 1), dtype=torch.bool)
            )[0]
        # poke-env calls this hook when the win line arrives.
        player._battle_finished_callback(battle)
        next_game = player.memory_inputs(_battle("battle-next", with_request=False))
        series_tokens = next_game.series_tokens[next_game.series_mask]

        assert [record[:3] for record in final_events] == [
            (EventKind.FAINT, EventPosition.NONE, EventPosition.OPPONENT_LEFT)
        ]
        assert battle_view(battle).decision == GAME_END_DECISION
        torch.testing.assert_close(series_tokens, with_final)
        assert not torch.allclose(series_tokens, decisions_only)
