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

    def test_window_alignment_ownership_and_full_game_summary(self, player: RLPlayer) -> None:
        battle = _battle("battle-window", with_request=True)
        saved_tokens = []
        snapshots = []
        for turn in range(1, 51):
            battle.parse_message(["", "turn", str(turn)])
            with torch.no_grad():
                encoded = player.policy.encode(
                    player.get_observation(battle).unsqueeze(0),
                    torch.from_numpy(action_mask(battle_view(battle).decision)).unsqueeze(0),
                )
            memory = player.memory_inputs(battle)
            count = min(len(saved_tokens), 48)
            assert memory.history_tokens.shape == (1, 48, player.policy.d_model)
            assert memory.history_mask.dtype == torch.bool
            assert not memory.history_mask[:, : 48 - count].any()
            assert memory.history_mask[:, 48 - count :].all()
            assert not memory.history_tokens[:, : 48 - count].any()
            assert not memory.history_tokens.requires_grad
            if count:
                torch.testing.assert_close(
                    memory.history_tokens[:, -count:], torch.stack(saved_tokens[-count:], dim=1)
                )
            if turn in (1, 2, 48, 49, 50):
                snapshots.append((memory, memory.history_tokens.clone()))
            player.choose_move(battle)
            saved_tokens.append(encoded.local_history_token)

        for memory, saved in snapshots:
            torch.testing.assert_close(memory.history_tokens, saved, rtol=0, atol=0)
        # A caller can modify its snapshot without changing the game's retained tokens.
        player.memory_inputs(battle).history_tokens.fill_(123)
        torch.testing.assert_close(
            player.memory_inputs(battle).history_tokens,
            torch.stack(saved_tokens[-48:], dim=1),
        )
        battle.won_by("StressRandomB")
        with torch.no_grad():
            final_token = player.policy.encode(
                player.get_observation(battle).unsqueeze(0),
                torch.from_numpy(action_mask(GAME_END_DECISION)).unsqueeze(0),
            ).local_history_token
            full_history = torch.stack([*saved_tokens, final_token], dim=1)
            expected = player.policy.series(
                full_history, torch.ones(full_history.shape[:2], dtype=torch.bool)
            )[0]
        player._battle_finished_callback(battle)
        next_battle = _battle("battle-window-next", with_request=True)
        next_memory = player.memory_inputs(next_battle)
        assert not next_memory.history_mask.any()
        torch.testing.assert_close(next_memory.series_tokens[next_memory.series_mask], expected)
        player.choose_move(next_battle)
        next_battle.won_by("StressRandomB")
        player._battle_finished_callback(next_battle)
        reset = player.memory_inputs(_battle("battle-new-series", with_request=False))
        assert not reset.history_mask.any()
        assert not reset.series_mask.any()

    @pytest.mark.parametrize("explicit", (False, True))
    def test_model_reload_rebuilds_memory_for_changed_width(
        self, player: RLPlayer, explicit: bool
    ) -> None:
        battle = _battle("battle-reload", with_request=True)
        player.choose_move(battle)
        before = player.memory_inputs(battle)
        policy = build_policy(ModelConfig(32, 4, 1, 64), player.policy.resources).eval()
        player.policy = policy
        if explicit:
            player.invalidate_memory_for_model_reload()
        after = player.memory_inputs(battle)
        assert before.history_mask.sum() == 1
        assert not after.history_mask.any()
        assert not after.series_mask.any()
        assert after.history_tokens.size(-1) == policy.d_model
        assert after.series_tokens.size(-1) == policy.d_model
        player.choose_move(battle)
        assert player.memory_inputs(battle).history_mask.sum() == 1

    @pytest.mark.parametrize("explicit", (False, True))
    def test_model_reload_before_finish_preserves_series_score(
        self, player: RLPlayer, explicit: bool
    ) -> None:
        first = _battle("battle-before-reload", with_request=True)
        player.choose_move(first)
        first.won_by("StressRandomB")
        player._battle_finished_callback(first)

        second = _battle("battle-reload-before-finish", with_request=True)
        player.choose_move(second)
        before = player.memory_inputs(second)
        assert before.series_mask.any()
        player.policy = build_policy(ModelConfig(32, 4, 1, 64), player.policy.resources).eval()
        if explicit:
            player.invalidate_memory_for_model_reload()
        second.won_by("StressRandomB")
        player._battle_finished_callback(second)

        third = _battle("battle-after-reload", with_request=True)
        after = player.memory_inputs(third)
        assert not after.history_mask.any()
        assert not after.series_mask.any()
        assert after.history_tokens.size(-1) == 32
        assert after.series_tokens.size(-1) == 32
        player.choose_move(third)
        third.won_by("StressRandomB")
        player._battle_finished_callback(third)
        fourth = player.memory_inputs(_battle("battle-new-series-second", with_request=False))
        # The third game's win starts a new series, so its summary survives.
        assert fourth.series_mask.any()

    def test_concurrent_battles_have_separate_history(self, player: RLPlayer) -> None:
        first = _battle("battle-first", with_request=True)
        second = _battle("battle-second", with_request=True)
        second.opponent_username = "AnotherOpponent"
        player.choose_move(first)
        saved = player.memory_inputs(first)
        assert not player.memory_inputs(second).history_mask.any()
        player.choose_move(second)
        torch.testing.assert_close(player.memory_inputs(first).history_tokens, saved.history_tokens)
        first.won_by("StressRandomB")
        player._battle_finished_callback(first)
        assert not player.memory_inputs(second).series_mask.any()
        assert player.memory_inputs(second).history_mask.sum() == 1
