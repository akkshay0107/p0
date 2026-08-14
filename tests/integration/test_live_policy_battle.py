import asyncio
import random
from typing import cast

import numpy as np
import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from p0.battle.actions import ACT_SIZE
from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import MemoryInputs
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import order_to_action
from p0.teams.source import FixedTeamSource
from tests.integration.helpers import capture_showdown_decisions, integration_count

TEAM = """
Pikachu @ Light Ball
Ability: Static
Level: 50
Jolly Nature
- Fake Out
- Protect
- Thunderbolt
- Electroweb

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Corrosion
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""


class TrackedPolicyPlayer(RLPlayer):
    def __init__(self, *args, **kwargs):
        self.preview_decisions = 0
        self.normal_decisions = 0
        self.history_tokens = []
        self.selected_actions = {}
        self.round_trips = []
        super().__init__(*args, **kwargs)

    def _get_action(self, battle):
        action = super()._get_action(battle)
        assert np.isfinite(action).all()
        self.history_tokens.append(
            self._battle_histories[self._battle_key(cast(DoubleBattle, battle))].resident_tokens[-1]
        )
        if battle.teampreview:
            self.preview_decisions += 1
        else:
            self.normal_decisions += 1
        self.selected_actions[self._battle_key(cast(DoubleBattle, battle))] = tuple(
            int(value) for value in action
        )
        return action

    def _record_order_round_trip(self, battle: DoubleBattle, order: BattleOrder) -> None:
        key = self._battle_key(battle)
        selected = self.selected_actions.pop(key, None)
        if selected is None:
            return
        encoded = tuple(int(value) for value in order_to_action(order, battle))
        self.round_trips.append((battle.teampreview, selected, encoded))

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        message = super().teampreview(battle)
        self._record_order_round_trip(battle, SingleBattleOrder(message))
        return message

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        order = super().choose_move(battle)
        self._record_order_round_trip(battle, order)
        return order


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("opponent_mode", ["self_policy", "random"])
async def test_checkpoint_free_policy_completes_live_battle(
    showdown_server,
    opponent_mode,
):
    torch.manual_seed(7)
    poke_env_patches.install()
    resources = default_runtime_resources()
    policy = build_policy(ModelConfig.baseline(), resources).eval()
    first_source = FixedTeamSource(TEAM)
    second_source = FixedTeamSource(TEAM)
    first = TrackedPolicyPlayer(
        policy=policy,
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team_source=first_source,
        team_rng=random.Random(11),
        observation_builder=ObservationBuilder(resources),
        max_concurrent_battles=1,
    )
    if opponent_mode == "self_policy":
        second = TrackedPolicyPlayer(
            policy=policy,
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team_source=second_source,
            team_rng=random.Random(13),
            observation_builder=ObservationBuilder(resources),
            max_concurrent_battles=1,
        )
    else:
        second = RandomPlayer(
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team=second_source.sample(random.Random(13)).packed,
            max_concurrent_battles=1,
        )
    try:
        await asyncio.wait_for(first.battle_against(second, n_battles=1), timeout=60.0)
    finally:
        await first.ps_client.stop_listening()
        await second.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()

    assert first.preview_decisions >= 1
    assert first.normal_decisions >= 1
    assert first.history_tokens
    assert not first._battle_histories
    if isinstance(second, TrackedPolicyPlayer):
        assert second.preview_decisions >= 1
        assert second.normal_decisions >= 1
        assert second.history_tokens
        assert not second._battle_histories
        assert first.history_tokens[0] is not second.history_tokens[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_policy_round_trips_live_preview_and_action_orders(
    showdown_server,
    model_policy,
) -> None:
    torch.manual_seed(19)
    poke_env_patches.install()
    first = TrackedPolicyPlayer(
        policy=model_policy,
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team_source=FixedTeamSource(TEAM),
        team_rng=random.Random(29),
        account_configuration=AccountConfiguration("RoundTripA", None),
        observation_builder=ObservationBuilder(model_policy.resources),
        max_concurrent_battles=1,
    )
    second = RandomPlayer(
        account_configuration=AccountConfiguration("RoundTripB", None),
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team=FixedTeamSource(TEAM).sample(random.Random(31)).packed,
        max_concurrent_battles=1,
    )

    try:
        await asyncio.wait_for(first.battle_against(second, n_battles=1), timeout=60.0)
    finally:
        await first.ps_client.stop_listening()
        await second.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()

    assert first.round_trips
    assert any(is_preview for is_preview, _, _ in first.round_trips)
    assert any(not is_preview for is_preview, _, _ in first.round_trips)
    assert all(selected == encoded for _, selected, encoded in first.round_trips)
    assert not first.selected_actions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observed_showdown_orders_round_trip_to_recorded_actions(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=integration_count("P0_INTEGRATION_ACTION_GAMES", 2),
    )
    assert decisions

    for decision in decisions:
        assert decision.chosen_action in decision.legal_joint_actions
        assert decision.chosen_order


@pytest.mark.integration
@pytest.mark.asyncio
async def test_showdown_order_sets_remain_nonempty_across_many_requests(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=integration_count("P0_INTEGRATION_ACTION_GAMES", 2),
    )
    assert decisions
    assert all(decision.legal_joint_actions for decision in decisions)
    for decision in decisions:
        projected = tuple(
            sorted({action for pair in decision.legal_joint_actions for action in pair})
        )
        observed = tuple(sorted(set(decision.legal_actions[0] + decision.legal_actions[1])))
        assert observed
        assert set(projected) <= set(observed)
        assert all(0 <= action < ACT_SIZE for action in observed)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", (1, 8))
async def test_policy_handles_showdown_captured_batches(
    showdown_server,
    model_policy,
    model_device: torch.device,
    batch_size: int,
) -> None:
    decisions = await capture_showdown_decisions(showdown_server, game_count=1)
    assert decisions
    selected = tuple(decisions[index % len(decisions)] for index in range(batch_size))
    observation = StructuredObservation.stack([decision.observation for decision in selected]).to(
        model_device
    )
    action_mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=model_device)
    for index, decision in enumerate(selected):
        for position, actions in enumerate(decision.legal_actions):
            action_mask[index, position, list(actions)] = True
    memory = MemoryInputs.empty(
        batch_size,
        model_policy.d_model,
        model_policy.device,
        next(model_policy.parameters()).dtype,
    )

    with torch.inference_mode():
        encoded = model_policy.encode(observation, action_mask)
        prepared = model_policy.prepare(encoded, memory)
        acted = model_policy.act(prepared, action_mask)
        evaluated = model_policy.evaluate(prepared, action_mask, acted.actions)

    assert acted.actions.shape == (batch_size, 2)
    assert torch.all((acted.actions >= 0) & (acted.actions < FORMAT.action_size))
    assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
    assert torch.isfinite(acted.log_probs).all()
    assert torch.isfinite(acted.value).all()
    assert torch.isfinite(evaluated.log_probs).all()
    assert torch.isfinite(evaluated.entropy).all()
    assert torch.isfinite(evaluated.value).all()
    for index, decision in enumerate(selected):
        actions = tuple(int(value) for value in acted.actions[index].tolist())
        assert actions in decision.legal_joint_actions

    selected_logits = evaluated.logits.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(selected_logits).all()
    assert torch.logical_or(
        torch.isfinite(evaluated.logits), torch.isneginf(evaluated.logits)
    ).all()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_capture_captures_decisions_across_repeated_games(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=integration_count("P0_INTEGRATION_SELF_PLAY_GAMES", 2),
        max_concurrent_battles=2,
    )
    assert decisions
    assert all(decision.legal_joint_actions for decision in decisions)
    assert all(decision.observation.numerical.isfinite().all() for decision in decisions)
