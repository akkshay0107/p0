import asyncio
import random
from typing import cast

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle
from poke_env.player import RandomPlayer

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.teams.source import FixedTeamSource

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
        super().__init__(*args, **kwargs)

    def _get_action(self, battle):
        action = super()._get_action(battle)
        assert np.isfinite(action).all()
        self.history_tokens.append(self._battle_history[self._battle_key(cast(DoubleBattle, battle))][-1])
        if battle.teampreview:
            self.preview_decisions += 1
        else:
            self.normal_decisions += 1
        return action


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
    assert not first._battle_history
    if isinstance(second, TrackedPolicyPlayer):
        assert second.preview_decisions >= 1
        assert second.normal_decisions >= 1
        assert second.history_tokens
        assert not second._battle_history
        assert first.history_tokens[0] is not second.history_tokens[0]
