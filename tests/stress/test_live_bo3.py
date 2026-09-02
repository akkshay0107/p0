from __future__ import annotations

import asyncio
import random

import pytest
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.teams.source import FixedTeamSource


class TestLiveBo3:
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_local_rl_player_completes_bo3_against_random(
        self, showdown_server, stress_policy
    ) -> None:
        """Play one complete local Showdown Bo3 and verify public player state is reclaimed."""
        poke_env_patches.install()
        source = FixedTeamSource(DEFAULT_TEST_TEAM)
        random_source = FixedTeamSource(DEFAULT_TEST_TEAM)
        rl_player = RLPlayer(
            policy=stress_policy,
            battle_format=FORMAT.bo3_format,
            server_configuration=showdown_server,
            team_source=source,
            team_rng=random.Random(17),
            observation_builder=ObservationBuilder(stress_policy.resources),
            account_configuration=AccountConfiguration("StressBo3RL", None),
            max_concurrent_battles=1,
        )
        random_player = RandomPlayer(
            battle_format=FORMAT.bo3_format,
            server_configuration=showdown_server,
            team=random_source.sample(random.Random(19)).packed,
            account_configuration=AccountConfiguration("StressBo3Random", None),
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
        poke_env_patches.enable_forced_open_team_sheet(random_player)

        try:
            await asyncio.wait_for(
                rl_player.battle_against(random_player, n_battles=1), timeout=120.0
            )
            deadline = asyncio.get_running_loop().time() + 120.0
            while (
                random_player.n_finished_battles < 2 or rl_player.battles
            ) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)
        finally:
            await rl_player.ps_client.stop_listening()
            await random_player.ps_client.stop_listening()
            poke_env_patches.uninstall_for_tests()

        assert 2 <= random_player.n_finished_battles <= 3
        assert rl_player.current_team_packed is not None
        assert rl_player.battles == {}
