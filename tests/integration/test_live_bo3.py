from __future__ import annotations

import asyncio
import random

import pytest
from poke_env import AccountConfiguration
from poke_env.battle import DoubleBattle
from poke_env.player import RandomPlayer

from p0.evaluation.harness import EvalPlayer
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.runtime import poke_env_patches
from tests.team_fixtures import default_test_corpus


class TestLiveBo3:
    @pytest.mark.heavy
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_local_rl_player_completes_bo3_against_random(
        self, showdown_server, model_policy
    ) -> None:
        """Forfeit parent sets, rematch, and complete a Bo3 with clean memory and teams."""
        poke_env_patches.install()
        source = default_test_corpus()
        random_source = default_test_corpus()
        rl_player = EvalPlayer(
            policy=model_policy,
            battle_format=FORMAT.bo3_format,
            server_configuration=showdown_server,
            team_source=source,
            team_rng=random.Random(17),
            observation_builder=ObservationBuilder(model_policy.resources),
            account_configuration=AccountConfiguration("StressBo3RL", None),
            max_concurrent_battles=1,
        )
        random_player = RandomPlayer(
            battle_format=FORMAT.bo3_format,
            server_configuration=showdown_server,
            team=random_source.sample(random.Random(19)),
            account_configuration=AccountConfiguration("StressBo3Random", None),
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
        poke_env_patches.enable_forced_open_team_sheet(random_player)

        try:
            expected_rng = random.Random(17)
            source.sample(expected_rng)
            for series_count in (1, 2):
                challenge = asyncio.create_task(
                    rl_player.battle_against(random_player, n_battles=1)
                )
                async with asyncio.timeout(30):
                    while True:
                        battle = next(iter(rl_player.battles.values()), None)
                        if isinstance(battle, DoubleBattle) and battle.opponent_username:
                            parent = poke_env_patches.parent_room_for_battle(
                                rl_player.ps_client, battle.battle_tag
                            )
                            if (
                                parent is not None
                                and rl_player.memory_inputs(battle).history_mask.any()
                            ):
                                break
                        await asyncio.sleep(0.01)
                    assert not rl_player.memory_inputs(battle).series_mask.any()
                    # A legal parent forfeit ends the set before either seat has
                    # two child wins; its final child result can arrive afterward.
                    await random_player.ps_client.send_message("/forfeit", parent)
                    await challenge
                    await poke_env_patches.wait_for_parent_results(
                        rl_player.ps_client, series_count
                    )
                    while rl_player.battles:
                        await asyncio.sleep(0.01)
                source.sample(expected_rng)
                assert rl_player.team_rng.getstate() == expected_rng.getstate()

            finished_before = random_player.n_finished_battles
            async with asyncio.timeout(120):
                await rl_player.battle_against(random_player, n_battles=1)
                await poke_env_patches.wait_for_parent_results(rl_player.ps_client, 3)
                while random_player.n_finished_battles - finished_before < 2 or rl_player.battles:
                    await asyncio.sleep(0.1)
            source.sample(expected_rng)
            assert rl_player.team_rng.getstate() == expected_rng.getstate()
        finally:
            await rl_player.ps_client.stop_listening()
            await random_player.ps_client.stop_listening()

        assert 2 <= random_player.n_finished_battles - finished_before <= 3
        assert rl_player.current_team_packed is not None
        assert rl_player.battles == {}
