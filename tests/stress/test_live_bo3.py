from __future__ import annotations

import asyncio
import random

import pytest
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.teams.source import FixedTeamSource


class _CountingTeamSource:
    def __init__(self, team: str):
        self._source = FixedTeamSource(team)
        self.sample_calls = 0

    def sample(self, rng: random.Random):
        self.sample_calls += 1
        return self._source.sample(rng)


class _TrackedBo3RLPlayer(RLPlayer):
    """Record the child-game to local-series mapping for the live Bo3 smoke test."""

    def __init__(self, *args, **kwargs):
        self.completed_child_games: list[tuple[str, str | None]] = []
        super().__init__(*args, **kwargs)

    def _battle_finished_callback(self, battle: AbstractBattle):
        if isinstance(battle, DoubleBattle):
            battle_key = self._battle_key(battle)
            state = self._series_by_battle.get(battle_key)
            self.completed_child_games.append((battle_key, None if state is None else state.key))
        super()._battle_finished_callback(battle)


@pytest.mark.stress
@pytest.mark.asyncio
async def test_local_rl_player_completes_bo3_against_random(showdown_server, stress_policy) -> None:
    """Play one complete local Showdown Bo3 and verify series state is reclaimed."""
    poke_env_patches.install()
    source = _CountingTeamSource(DEFAULT_TEST_TEAM)
    random_source = FixedTeamSource(DEFAULT_TEST_TEAM)
    rl_player = _TrackedBo3RLPlayer(
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
        max_concurrent_battles=1,
    )
    setattr(random_player.ps_client, "_p0_force_open_team_sheet", True)

    try:
        await asyncio.wait_for(rl_player.battle_against(random_player, n_battles=1), timeout=120.0)
        deadline = asyncio.get_running_loop().time() + 120.0
        while (len(rl_player.completed_child_games) < 2 or rl_player._series_by_opponent) and (
            asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)
    finally:
        await rl_player.ps_client.stop_listening()
        await random_player.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()

    child_games = rl_player.completed_child_games
    assert 2 <= len(child_games) <= 3
    assert len({battle_key for battle_key, _ in child_games}) == len(child_games)
    assert len({series_key for _, series_key in child_games}) == 1
    assert not rl_player._battle_histories
    assert not rl_player._series_by_opponent
    assert not rl_player._series_by_battle
    assert not rl_player._series_store._store
    # One sample initializes the series and one follows its final child game;
    # intermediate Bo3 games must not resample the team.
    assert source.sample_calls == 2
