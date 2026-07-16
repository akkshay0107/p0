"""Composition functions for live environments and players."""

from __future__ import annotations

import random
from typing import TypeVar

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.runtime.env import SimEnv
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import TeamSource

EnvT = TypeVar("EnvT", bound=SimEnv)


def build_sim_env(
    *,
    account_configuration1: AccountConfiguration,
    account_configuration2: AccountConfiguration,
    server_port: int,
    agent_team_source: TeamSource,
    opponent_team_source: TeamSource,
    observation_builder: ObservationBuilder,
    agent_seed: int = 0,
    opponent_seed: int = 1,
    env_type: type[EnvT] = SimEnv,
) -> EnvT:
    agent_rng = random.Random(agent_seed)
    opponent_rng = random.Random(opponent_seed)
    initial_agent_team = agent_team_source.sample(random.Random(agent_seed))
    env = env_type(
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        server_configuration=ServerConfiguration(
            f"ws://127.0.0.1:{server_port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=FORMAT.battle_format,
        accept_open_team_sheet=True,
        start_timer_on_battle_start=False,
        log_level=25,
        team=initial_agent_team.packed,
        observation_builder=observation_builder,
        battle_view_factory=battle_view,
        agent_team_source=agent_team_source,
        opponent_team_source=opponent_team_source,
        agent_rng=agent_rng,
        opponent_rng=opponent_rng,
    )
    env.agent2.update_team(opponent_team_source.sample(random.Random(opponent_seed)).packed)
    return env
