from __future__ import annotations

from urllib.parse import urlparse

import pytest
from poke_env import AccountConfiguration

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.model.observation_builder import ObservationBuilder
from p0.runtime import poke_env_patches
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.teams.source import FixedTeamSource
from p0.training.config import TrainingConfig
from p0.training.rollout import RolloutCollector
from p0.training.vector_env import ThreadVecEnv


@pytest.mark.integration
def test_two_simulated_bo3_series_run_concurrently(showdown_server, model_policy) -> None:
    """Run two real local PPO simulations concurrently through ThreadVecEnv."""
    server_port = urlparse(showdown_server.websocket_url).port
    if server_port is None:
        raise ValueError("The integration server configuration has no websocket port")

    agent_sources = [FixedTeamSource(DEFAULT_TEST_TEAM) for _ in range(2)]
    opponent_sources = [FixedTeamSource(DEFAULT_TEST_TEAM) for _ in range(2)]
    envs: list[SimEnv] = []
    vector_env: ThreadVecEnv | None = None
    completed_series: set[int] = set()
    completed_series_ids: set[str] = set()

    poke_env_patches.install()
    try:
        for index, (agent_source, opponent_source) in enumerate(
            zip(agent_sources, opponent_sources, strict=True)
        ):
            envs.append(
                build_sim_env(
                    account_configuration1=AccountConfiguration(f"PPOAgent{index}", None),
                    account_configuration2=AccountConfiguration(f"PPOOpponent{index}", None),
                    server_port=server_port,
                    agent_team_source=agent_source,
                    opponent_team_source=opponent_source,
                    observation_builder=ObservationBuilder(model_policy.resources),
                    agent_seed=index * 2,
                    opponent_seed=index * 2 + 1,
                )
            )

        vector_env = ThreadVecEnv(envs)
        vector_env.reset()
        collector = RolloutCollector(
            vector_env,
            model_policy,
            TrainingConfig(n_envs=2, rollout_steps=1),
        )

        for _ in range(1_200):
            active_series_ids = tuple(
                str(info["series_id"]) for info in (vector_env.last_infos or ())
            )
            collector.collect()
            infos = vector_env.last_infos
            if infos is None:
                raise AssertionError("ThreadVecEnv did not publish battle metadata")
            for index, info in enumerate(infos):
                if info.get("series_complete"):
                    completed_series.add(index)
                    completed_series_ids.add(active_series_ids[index])
            if completed_series == {0, 1}:
                break

        assert completed_series == {0, 1}
        assert len(collector.completed_trajectories) >= 8
        collector_state = collector.training_state()
        store1 = collector_state["series_store1"]
        store2 = collector_state["series_store2"]
        history1 = collector_state["series_history1"]
        history2 = collector_state["series_history2"]
        assert isinstance(store1, dict) and isinstance(store2, dict)
        assert isinstance(history1, dict) and isinstance(history2, dict)
        assert not completed_series_ids.intersection(store1)
        assert not completed_series_ids.intersection(store2)
        assert not completed_series_ids.intersection(history1)
        assert not completed_series_ids.intersection(history2)
    finally:
        if vector_env is not None:
            vector_env.shutdown()
        poke_env_patches.uninstall_for_tests()
