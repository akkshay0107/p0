from __future__ import annotations

from urllib.parse import urlparse

import pytest
from poke_env import AccountConfiguration

from p0.battle.actions import PASS_ACTION
from p0.model.observation_builder import ObservationBuilder
from p0.runtime import poke_env_patches
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.training.config import PPOConfig
from p0.training.rollout import RolloutCollector
from p0.training.trajectory import CollectedTrajectory
from p0.training.vector_env import ThreadVecEnv
from tests.team_fixtures import default_test_corpus


@pytest.mark.heavy
class TestPpoSimulation:
    @pytest.mark.integration
    def test_two_simulated_bo3_series_run_concurrently(self, showdown_server, model_policy) -> None:
        """Run two real local PPO simulations concurrently through ThreadVecEnv."""
        server_port = urlparse(showdown_server.websocket_url).port
        if server_port is None:
            raise ValueError("The integration server configuration has no websocket port")

        agent_sources = [default_test_corpus() for _ in range(2)]
        opponent_sources = [default_test_corpus() for _ in range(2)]
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
                PPOConfig(n_envs=2, rollout_steps=1),
            )

            uneven_game = False
            previous_games: dict[int, tuple[str, tuple[CollectedTrajectory, ...]]] = {}
            final_entry_checks = 0
            for _ in range(1_200):
                completed_before = len(collector.completed_trajectories)
                collector.collect()
                infos = vector_env.last_infos
                finished_games = vector_env.last_finished
                if infos is None or finished_games is None:
                    raise AssertionError("ThreadVecEnv did not publish battle metadata")
                # Games finishing in one step are completed in environment order,
                # seat 1 then seat 2.
                new_trajectories = collector.completed_trajectories[completed_before:]
                finished = [
                    (index, game) for index, game in enumerate(finished_games) if game is not None
                ]
                for (index, game), seats in zip(
                    finished,
                    zip(new_trajectories[::2], new_trajectories[1::2], strict=True),
                    strict=True,
                ):
                    previous = previous_games.get(index)
                    if previous is not None and previous[0] == game.series_id:
                        for prior, current in zip(previous[1], seats, strict=True):
                            # A finished game adds its final board after the recorded
                            # decisions; a truncated game adds nothing.
                            final_entries = int(prior.dones[-1].item())
                            assert current.series_history[-1].size(0) == (
                                prior.length + final_entries
                            )
                            final_entry_checks += final_entries
                    previous_games[index] = (game.series_id, seats)
                    if game.series_complete:
                        # The reset started a new series, which the info describes.
                        assert infos[index]["series_id"] != game.series_id
                        completed_series.add(index)
                        completed_series_ids.add(game.series_id)
                # Each game appends seat 1 then seat 2. Opponent-only replacements give
                # the seats different decision counts, which game completion must accept;
                # keep playing until at least one such game has finished.
                trajectories = collector.completed_trajectories
                uneven_game = any(
                    first.length != second.length
                    for first, second in zip(trajectories[::2], trajectories[1::2], strict=True)
                )
                if completed_series == {0, 1} and uneven_game:
                    break

            assert completed_series == {0, 1}
            assert uneven_game
            assert final_entry_checks > 0
            trajectories = collector.completed_trajectories
            assert len(trajectories) >= 8
            for trajectory in trajectories:
                # A waiting seat's double-pass step is not a decision and is never recorded.
                pass_only = trajectory.action_masks[..., PASS_ACTION] & (
                    trajectory.action_masks.sum(-1) == 1
                )
                assert not pass_only.all(dim=-1).any()
                assert not trajectory.rewards[:-1].any()
        finally:
            if vector_env is not None:
                vector_env.shutdown()
            poke_env_patches.uninstall_for_tests()
