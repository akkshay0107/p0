from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import time

import pytest
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import DefaultBattleOrder

from p0.evaluation.harness import EvalRandomPlayer, EvaluationHarness
from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS
from p0.runtime import poke_env_patches
from p0.runtime.showdown import allocate_loopback_ports
from p0.teams.source import FixedTeamSource
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.fixture(scope="module")
def forced_result_server(showdown_assets, tmp_path_factory: pytest.TempPathFactory):
    """Run one isolated local server with result commands enabled for tie tests."""
    del showdown_assets
    directory = tmp_path_factory.mktemp("forced-results")
    preload = directory / "allow-forced-results.cjs"
    config_path = DEFAULT_PATHS.showdown_root / "config/config.js"
    preload.write_text(
        f"const config = require({json.dumps(str(config_path))});\n"
        "config.bindaddress = '127.0.0.1';\n"
        "const originalStartup = config.startuphook;\n"
        "config.startuphook = () => {\n"
        "  if (originalStartup) originalStartup();\n"
        "  global.Config.groups[' '].forcewin = true;\n"
        "};\n"
    )
    port = allocate_loopback_ports(1)[0]
    server_log = directory / "showdown.log"
    with server_log.open("w") as output:
        process = subprocess.Popen(
            [
                "node",
                "--require",
                str(preload),
                "pokemon-showdown",
                "start",
                "--no-security",
                "--skip-build",
                str(port),
            ],
            cwd=DEFAULT_PATHS.showdown_root,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Forced-result Showdown did not start: {server_log.read_text()}"
                        )
                    time.sleep(0.05)
            yield ServerConfiguration(
                f"ws://127.0.0.1:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)


class _ResultDrivingPlayer(EvalRandomPlayer):
    """Send result commands through real child battles on the isolated server."""

    def __init__(self, *, outcomes: tuple[str, ...], **kwargs) -> None:
        self.outcomes = outcomes
        self.commanded_games: set[str] = set()
        super().__init__(**kwargs)

    async def choose_move(self, battle):  # pyright: ignore[reportIncompatibleMethodOverride]
        if battle.teampreview or not any(battle.active_pokemon):
            return DefaultBattleOrder()
        if battle.battle_tag not in self.commanded_games:
            index = len(self.commanded_games)
            self.commanded_games.add(battle.battle_tag)
            outcome = self.outcomes[index]
            command = "/forcetie" if outcome == "tie" else f"/forcewin {outcome}"
            await self.ps_client.send_message(command, battle.battle_tag)
        return super().choose_move(battle)


@pytest.mark.heavy
class TestEvaluation:
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcomes", "expected_winner"),
        (
            (("a", "tie", "tie"), "a"),
            (("b", "tie", "tie"), "b"),
            (("a", "b", "tie"), None),
            (("a", "a"), "a"),
            (("a", "b", "a"), "a"),
        ),
    )
    async def test_parent_series_result_with_ties(
        self, forced_result_server, outcomes: tuple[str, ...], expected_winner: str | None
    ) -> None:
        label = "".join(outcome[0] for outcome in outcomes)
        name_a = f"TieA{label}"
        name_b = f"TieB{label}"
        commands = tuple(
            name_a if outcome == "a" else name_b if outcome == "b" else "tie"
            for outcome in outcomes
        )
        source = FixedTeamSource(DEFAULT_TEST_TEAM)
        poke_env_patches.install()
        first = _ResultDrivingPlayer(
            outcomes=commands,
            account_configuration=AccountConfiguration(name_a, None),
            battle_format=FORMAT.bo3_format,
            server_configuration=forced_result_server,
            team_source=source,
            team_rng=random.Random(0),
            max_concurrent_battles=1,
        )
        second = EvalRandomPlayer(
            account_configuration=AccountConfiguration(name_b, None),
            battle_format=FORMAT.bo3_format,
            server_configuration=forced_result_server,
            team_source=source,
            team_rng=random.Random(1),
            max_concurrent_battles=1,
        )
        try:
            await first.battle_against(second, n_battles=1)
            parent_a, winner = await asyncio.wait_for(
                poke_env_patches.wait_for_parent_result(first.ps_client, 1), timeout=30
            )
            parent_b, opponent_winner = await asyncio.wait_for(
                poke_env_patches.wait_for_parent_result(second.ps_client, 1), timeout=30
            )
            assert parent_a == parent_b
            assert winner == opponent_winner
            expected = (
                name_a if expected_winner == "a" else name_b if expected_winner == "b" else None
            )
            assert winner == expected
            assert len(first.commanded_games) == len(outcomes)
        finally:
            await first.ps_client.stop_listening()
            await second.ps_client.stop_listening()
            poke_env_patches.uninstall_for_tests()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evaluation_harness_completes_matchup(self, showdown_server) -> None:
        """
        Verify EvaluationHarness runs a live matchup between baseline random players and tracks stats.

        Checks that the matchup completes without unhandled exceptions, computes win rates,
        aggregates results per team archetype, and produces a complete dictionary serialization
        containing confidence intervals and metadata suitable for logging.
        """
        poke_env_patches.install()

        harness = EvaluationHarness(
            episodes_per_matchup=1,
            seed=42,
        )
        fallback = FixedTeamSource(DEFAULT_TEST_TEAM)

        try:
            result = await harness.run_matchup(
                name_a="RandomA",
                policy_a=None,
                name_b="RandomB",
                policy_b=None,
                team_category="fallback",
                team_source=fallback,
                server_configuration=showdown_server,
            )
        finally:
            poke_env_patches.uninstall_for_tests()

        assert result.policy_a == "RandomA"
        assert result.policy_b == "RandomB"
        assert result.team_category == "fallback"
        assert result.total_games == 1
        assert result.wins_a + result.wins_b == 1
        assert result.ties == 0
        assert result.win_rate_a in (0.0, 1.0)
        assert len(result.per_team_results) == 1
        stats = next(iter(result.per_team_results.values()))
        assert stats["games"] == 1
        assert stats["wins"] == result.wins_a
        assert stats["win_rate"] == pytest.approx(stats["wins"] / stats["games"])

        # Verify dictionary serialization schema matches downstream reporting contracts
        dct = result.to_dict()
        assert set(dct) == {
            "policy_a",
            "policy_b",
            "team_category",
            "total_games",
            "wins_a",
            "wins_b",
            "ties",
            "win_rate_a",
            "confidence_interval_a",
            "per_team_results",
            "source_description",
            "per_team_a_results",
            "per_team_b_results",
        }
        assert dct["confidence_interval_a"] == list(result.confidence_interval_a)
        assert dct["total_games"] == 1
        assert len(dct["per_team_results"]) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy_side", ("a", "b"))
    async def test_live_evaluation_runs_model_policy_against_random_opponent(
        self,
        showdown_server,
        model_policy,
        policy_side: str,
    ) -> None:
        """Verify live evaluation works symmetrically when the neural policy is player A or player B."""
        # Assign the model policy to the parametrized player side and leave the other as None (RandomPlayer)
        policy_a = model_policy if policy_side == "a" else None
        policy_b = model_policy if policy_side == "b" else None
        name_a = "ModelA" if policy_a is not None else "RandomA"
        name_b = "ModelB" if policy_b is not None else "RandomB"

        poke_env_patches.install()
        try:
            result = await EvaluationHarness(
                episodes_per_matchup=1,
                seed=23,
            ).run_matchup(
                name_a=name_a,
                policy_a=policy_a,
                name_b=name_b,
                policy_b=policy_b,
                team_category="fallback",
                team_source=FixedTeamSource(DEFAULT_TEST_TEAM),
                server_configuration=showdown_server,
            )
        finally:
            poke_env_patches.uninstall_for_tests()

        assert (result.policy_a, result.policy_b) == (name_a, name_b)
        assert result.total_games == 1
        assert result.wins_a + result.wins_b + result.ties == result.total_games
        assert result.win_rate_a == pytest.approx(result.wins_a / result.total_games)
        assert result.per_team_a_results
        assert result.per_team_b_results
        assert sum(stats["games"] for stats in result.per_team_a_results.values()) == 1
        assert sum(stats["games"] for stats in result.per_team_b_results.values()) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_evaluation_matchup_serializes_per_team_outcomes(
        self, showdown_server
    ) -> None:
        poke_env_patches.install()
        try:
            harness = EvaluationHarness(episodes_per_matchup=2, seed=17)
            result = await harness.run_matchup(
                name_a="random-a",
                policy_a=None,
                name_b="random-b",
                policy_b=None,
                team_category="fallback",
                team_source=FixedTeamSource(DEFAULT_TEST_TEAM),
                server_configuration=showdown_server,
            )
        finally:
            poke_env_patches.uninstall_for_tests()

        assert result.total_games == 2
        assert result.wins_a + result.wins_b + result.ties == result.total_games

        # Validate accounting consistency across overall per_team, player A per-team, and player B per-team maps
        aggregates = (
            (result.per_team_results, result.wins_a),
            (result.per_team_a_results, result.wins_a),
            (result.per_team_b_results, result.wins_b),
        )
        for stats_by_team, expected_wins in aggregates:
            assert stats_by_team
            assert all(
                set(stats) == {"wins", "games", "win_rate"} for stats in stats_by_team.values()
            )
            # The sum of games across all team keys must equal the total matchup episode count
            assert sum(stats["games"] for stats in stats_by_team.values()) == result.total_games
            # The sum of recorded wins across teams must equal the corresponding player's total wins
            assert sum(stats["wins"] for stats in stats_by_team.values()) == expected_wins
            assert all(
                0 <= stats["wins"] <= stats["games"]
                and stats["win_rate"] == pytest.approx(stats["wins"] / stats["games"])
                for stats in stats_by_team.values()
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("opponent_type", ("max_power", "simple_heuristics"))
    async def test_live_evaluation_runs_against_heuristic_opponents(
        self,
        showdown_server,
        model_policy,
        opponent_type: str,
    ) -> None:
        """Verify live evaluation completes matchups against MaxBasePowerPlayer and SimpleHeuristicsPlayer."""
        poke_env_patches.install()
        try:
            harness = EvaluationHarness(episodes_per_matchup=1, seed=42)
            result = await harness.run_matchup(
                name_a="ModelPlayer",
                policy_a=model_policy,
                name_b=f"Opponent_{opponent_type}",
                policy_b=opponent_type,
                team_category="test_heuristics",
                team_source=FixedTeamSource(DEFAULT_TEST_TEAM),
                server_configuration=showdown_server,
            )
        finally:
            poke_env_patches.uninstall_for_tests()

        assert result.policy_a == "ModelPlayer"
        assert result.policy_b == f"Opponent_{opponent_type}"
        assert result.total_games >= 1
        assert result.wins_a + result.wins_b + result.ties == result.total_games
        assert 0.0 <= result.win_rate_a <= 1.0
