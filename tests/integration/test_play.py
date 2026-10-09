"""Live completion tests for the play command."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from p0.format_config import FORMAT
from p0.runtime import poke_env_patches
from p0.teams.corpus import pack_showdown_team
from p0.training.checkpoint import CheckpointStore
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
@pytest.mark.integration
class TestPlayCommand:
    @pytest.mark.asyncio
    async def test_challenge_limit_waits_for_parent_series(
        self, showdown_server, model_policy, tmp_path: Path
    ) -> None:
        """The command exits only after all children of its accepted Bo3 finish."""
        checkpoint = tmp_path / "policy.pt"
        team_file = tmp_path / "team.txt"
        config_file = tmp_path / "play-config.yaml"
        log_file = tmp_path / "play.log"
        CheckpointStore().save_policy(checkpoint, model_policy)
        team_file.write_text(DEFAULT_TEST_TEAM)
        config_file.write_text(
            "bot:\n"
            f"  websocket_url: {json.dumps(showdown_server.websocket_url)}\n"
            f"  authentication_url: {json.dumps(showdown_server.authentication_url)}\n",
            encoding="utf-8",
        )
        poke_env_patches.install()

        opponent = RandomPlayer(
            battle_format=FORMAT.bo3_format,
            server_configuration=showdown_server,
            team=pack_showdown_team(DEFAULT_TEST_TEAM),
            account_configuration=AccountConfiguration("PlayOpponent", None),
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
        poke_env_patches.enable_forced_open_team_sheet(opponent)
        process: subprocess.Popen[str] | None = None
        try:
            with log_file.open("w") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "p0.cli.play",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                        "--team-file",
                        str(team_file),
                        "--challenge-limit",
                        "1",
                        "--opponent",
                        "PlayOpponent",
                    ],
                    stdout=output,
                    stderr=output,
                    text=True,
                    env=os.environ
                    | {"SHOWDOWN_USERNAME": "PlayLifecycleBot", "SHOWDOWN_PASSWORD": ""},
                )
                deadline = asyncio.get_running_loop().time() + 30
                while "|updateuser| PlayLifecycleBot|1" not in log_file.read_text():
                    if process.poll() is not None:
                        pytest.fail(f"Play command exited during startup: {log_file.read_text()}")
                    if asyncio.get_running_loop().time() >= deadline:
                        pytest.fail(f"Play command did not log in: {log_file.read_text()}")
                    await asyncio.sleep(0.1)

                await asyncio.wait_for(opponent.send_challenges("PlayLifecycleBot", 1), timeout=120)
                exit_code = await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=120)

            assert exit_code == 0, log_file.read_text()
            assert 2 <= opponent.n_finished_battles <= 3
            assert all(battle.finished for battle in opponent.battles.values())
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                await asyncio.to_thread(process.wait, timeout=10)
            await opponent.ps_client.stop_listening()
            poke_env_patches.uninstall_for_tests()
