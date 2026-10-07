"""Command-line interface for running the Showdown RL bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

from p0.cli import LOG_FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.rl_player import DEFAULT_BATTLE_FORMAT, RLPlayer, load_player_policy
from p0.runtime import poke_env_patches
from p0.teams.source import FileTeamSource, TeamSource, build_team_source
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore
from p0.training.config import BotConfig, GlobalConfig, load_config

logger = logging.getLogger("p0.cli.play")


def _build_parser() -> argparse.ArgumentParser:
    """Build the live-play parser."""
    parser = argparse.ArgumentParser(prog="p0-play", description="Run the Pokemon Showdown RL bot.")
    parser.add_argument("--config", type=Path, help="Path to the YAML application configuration.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the policy checkpoint; defaults to bot.checkpoint_path.",
    )
    team_source = parser.add_mutually_exclusive_group()
    team_source.add_argument(
        "--team-file",
        action="append",
        type=Path,
        help="Team file to include. Can be repeated.",
    )
    team_source.add_argument(
        "--team-pool",
        choices=("all", "reduced"),
        help="Named team pool under the teams directory (default: all).",
    )
    parser.add_argument(
        "--challenge-limit",
        type=int,
        help="Number of challenges to accept; defaults to bot.challenge_limit.",
    )
    parser.add_argument(
        "--opponent",
        help="Only accept challenges from this opponent username; defaults to bot.opponent.",
    )
    return parser


def _server_configuration(bot: BotConfig) -> ServerConfiguration:
    if bot.websocket_url is None and bot.authentication_url is None:
        return LocalhostServerConfiguration
    if bot.websocket_url is None or bot.authentication_url is None:
        raise ValueError(
            "bot.websocket_url and bot.authentication_url must be configured together."
        )
    return ServerConfiguration(bot.websocket_url, bot.authentication_url)


async def run_bot(
    app_config: GlobalConfig,
    checkpoint_path: Path | None,
    team_source: TeamSource,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    *,
    challenge_limit: int,
    opponent: str | None,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
) -> None:
    """Boot the RL bot and run its Showdown listener."""
    poke_env_patches.install()

    policy = load_player_policy(
        checkpoint_path,
        allow_random_init=app_config.bot.allow_random_init,
        policy_store=policy_store,
    )
    bot_player = RLPlayer(
        policy=policy,
        top_p=app_config.bot.top_p,
        observation_builder=ObservationBuilder(policy.resources),
        team_rng=random.Random(),
        account_configuration=account_configuration,
        battle_format=DEFAULT_BATTLE_FORMAT,
        server_configuration=server_configuration,
        team_source=team_source,
        max_concurrent_battles=1,
    )

    logger.info(
        "Starting RL bot as '%s' against %s (checkpoint: %s)",
        account_configuration.username,
        server_configuration.websocket_url,
        checkpoint_path or "random-init",
    )

    try:
        await bot_player.accept_challenges(opponent, challenge_limit)
        await poke_env_patches.wait_for_parent_results(bot_player.ps_client, challenge_limit)
    except asyncio.CancelledError:
        logger.info("Bot listener received cancellation.")
    finally:
        await bot_player.ps_client.stop_listening()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the bot."""
    args = _build_parser().parse_args(argv)
    try:
        app_config = load_config(args.config)
        checkpoint_path = (
            args.checkpoint if args.checkpoint is not None else app_config.bot.checkpoint_path
        )
        challenge_limit = (
            app_config.bot.challenge_limit if args.challenge_limit is None else args.challenge_limit
        )
        if challenge_limit < 1:
            raise ValueError("--challenge-limit must be a positive integer.")
        opponent = app_config.bot.opponent if args.opponent is None else args.opponent
        username = os.getenv("SHOWDOWN_USERNAME", app_config.bot.username)
        if not username.strip():
            raise ValueError("SHOWDOWN_USERNAME must not be empty.")
        account_configuration = AccountConfiguration(
            username,
            os.getenv("SHOWDOWN_PASSWORD") or None,
        )
        server_configuration = _server_configuration(app_config.bot)
        team_source = (
            FileTeamSource.from_files(args.team_file)
            if args.team_file
            else build_team_source(
                app_config.teams.all if args.team_pool != "reduced" else app_config.teams.reduced,
                expected_format_id=DEFAULT_BATTLE_FORMAT,
            )
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Error preparing live play: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.getLevelNamesMapping()[app_config.bot.log_level.upper()],
        format=LOG_FORMAT,
    )

    try:
        asyncio.run(
            run_bot(
                app_config,
                checkpoint_path,
                team_source,
                account_configuration,
                server_configuration,
                challenge_limit=challenge_limit,
                opponent=opponent,
            )
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info("Exiting on user interrupt.")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Run error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
