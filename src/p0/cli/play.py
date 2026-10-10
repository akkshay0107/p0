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
from p0.paths import DEFAULT_PATHS
from p0.rl_player import DEFAULT_BATTLE_FORMAT, RLPlayer, load_player_policy
from p0.runtime import poke_env_patches
from p0.teams.corpus import TeamCorpus, corpus_from_team_files, load_team_corpus
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore
from p0.training.config import BotConfig, load_config

logger = logging.getLogger("p0.cli.play")

DEFAULT_CHALLENGE_LIMIT = 1_000_000
DEFAULT_TOP_P = 0.9


def _build_parser() -> argparse.ArgumentParser:
    """Build the live-play parser."""
    parser = argparse.ArgumentParser(prog="p0-play", description="Run the Pokemon Showdown RL bot.")
    parser.add_argument("--config", type=Path, help="Path to the YAML application configuration.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to the policy checkpoint; omit only with --allow-random-init.",
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
        default=DEFAULT_CHALLENGE_LIMIT,
        help="Number of challenges to accept.",
    )
    parser.add_argument(
        "--opponent",
        help="Only accept challenges from this opponent username.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Sample each action from the smallest set of actions with this total probability.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Play with untrained random weights when no checkpoint is given.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
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
    checkpoint_path: Path | None,
    team_source: TeamCorpus,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    *,
    challenge_limit: int,
    opponent: str | None,
    top_p: float = DEFAULT_TOP_P,
    allow_random_init: bool = False,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
) -> None:
    """Boot the RL bot and run its Showdown listener."""
    poke_env_patches.install()

    policy = load_player_policy(
        checkpoint_path,
        allow_random_init=allow_random_init,
        policy_store=policy_store,
    )
    bot_player = RLPlayer(
        policy=policy,
        top_p=top_p,
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
        bot = load_config(args.config).bot
        if args.challenge_limit < 1:
            raise ValueError("--challenge-limit must be a positive integer.")
        if not 0.0 < args.top_p <= 1.0:
            raise ValueError("--top-p must be in (0, 1].")
        username = os.getenv("SHOWDOWN_USERNAME", bot.username)
        if not username.strip():
            raise ValueError("SHOWDOWN_USERNAME must not be empty.")
        account_configuration = AccountConfiguration(
            username,
            os.getenv("SHOWDOWN_PASSWORD") or None,
        )
        server_configuration = _server_configuration(bot)
        team_source = (
            corpus_from_team_files(args.team_file, DEFAULT_BATTLE_FORMAT)
            if args.team_file
            else load_team_corpus(
                DEFAULT_PATHS.teams_root / (args.team_pool or "all"),
                DEFAULT_BATTLE_FORMAT,
            )
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Error preparing live play: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.getLevelNamesMapping()[args.log_level],
        format=LOG_FORMAT,
    )

    try:
        asyncio.run(
            run_bot(
                args.checkpoint,
                team_source,
                account_configuration,
                server_configuration,
                challenge_limit=args.challenge_limit,
                opponent=args.opponent,
                top_p=args.top_p,
                allow_random_init=args.allow_random_init,
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
