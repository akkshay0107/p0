"""Training command-line composition root."""

from __future__ import annotations

import argparse
import signal
import threading

from p0.training.config import load_config
from p0.training.ppo_runner import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the p0 VGC policy.")
    parser.add_argument("--config", help="Path to the YAML application configuration.")
    parser.add_argument(
        "--agent-team-source",
        choices=("all", "reduced"),
        default="all",
        help="Team pool used by the self-play agent; the opponent always uses all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stop = threading.Event()
    previous = {
        name: signal.signal(name, lambda *_: stop.set()) for name in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run_training(
            load_config(args.config),
            cancel_requested=stop.is_set,
            agent_team_source=args.agent_team_source,
        )
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)
    return 0
