"""Training command-line composition root."""

from __future__ import annotations

import argparse

from p0.training.config import load_config
from p0.training.files import cancellation_signals
from p0.training.ppo_runner import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the p0 VGC policy.")
    parser.add_argument("--config", help="Path to the YAML application configuration.")
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Sample one seat's teams from the reduced corpus; the other seat keeps all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with cancellation_signals() as cancel_requested:
        run_training(
            load_config(args.config),
            cancel_requested=cancel_requested,
            reduced=args.reduced,
        )
    return 0
