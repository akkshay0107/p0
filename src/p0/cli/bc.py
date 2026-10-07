"""Command-line entry point for replay behaviour cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p0.training.bc_runner import evaluate_bc, train_bc
from p0.training.config import load_config
from p0.training.files import cancellation_signals


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser for BC training and evaluation."""
    parser = argparse.ArgumentParser(prog="p0-bc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config", type=Path, help="Path to the YAML application configuration."
        )
        command.add_argument("--device", default=None)

    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """BC CLI entrypoint."""
    args = _parser().parse_args(argv)
    config = load_config(args.config).bc

    if args.command == "train":
        with cancellation_signals() as cancel_requested:
            result = train_bc(config, device=args.device, cancel_requested=cancel_requested)
    else:
        result = evaluate_bc(
            config,
            args.checkpoint.resolve(),
            split=args.split,
            device=args.device,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
