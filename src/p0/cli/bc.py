"""Command-line entry point for replay behaviour cloning."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from p0.training.bc_composition import evaluate_bc, train_bc
from p0.training.config import BCConfig, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-bc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("config.yaml"))
        command.add_argument("--shard-manifest", type=Path, required=True)
        command.add_argument("--split-manifest", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, default=None)
        command.add_argument("--device", default=None)
    train = subparsers.choices["train"]
    train.add_argument("--resume-checkpoint", type=Path, default=None)
    train.add_argument("--overfit", action="store_true")
    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    return parser


def _resolved_config(args: argparse.Namespace) -> BCConfig:
    config = load_config(args.config).bc
    output_dir = config.output_dir if args.output_dir is None else args.output_dir.resolve()
    resume = getattr(args, "resume_checkpoint", None)
    return replace(
        config,
        shard_manifest=args.shard_manifest.resolve(),
        split_manifest=args.split_manifest.resolve(),
        output_dir=output_dir,
        resume_checkpoint=(config.resume_checkpoint if resume is None else resume.resolve()),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = _resolved_config(args)
    if args.command == "train":
        result = train_bc(config, overfit=args.overfit, device=args.device)
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
