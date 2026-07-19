"""Command-line composition roots for replay scraping and compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p0.replays.compile import compile_raw_cache, write_compilation
from p0.replays.scrape import ReplayFetcher, ScrapeConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-replays")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scrape = subparsers.add_parser("scrape")
    scrape.add_argument("--format-id", required=True)
    scrape.add_argument("--cache-dir", type=Path, required=True)
    scrape.add_argument("--max-pages", type=int, default=1)
    scrape.add_argument("--concurrency", type=int, default=4)
    scrape.add_argument("--replay-id", action="append", default=None)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--format-id", required=True)
    compile_parser.add_argument("--cache-dir", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--max-candidates", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "scrape":
        config = ScrapeConfig(
            format_id=args.format_id,
            cache_dir=args.cache_dir,
            max_pages=args.max_pages,
            concurrency=args.concurrency,
        )
        entries = ReplayFetcher(config).acquire(args.replay_id)
        print(json.dumps([entry.to_dict() for entry in entries], sort_keys=True))
        return
    result = compile_raw_cache(
        args.cache_dir,
        format_id=args.format_id,
        max_candidates=args.max_candidates,
    )
    write_compilation(result, args.output)
    print(json.dumps(result.metrics.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
