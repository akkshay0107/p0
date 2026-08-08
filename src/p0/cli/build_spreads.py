"""Build the empirical Stat Point spread priors from Showdown usage exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_json_save
from p0.teams.spread_usage import (
    MAX_SPREADS_PER_BUCKET,
    MIN_NATURE_SHARE,
    build_spread_table,
    load_spread_table,
)

ROOT = DEFAULT_PATHS.repository_root
DEFAULT_USAGE_DIR = ROOT / "data" / "raw" / "usage"
DEFAULT_DEX = ROOT / "data" / "champions_dex.json"
DEFAULT_OUTPUT = ROOT / "data" / "spread_usage.json"
DEFAULT_CUTOFF = 1760


def _read_chaos(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Usage export not found: {path}. Download it from "
            "https://www.smogon.com/stats/<YYYY-MM>/chaos/ into data/raw/usage/"
        )
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise ValueError(f"Malformed usage export: {path}") from exc


def build(
    bo1_path: Path,
    bo3_path: Path,
    dex_path: Path,
    output_path: Path,
    max_spreads: int,
    min_nature_share: float,
) -> dict[str, Any]:
    """Blend both usage exports into the spread-prior artifact and write it."""
    payload = build_spread_table(
        _read_chaos(bo1_path),
        _read_chaos(bo3_path),
        format_id=FORMAT.battle_format,
        dex=_read_chaos(dex_path),
        max_spreads=max_spreads,
        min_nature_share=min_nature_share,
    )

    # Round-trip before writing so a malformed artifact fails here rather than at
    # the first battle that needs a lookup.
    load_spread_table(payload)
    atomic_json_save(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Build spread priors CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage-dir", type=Path, default=DEFAULT_USAGE_DIR)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--dex", type=Path, default=DEFAULT_DEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-spreads", type=int, default=MAX_SPREADS_PER_BUCKET)
    parser.add_argument("--min-nature-share", type=float, default=MIN_NATURE_SHARE)
    args = parser.parse_args(argv)

    payload = build(
        args.usage_dir / f"{FORMAT.battle_format}-{args.cutoff}.json",
        args.usage_dir / f"{FORMAT.bo3_format}-{args.cutoff}.json",
        args.dex,
        args.out,
        args.max_spreads,
        args.min_nature_share,
    )

    buckets = sum(len(by_nature) for by_nature in payload["spreads"].values())
    print(
        f"Wrote {args.out}: {len(payload['spreads'])} species, {buckets} buckets, "
        f"{args.out.stat().st_size / 1024:.0f} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
