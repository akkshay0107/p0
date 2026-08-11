"""Build the empirical Stat Point spread priors from Showdown usage exports.

The raw exports live under a gitignored cache, so a clean checkout cannot rebuild
the artifact without re-downloading them. --fetch performs that download so a
rebuild is reproducible from the repository alone.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

# The usage month the shipped artifact is built from. Bumping this is the intended
# way to refresh the priors; rebuild and re-verify accuracy before committing.
DEFAULT_MONTH = "2026-07"
USAGE_URL = "https://www.smogon.com/stats/{month}/chaos/{format_id}-{cutoff}.json.gz"


def fetch_usage_export(month: str, format_id: str, cutoff: int, destination: Path) -> Path:
    """Download and decompress one chaos export, returning the written path.

    Arguments:
        month: Usage month in YYYY-MM form.
        format_id: Showdown format the export covers.
        cutoff: Rating cutoff the export was computed at.
        destination: Directory to write the decompressed JSON into.

    Returns:
        The path of the decompressed export.
    """
    url = USAGE_URL.format(month=month, format_id=format_id, cutoff=cutoff)
    request = Request(url, headers={"User-Agent": "p0-usage-acquirer/1"})
    try:
        with urlopen(request, timeout=120) as response:
            payload = gzip.decompress(response.read())
    except (HTTPError, URLError, OSError, gzip.BadGzipFile) as exc:
        raise RuntimeError(f"Failed to download usage export: {url}") from exc

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{format_id}-{cutoff}.json"
    path.write_bytes(payload)
    return path


def _read_chaos(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Usage export not found: {path}. Re-download it with 'p0-build-spreads --fetch'."
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
    month: str = DEFAULT_MONTH,
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

    # Name the exports the artifact came from, so a stale month is visible in the
    # committed file rather than only in whoever ran the build.
    payload["source"] = {
        "month": month,
        "exports": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (bo1_path, bo3_path)
        },
    }

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
    parser.add_argument("--month", default=DEFAULT_MONTH, help="Usage month, as YYYY-MM.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download the usage exports before building instead of reusing the cache.",
    )
    parser.add_argument("--dex", type=Path, default=DEFAULT_DEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-spreads", type=int, default=MAX_SPREADS_PER_BUCKET)
    parser.add_argument("--min-nature-share", type=float, default=MIN_NATURE_SHARE)
    args = parser.parse_args(argv)

    if args.fetch:
        for format_id in (FORMAT.battle_format, FORMAT.bo3_format):
            written = fetch_usage_export(args.month, format_id, args.cutoff, args.usage_dir)
            print(f"Fetched {written.name} ({written.stat().st_size / 1024:.0f} KiB)")

    payload = build(
        args.usage_dir / f"{FORMAT.battle_format}-{args.cutoff}.json",
        args.usage_dir / f"{FORMAT.bo3_format}-{args.cutoff}.json",
        args.dex,
        args.out,
        args.max_spreads,
        args.min_nature_share,
        args.month,
    )

    buckets = sum(len(by_nature) for by_nature in payload["spreads"].values())
    print(
        f"Wrote {args.out}: {len(payload['spreads'])} species, {buckets} buckets, "
        f"{args.out.stat().st_size / 1024:.0f} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
