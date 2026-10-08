"""
Build the empirical Stat Point spread priors from Showdown usage exports.

Raw exports are cached by month. --fetch downloads missing exports and reuses
existing bytes without downloading them again.
"""

from __future__ import annotations

import argparse
import gzip
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import orjson

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

USAGE_URL = "https://www.smogon.com/stats/{month}/chaos/{format_id}-{cutoff}.json.gz"


def fetch_usage_export(month: str, format_id: str, cutoff: int, destination: Path) -> Path:
    """
    Download and decompress one chaos export, returning the written path.

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
    month: str,
) -> dict[str, Any]:
    """Blend both usage exports into the spread-prior artifact and write it."""
    dex = _read_chaos(dex_path)
    payload = build_spread_table(
        _read_chaos(bo1_path),
        _read_chaos(bo3_path),
        format_id=dex["source"]["battleFormat"],
        dex=dex,
        max_spreads=max_spreads,
        min_nature_share=min_nature_share,
    )

    # Record the usage month and downloaded export names.
    payload["source"] = {
        "month": month,
        "exports": [bo1_path.name, bo3_path.name],
    }

    # Round-trip before writing so a malformed artifact fails here rather than at
    # the first battle that needs a lookup.
    load_spread_table(payload)
    atomic_json_save(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Build spread priors CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="Usage month, as YYYY-MM.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download missing month-specific usage exports before building.",
    )
    args = parser.parse_args(argv)

    date.fromisoformat(f"{args.month}-01")

    source = _read_chaos(DEFAULT_DEX)["source"]
    usage_dir = DEFAULT_USAGE_DIR / args.month
    for format_id in (source["battleFormat"], source["bo3Format"]):
        if args.fetch and not (usage_dir / f"{format_id}-{DEFAULT_CUTOFF}.json").is_file():
            written = fetch_usage_export(args.month, format_id, DEFAULT_CUTOFF, usage_dir)
            print(f"Fetched {written.name} ({written.stat().st_size / 1024:.0f} KiB)")

    payload = build(
        usage_dir / f"{source['battleFormat']}-{DEFAULT_CUTOFF}.json",
        usage_dir / f"{source['bo3Format']}-{DEFAULT_CUTOFF}.json",
        DEFAULT_DEX,
        DEFAULT_OUTPUT,
        MAX_SPREADS_PER_BUCKET,
        MIN_NATURE_SHARE,
        args.month,
    )

    buckets = sum(len(by_nature) for by_nature in payload["spreads"].values())
    print(
        f"Wrote {DEFAULT_OUTPUT}: {len(payload['spreads'])} species, {buckets} buckets, "
        f"{DEFAULT_OUTPUT.stat().st_size / 1024:.0f} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
