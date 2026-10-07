"""Spread generation reads selected dex metadata and reuses month-specific bytes."""

import json
from pathlib import Path

from p0.cli.build_spreads import (
    DEFAULT_CUTOFF,
    DEFAULT_DEX,
    MAX_SPREADS_PER_BUCKET,
    MIN_NATURE_SHARE,
    build,
)
from p0.paths import DEFAULT_PATHS


class TestBuildSpreads:
    def test_build_does_not_modify_cached_exports(self, tmp_path: Path) -> None:
        source = json.loads((DEFAULT_PATHS.data_root / "spread_usage.json").read_text())["source"]
        month = source["month"]
        cache = DEFAULT_PATHS.data_root / "raw" / "usage" / month
        before = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache.iterdir()
        }
        output = tmp_path / "spreads.json"
        dex_source = json.loads(DEFAULT_DEX.read_text())["source"]
        generated = build(
            cache / f"{dex_source['battleFormat']}-{DEFAULT_CUTOFF}.json",
            cache / f"{dex_source['bo3Format']}-{DEFAULT_CUTOFF}.json",
            DEFAULT_DEX,
            output,
            MAX_SPREADS_PER_BUCKET,
            MIN_NATURE_SHARE,
            month,
        )
        assert generated["source"] == source
        assert generated["format_id"] == dex_source["battleFormat"]
        after = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache.iterdir()
        }
        assert after == before
