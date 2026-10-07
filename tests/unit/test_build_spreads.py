"""Spread generation reads selected dex metadata and reuses month-specific bytes."""

import json
import shutil
from pathlib import Path

from p0.cli.build_spreads import main
from p0.paths import DEFAULT_PATHS


class TestBuildSpreads:
    def test_cached_month_is_reused_without_changing_source_bytes(self, tmp_path: Path) -> None:
        source = json.loads((DEFAULT_PATHS.data_root / "spread_usage.json").read_text())["source"]
        month = source["month"]
        cache = tmp_path / "usage" / month
        cache.mkdir(parents=True)
        for name in source["exports"]:
            shutil.copyfile(DEFAULT_PATHS.data_root / "raw" / "usage" / month / name, cache / name)
        before = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache.iterdir()
        }
        output = tmp_path / "spreads.json"

        assert (
            main(
                [
                    "--usage-dir",
                    str(cache.parent),
                    "--month",
                    month,
                    "--fetch",
                    "--out",
                    str(output),
                ]
            )
            == 0
        )

        generated = json.loads(output.read_text())
        assert generated["source"] == source
        dex_source = json.loads((DEFAULT_PATHS.data_root / "champions_dex.json").read_text())[
            "source"
        ]
        assert generated["format_id"] == dex_source["battleFormat"]
        assert {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache.iterdir()
        } == before
        assert not list(cache.parent.glob("*.json"))
