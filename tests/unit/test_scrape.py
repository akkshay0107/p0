"""Tests for replay statistics and scraping."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from p0.replays.schema import FetchIndexEntry
from p0.replays.scrape import (
    ReplayFetcher,
    ScrapeConfig,
    read_fetch_index,
    select_replays,
    write_fetch_index,
)


class TestReplayScraping:
    def test_scrape_config_validation(self, tmp_path: Path) -> None:
        """Verify ScrapeConfig raises ValueError for empty format_id, non-positive page_size, negative backoff, or zero timeout."""
        with pytest.raises(ValueError, match="format_id must be non-empty"):
            ScrapeConfig(format_id="", cache_dir=tmp_path)
        with pytest.raises(ValueError, match="page_size must be positive"):
            ScrapeConfig(format_id="test", cache_dir=tmp_path, page_size=0)
        with pytest.raises(ValueError, match="backoff and rate limit must be nonnegative"):
            ScrapeConfig(format_id="test", cache_dir=tmp_path, backoff_seconds=-1.0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ScrapeConfig(format_id="test", cache_dir=tmp_path, timeout_seconds=0.0)

    def test_fetcher_recovery(self, tmp_path: Path) -> None:
        """Verify unindexed raw and metadata files are recovered and indexed without rewriting metadata."""
        replay_id = "gen9stress-recover"
        config = ScrapeConfig(
            format_id="gen9stress", cache_dir=tmp_path, retries=1, rate_limit_per_second=0
        )
        raw_path = tmp_path / config.format_id / "raw" / f"{replay_id}.json.gz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = json.dumps(
            {"id": replay_id, "format": "gen9stress", "log": "|turn|1"}
        ).encode()
        raw_path.write_bytes(gzip.compress(raw_payload, mtime=0))

        fetcher = ReplayFetcher(config)
        entries = fetcher.acquire((replay_id,))

        assert len(entries) == 1
        assert entries[0].replay_id == replay_id
        assert entries[0].raw_path == f"raw/{replay_id}.json.gz"
        assert read_fetch_index(fetcher.index_path) == tuple(entries)

    def test_fetch_index_rejects_invalid_json(self, tmp_path: Path) -> None:
        index = tmp_path / "index.jsonl"
        index.write_bytes(b"not-json\n")
        with pytest.raises(ValueError, match="Malformed fetch index"):
            read_fetch_index(index)


class TestReplayCatalog:
    def test_filters_ratings_dates_and_players_and_excludes_unknown_metadata(
        self, tmp_path: Path
    ) -> None:
        rows = (
            FetchIndexEntry(
                "r1",
                "f",
                "https://example.com/r1",
                "2026-01-01T00:00:00Z",
                200,
                "raw/r1.json.gz",
                "2026-01-05T00:00:00Z",
                1800,
                ("Alice", "Bob"),
            ),
            FetchIndexEntry(
                "r2",
                "f",
                "https://example.com/r2",
                "2026-01-01T00:00:00Z",
                200,
                "raw/r2.json.gz",
                "2025-12-31T00:00:00Z",
                1400,
                ("Alice", "Carol"),
            ),
            FetchIndexEntry(
                "r3", "f", "https://example.com/r3", "2026-01-01T00:00:00Z", 200, "raw/r3.json.gz"
            ),
        )
        path = tmp_path / "index.jsonl"
        write_fetch_index(path, rows)
        catalog = read_fetch_index(path)
        selected = select_replays(
            catalog,
            format_id="f",
            min_rating=1500,
            max_rating=1900,
            after="2026-01-01",
            before="2026-01-06",
            player="alice",
        )
        assert [row.replay_id for row in selected] == ["r1"]
        assert len(select_replays(catalog)) == 3
