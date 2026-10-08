"""Tests for replay statistics and scraping."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_STAT_PROVENANCE,
    NUM_IDX_LEVEL_STATS,
    StatProvenance,
)
from p0.replays.compile import (
    _replay_observation,
    compile_payloads,
)
from p0.replays.schema import FetchIndexEntry
from p0.replays.scrape import (
    ReplayFetcher,
    ScrapeConfig,
    read_fetch_index,
    select_replays,
    write_fetch_index,
)
from tests.unit.replay_fixtures import payload_with_ots_natures


class TestReplayScraping:
    def test_replay_stats_are_imputed_from_the_pinned_dex(self) -> None:
        result = compile_payloads((payload_with_ots_natures("imputed"),))

        assert result.metrics.counters["imputations"] > 0
        assert result.metrics.counters["imputation_confidence_sum"] > 0.0

    def test_replay_stats_are_applied_to_both_projected_sides(self) -> None:
        """Verify numeric replay stat estimates reach both projected observation perspectives."""
        result = compile_payloads((payload_with_ots_natures("own-side"),))
        assert result.accepted_series

        game = result.accepted_series[0].games[0]
        pikachu_estimate = next(
            est
            for est in game.stat_estimates
            if est.member_id.side.value == "p1" and est.member_id.roster_index == 0
        )
        assert pikachu_estimate.values == (111, 107, 61, 63, 71, 155)
        assert pikachu_estimate.provenance == "IMPUTED"

        stat_overrides = {estimate.member_id: estimate.values for estimate in game.stat_estimates}
        builder = ObservationBuilder(default_runtime_resources())
        expected_pikachu = [111 / 300, 107 / 300, 61 / 300, 63 / 300, 71 / 300, 155 / 300]

        # Perspective 0: Pikachu is on player team (slot 0)
        p0_obs = _replay_observation(
            game.perspectives[0].snapshots[0].view, builder, stat_overrides
        )
        assert p0_obs.categorical[0, CAT_IDX_STAT_PROVENANCE] == StatProvenance.IMPUTED
        p0_pikachu = p0_obs.numerical[0, NUM_IDX_LEVEL_STATS : NUM_IDX_LEVEL_STATS + 6].tolist()
        assert p0_pikachu == pytest.approx(expected_pikachu, rel=1e-5)

        # Perspective 1: Pikachu is on opponent team (slot 6)
        p1_obs = _replay_observation(
            game.perspectives[1].snapshots[0].view, builder, stat_overrides
        )
        assert p1_obs.categorical[6, CAT_IDX_STAT_PROVENANCE] == StatProvenance.IMPUTED
        p1_pikachu = p1_obs.numerical[6, NUM_IDX_LEVEL_STATS : NUM_IDX_LEVEL_STATS + 6].tolist()
        assert p1_pikachu == pytest.approx(expected_pikachu, rel=1e-5)

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
