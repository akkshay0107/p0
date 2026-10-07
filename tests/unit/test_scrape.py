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
from p0.replays.schema import FetchMetadata
from p0.replays.scrape import (
    ReplayFetcher,
    ScrapeConfig,
    read_fetch_index,
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
        assert pikachu_estimate.values == (112, 107, 60, 63, 70, 156)
        assert pikachu_estimate.provenance == "IMPUTED"

        stat_overrides = {estimate.member_id: estimate.values for estimate in game.stat_estimates}
        builder = ObservationBuilder(default_runtime_resources())
        expected_pikachu = [112 / 300, 107 / 300, 60 / 300, 63 / 300, 70 / 300, 156 / 300]

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

        meta_path = tmp_path / config.format_id / "metadata" / f"{replay_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = FetchMetadata(
            source_url=f"https://replay.pokemonshowdown.com/{replay_id}.json",
            fetched_at="2026-01-01T00:00:00Z",
            http_status=200,
            attempt=1,
            retry_count=0,
            elapsed_ms=42,
        )
        meta_bytes = json.dumps(meta.to_dict()).encode() + b"\n"
        meta_path.write_bytes(meta_bytes)
        initial_mtime = meta_path.stat().st_mtime_ns

        fetcher = ReplayFetcher(config)
        entries = fetcher.acquire((replay_id,))

        assert len(entries) == 1
        assert entries[0].replay_id == replay_id
        assert entries[0].byte_size == len(raw_payload)
        assert read_fetch_index(fetcher.index_path) == tuple(entries)
        assert meta_path.read_bytes() == meta_bytes
        assert meta_path.stat().st_mtime_ns == initial_mtime

    def test_fetch_index_rejects_invalid_json(self, tmp_path: Path) -> None:
        index = tmp_path / "index.jsonl"
        index.write_bytes(b"not-json\n")
        with pytest.raises(ValueError, match="Malformed fetch index"):
            read_fetch_index(index)
