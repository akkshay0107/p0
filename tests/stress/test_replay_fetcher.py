from __future__ import annotations

import gzip
import json
import threading
from pathlib import Path

import pytest

from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ScrapeConfig,
    load_raw_replay,
    read_fetch_index,
)
from tests.stress._helpers import stress_count, stress_rng


class _RetryingTransport:
    """Mock HTTP transport that injects transient 503 failures on first attempt for designated IDs."""

    def __init__(self, retry_ids: set[str]) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._first_attempt = set(retry_ids)

    def __call__(self, url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        with self._lock:
            self.calls.append(replay_id)
            # Return HTTP 503 on the first attempt if marked for retry, then pop to allow subsequent success
            if replay_id in self._first_attempt:
                self._first_attempt.remove(replay_id)
                return HttpResponse(503, b"temporary")
        body = json.dumps({"log": [f"|turn|{replay_id}"]}).encode()
        return HttpResponse(200, body)


@pytest.mark.stress
def test_replay_fetcher_retries_caches_and_resumes_many_ids(tmp_path: Path) -> None:
    """Stress test ReplayFetcher across high concurrency, retry resilience, and cache resumption.
    
    Verifies that:
    1. ReplayFetcher concurrently downloads 1000+ replay payloads.
    2. Transient HTTP 503 failures are automatically retried and succeed without data loss.
    3. Acquired entries are sorted canonically by replay ID.
    4. Payloads are written to disk as valid gzipped JSON documents.
    5. A second acquire call over the same ID set resumes from disk cache with zero additional HTTP requests.
    """
    format_id = "gen9stress"
    count = stress_count("P0_STRESS_FETCH_REPLAYS", 1000)
    replay_ids = tuple(f"{format_id}-{index}" for index in range(count))
    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        concurrency=stress_count("P0_STRESS_FETCH_CONCURRENCY", 16),
        retries=3,
        backoff_seconds=0,
        rate_limit_per_second=0,
        limit_games=count,
    )
    rng = stress_rng()
    # Randomly select ~6% of replay IDs to fail on their first attempt
    retry_ids = {
        replay_ids[0],
        *(replay_id for replay_id in replay_ids[1:] if rng.randrange(17) == 0),
    }
    transport = _RetryingTransport(retry_ids)

    fetcher = ReplayFetcher(config, transport=transport)
    entries = fetcher.acquire(replay_ids)
    # Acquired replay entries must be returned in canonical sorted order
    assert [entry.replay_id for entry in entries] == sorted(replay_ids)
    # Verify retried items were invoked exactly twice and unfailed items invoked exactly once
    assert all(
        transport.calls.count(replay_id) == (2 if replay_id in retry_ids else 1)
        for replay_id in replay_ids
    )
    assert len(transport.calls) == count + len(retry_ids)

    # Verify on-disk storage format: compressed gzip containing parsed JSON
    first_index = fetcher.index_path.read_bytes()
    first_raw = load_raw_replay(fetcher._raw_path(f"{format_id}-1"))
    assert gzip.decompress(fetcher._raw_path(f"{format_id}-1").read_bytes()) == first_raw
    assert json.loads(first_raw)["log"] == [f"|turn|{format_id}-1"]

    # Resume test: acquiring the same replays must hit the local disk cache with 0 network calls
    resumed = ReplayFetcher(config, transport=transport).acquire(replay_ids)
    assert resumed == entries
    assert fetcher.index_path.read_bytes() == first_index
    # Total call count must remain identical (count + len(retry_ids)), confirming zero new network requests
    assert len(transport.calls) == count + len(retry_ids)
    assert len(read_fetch_index(fetcher.index_path)) == count
