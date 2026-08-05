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


@pytest.mark.stress
def test_replay_fetcher_retries_caches_and_resumes_many_ids(tmp_path: Path) -> None:
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
    calls: list[str] = []
    lock = threading.Lock()
    rng = stress_rng()
    retry_ids = {replay_id for replay_id in replay_ids if rng.randrange(17) == 0}
    first_attempt = set(retry_ids)

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        with lock:
            calls.append(replay_id)
            if replay_id in first_attempt:
                first_attempt.remove(replay_id)
                return HttpResponse(503, b"temporary")
        body = json.dumps({"log": [f"|turn|{replay_id}"]}).encode()
        return HttpResponse(200, body)

    fetcher = ReplayFetcher(config, transport=transport)
    entries = fetcher.acquire(replay_ids)
    assert [entry.replay_id for entry in entries] == sorted(replay_ids)
    assert all(
        calls.count(replay_id) == (2 if replay_id in retry_ids else 1) for replay_id in replay_ids
    )
    assert len(calls) == count + len(retry_ids)

    first_index = fetcher.index_path.read_bytes()
    first_raw = load_raw_replay(fetcher._raw_path(f"{format_id}-1"))
    assert gzip.decompress(fetcher._raw_path(f"{format_id}-1").read_bytes()) == first_raw

    resumed = ReplayFetcher(config, transport=transport).acquire(replay_ids)
    assert resumed == entries
    assert fetcher.index_path.read_bytes() == first_index
    assert len(calls) == count + len(retry_ids)
    assert len(read_fetch_index(fetcher.index_path)) == count
