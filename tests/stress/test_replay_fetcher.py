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


@pytest.mark.stress
def test_replay_fetcher_retries_caches_and_resumes_many_ids(tmp_path: Path) -> None:
    format_id = "gen9stress"
    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        concurrency=4,
        retries=3,
        backoff_seconds=0,
        rate_limit_per_second=0,
        limit_games=4,
    )
    calls: list[str] = []
    lock = threading.Lock()
    first_attempt = {f"{format_id}-1": True}

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        with lock:
            calls.append(replay_id)
            if replay_id in first_attempt:
                del first_attempt[replay_id]
                return HttpResponse(503, b"temporary")
        body = json.dumps({"log": [f"|turn|{replay_id}"]}).encode()
        return HttpResponse(200, body)

    fetcher = ReplayFetcher(config, transport=transport)
    entries = fetcher.acquire([f"{format_id}-{index}" for index in range(4)])
    assert [entry.replay_id for entry in entries] == [f"{format_id}-{index}" for index in range(4)]
    assert calls.count(f"{format_id}-1") == 2
    assert len(calls) == 5

    first_index = fetcher.index_path.read_bytes()
    first_raw = load_raw_replay(fetcher._raw_path(f"{format_id}-1"))
    assert gzip.decompress(fetcher._raw_path(f"{format_id}-1").read_bytes()) == first_raw

    resumed = ReplayFetcher(config, transport=transport).acquire(
        [f"{format_id}-{index}" for index in range(4)]
    )
    assert resumed == entries
    assert fetcher.index_path.read_bytes() == first_index
    assert len(read_fetch_index(fetcher.index_path)) == 4
