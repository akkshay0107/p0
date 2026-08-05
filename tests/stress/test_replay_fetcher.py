from __future__ import annotations

import gzip
import json
import threading
from pathlib import Path

import pytest

from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ReplayFetchError,
    ReplayUnavailableError,
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


@pytest.mark.stress
def test_replay_fetcher_filters_discovery_and_rejects_unsafe_cache_ids(tmp_path: Path) -> None:
    config = ScrapeConfig(
        format_id="gen9stress",
        cache_dir=tmp_path,
        page_size=3,
        max_pages=3,
        retries=1,
        rate_limit_per_second=0,
    )

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "page=1" in url:
            body = json.dumps(
                [
                    {"id": "gen9stress-2", "format": "gen9stress"},
                    {"id": "other-1", "format": "other"},
                    {"id": "gen9stress-2", "format": "gen9stress"},
                ]
            ).encode()
            return HttpResponse(200, body)
        return HttpResponse(200, b"[]")

    fetcher = ReplayFetcher(config, transport=transport)
    assert fetcher.discover_ids() == ("gen9stress-2",)
    with pytest.raises(ReplayFetchError, match="unsafe path"):
        fetcher._write_immutable("../escape", b"{}")


@pytest.mark.stress
@pytest.mark.parametrize("status, error", ((404, ReplayUnavailableError), (429, ReplayFetchError)))
def test_replay_fetcher_handles_http_error_matrix(tmp_path: Path, status: int, error) -> None:
    config = ScrapeConfig(format_id="gen9stress", cache_dir=tmp_path, retries=1, backoff_seconds=0)

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(status, b"missing")

    fetcher = ReplayFetcher(config, transport=transport)
    if status == 404:
        assert fetcher.acquire(("gen9stress-missing",)) == ()
    else:
        with pytest.raises(error):
            fetcher.acquire(("gen9stress-missing",))


@pytest.mark.stress
def test_replay_fetcher_recovers_corrupt_raw_cache_and_rejects_bad_index(tmp_path: Path) -> None:
    config = ScrapeConfig(format_id="gen9stress", cache_dir=tmp_path, retries=1)

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, json.dumps({"log": [f"|turn|{replay_id}"]}).encode())

    fetcher = ReplayFetcher(config, transport=transport)
    raw = fetcher._raw_path("gen9stress-1")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"not gzip")
    assert fetcher.acquire(("gen9stress-1",))
    fetcher.index_path.write_bytes(b"not-json\n")
    with pytest.raises((ValueError, ReplayFetchError)):
        read_fetch_index(fetcher.index_path)
