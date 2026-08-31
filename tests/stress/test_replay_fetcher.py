from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import pytest

from p0.replays.scrape import ReplayFetcher, ScrapeConfig, load_raw_replay, read_fetch_index
from tests.stress._helpers import stress_count, stress_rng


class _ReplayRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast(_ReplayHttpServer, getattr(self.server, "replay_fixture"))
        replay_id = urlsplit(self.path).path.rsplit("/", 1)[-1].removesuffix(".json")
        status, body = server.response(replay_id)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _ReplayHttpServer:
    """Real loopback HTTP service used to exercise fetch retries and concurrency."""

    def __init__(self, retry_ids: set[str]) -> None:
        self._retry_ids = set(retry_ids)
        self._calls: list[str] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _ReplayRequestHandler)
        self._server.replay_fixture = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def calls(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._calls)

    def __enter__(self) -> _ReplayHttpServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def response(self, replay_id: str) -> tuple[int, bytes]:
        with self._lock:
            self._calls.append(replay_id)
            if replay_id in self._retry_ids:
                self._retry_ids.remove(replay_id)
                return 503, b"temporary"
        body = json.dumps({"log": [f"|turn|{replay_id}"]}).encode()
        return 200, body


@pytest.mark.stress
def test_replay_fetcher_retries_caches_and_resumes_many_ids(tmp_path: Path) -> None:
    """Stress ReplayFetcher through a real local HTTP service and resumable disk cache."""
    format_id = "gen9stress"
    count = stress_count("P0_STRESS_FETCH_REPLAYS", 1000)
    replay_ids = tuple(f"{format_id}-{index}" for index in range(count))
    rng = stress_rng()
    retry_ids = {
        replay_ids[0],
        *(replay_id for replay_id in replay_ids[1:] if rng.randrange(17) == 0),
    }

    with _ReplayHttpServer(retry_ids) as server:
        config = ScrapeConfig(
            format_id=format_id,
            cache_dir=tmp_path,
            replay_url_template=f"http://127.0.0.1:{server.port}/{{replay_id}}.json",
            concurrency=stress_count("P0_STRESS_FETCH_CONCURRENCY", 16),
            retries=3,
            backoff_seconds=0,
            rate_limit_per_second=0,
            limit_games=count,
        )
        fetcher = ReplayFetcher(config)
        entries = fetcher.acquire(replay_ids)

        assert [entry.replay_id for entry in entries] == sorted(replay_ids)
        assert all(
            server.calls.count(replay_id) == (2 if replay_id in retry_ids else 1)
            for replay_id in replay_ids
        )
        assert len(server.calls) == count + len(retry_ids)

        first_index = fetcher.index_path.read_bytes()
        raw_path = tmp_path / format_id / "raw" / f"{format_id}-1.json.gz"
        first_raw = load_raw_replay(raw_path)
        assert gzip.decompress(raw_path.read_bytes()) == first_raw
        assert json.loads(first_raw)["log"] == [f"|turn|{format_id}-1"]

        resumed = ReplayFetcher(config).acquire(replay_ids)
        assert resumed == entries
        assert fetcher.index_path.read_bytes() == first_index
        assert len(server.calls) == count + len(retry_ids)
        assert len(read_fetch_index(fetcher.index_path)) == count
