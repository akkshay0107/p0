"""Replay fetching through real loopback HTTP and filesystem storage."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from p0.format_config import FORMAT
from p0.replays.scrape import (
    ReplayFetcher,
    ReplayFetchError,
    ScrapeConfig,
    load_raw_replay,
)
from tests.replay_http_server import ReplayHTTPServer
from tests.unit.replay_fixtures import sample_replay_payload

pytestmark = pytest.mark.integration


class TestReplayHTTP:
    @pytest.mark.parametrize("status", (429, 500, 503))
    def test_retries_then_preserves_raw_cache(self, tmp_path: Path, status: int) -> None:
        replay_id = "f-1"
        body = json.dumps(sample_replay_payload(replay_id)).encode()
        path = f"/{replay_id}.json"
        search_path = "/search.json?format=f&page=1"
        with ReplayHTTPServer(
            {
                search_path: ((status, b"retry"), (200, b'[{"id":"f-1","uploadtime":1750000000}]')),
                path: ((status, b"retry"), (200, body)),
            }
        ) as server:
            config = ScrapeConfig(
                format_id="f",
                cache_dir=tmp_path,
                search_url=f"{server.url}/search.json",
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                retries=2,
                backoff_seconds=0,
                rate_limit_per_second=0,
            )
            entries = ReplayFetcher(config).acquire()

            assert [entry.replay_id for entry in entries] == [replay_id]
            assert server.calls == {search_path: 2, path: 2}
            raw_path = tmp_path / "f/raw/f-1.json.gz"
            assert load_raw_replay(raw_path) == body
            original = raw_path.read_bytes()
            modified_at = raw_path.stat().st_mtime_ns
            assert entries[0].http_status == 200

            assert ReplayFetcher(config).acquire((replay_id,)) == entries
            assert server.calls == {search_path: 2, path: 2}
            assert raw_path.read_bytes() == original
            assert raw_path.stat().st_mtime_ns == modified_at

    def test_naive_cutoff_is_treated_as_utc(self, tmp_path: Path, monkeypatch) -> None:
        body = json.dumps(
            [
                {"id": "test-before", "formatid": "test", "uploadtime": 119},
                {"id": "test-at", "formatid": "test", "uploadtime": 120},
                {"id": "test-after", "formatid": "test", "uploadtime": 121},
            ]
        ).encode()
        try:
            with monkeypatch.context() as environment:
                environment.setenv("TZ", "Etc/GMT+5")
                time.tzset()
                assert time.timezone == 5 * 3600
                with ReplayHTTPServer(
                    {"/search.json?format=test&page=1": ((200, body),)}
                ) as server:
                    for cutoff in (
                        "1970-01-01T00:02:00",
                        "1970-01-01T00:02:00Z",
                        "1970-01-01T01:02:00+01:00",
                        "1969-12-31T19:02:00-05:00",
                    ):
                        config = ScrapeConfig(
                            format_id="test",
                            cache_dir=tmp_path,
                            search_url=f"{server.url}/search.json",
                            cutoff=cutoff,
                            rate_limit_per_second=0,
                        )
                        assert ReplayFetcher(config).discover_ids() == ("test-after", "test-at")
        finally:
            time.tzset()

    def test_soft_limit_completes_the_final_linked_series(self, tmp_path: Path) -> None:
        format_id = FORMAT.bo3_format
        seeds = [f"{format_id}-{number}" for number in (100, 200, 300)]
        siblings = [f"{format_id}-{number}" for number in (101, 201, 301)]
        responses = {}
        for seed, sibling in zip(seeds, siblings, strict=True):
            first = sample_replay_payload(seed, parent=f"series-{seed}")
            first["log"] = f'|uhtml|next|<a href="/battle-{sibling}">Game 2</a>\n{first["log"]}'
            responses[f"/{seed}.json"] = ((200, json.dumps(first).encode()),)
            responses[f"/{sibling}.json"] = (
                (200, json.dumps(sample_replay_payload(sibling, parent=f"series-{seed}")).encode()),
            )
        responses[f"/search.json?format={format_id}&page=1"] = (
            (200, json.dumps([{"id": seed, "formatid": format_id} for seed in seeds]).encode()),
        )
        with ReplayHTTPServer(responses) as server:
            config = ScrapeConfig(
                format_id=format_id,
                cache_dir=tmp_path,
                search_url=f"{server.url}/search.json",
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                page_size=50,
                limit_games=3,
                rate_limit_per_second=0,
            )
            entries = ReplayFetcher(config).acquire()

            assert {entry.replay_id for entry in entries} == {
                seeds[0],
                siblings[0],
                seeds[1],
                siblings[1],
            }
            assert server.calls[f"/{seeds[2]}.json"] == 0
            assert server.calls[f"/{siblings[2]}.json"] == 0

    def test_display_formats_and_sibling_links(self, tmp_path: Path) -> None:
        format_id = FORMAT.bo3_format
        first_id, second_id = f"{format_id}-100", f"{format_id}-101"
        display_format = "[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)"
        first = sample_replay_payload(first_id, game_number=1)
        first["format"] = display_format
        first["log"] = (
            f'|uhtml|bestof|<a href="/game-bestof3-{format_id}-99">a best-of-3</a>\n'
            f'|uhtml|next|<a href="/battle-{second_id}">Game 2 of 3</a>\n{first["log"]}'
        )
        second = sample_replay_payload(second_id, game_number=2)
        second["format"] = display_format
        with ReplayHTTPServer(
            {
                f"/search.json?format={format_id}&page=1": (
                    (200, json.dumps([{"id": first_id, "format": display_format}]).encode()),
                ),
                f"/{first_id}.json": ((200, json.dumps(first).encode()),),
                f"/{second_id}.json": ((200, json.dumps(second).encode()),),
            }
        ) as server:
            config = ScrapeConfig(
                format_id=format_id,
                cache_dir=tmp_path,
                search_url=f"{server.url}/search.json",
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                rate_limit_per_second=0,
            )
            fetcher = ReplayFetcher(config)

            assert fetcher.discover_ids() == (first_id,)
            assert tuple(entry.replay_id for entry in fetcher.acquire((first_id,))) == (
                first_id,
                second_id,
            )

    def test_404_is_not_retried_or_cached(self, tmp_path: Path, caplog) -> None:
        with ReplayHTTPServer(
            {
                "/search.json?format=f&page=1": ((200, b'[{"id":"f-missing"}]'),),
                "/f-missing.json": ((404, b"not found"),),
            }
        ) as server:
            config = ScrapeConfig(
                format_id="f",
                cache_dir=tmp_path,
                search_url=f"{server.url}/search.json",
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                retries=3,
                backoff_seconds=0,
                rate_limit_per_second=0,
            )
            with caplog.at_level("WARNING", logger="p0.replays.scrape"):
                assert ReplayFetcher(config).acquire() == ()

            assert server.calls["/f-missing.json"] == 1
            assert not (tmp_path / "f/raw/f-missing.json.gz").exists()
            assert not (tmp_path / "f/metadata/f-missing.json").exists()
            assert "skipping unavailable replay" in caplog.text

    @pytest.mark.parametrize("status", (429, 500, 503))
    def test_retry_exhaustion_does_not_cache_error_body(self, tmp_path: Path, status: int) -> None:
        with ReplayHTTPServer({"/f-error.json": ((status, b"unavailable"),)}) as server:
            config = ScrapeConfig(
                format_id="f",
                cache_dir=tmp_path,
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                retries=2,
                backoff_seconds=0,
                rate_limit_per_second=0,
            )

            with pytest.raises(ReplayFetchError, match="failed after 2 attempts"):
                ReplayFetcher(config).acquire(("f-error",))
            assert server.calls["/f-error.json"] == 2
            assert not (tmp_path / "f/raw/f-error.json.gz").exists()
            assert not (tmp_path / "f/metadata/f-error.json").exists()

    def test_discovery_filters_duplicate_and_wrong_format_ids(self, tmp_path: Path) -> None:
        body = json.dumps(
            [
                {"id": "gen9stress-2", "format": "gen9stress"},
                {"id": "other-1", "format": "other"},
                {"id": "gen9stress-2", "format": "gen9stress"},
            ]
        ).encode()
        with ReplayHTTPServer(
            {
                "/search.json?format=gen9stress&page=1": ((200, body),),
                "/search.json?format=gen9stress&page=2": ((200, b"[]"),),
            }
        ) as server:
            config = ScrapeConfig(
                format_id="gen9stress",
                cache_dir=tmp_path,
                search_url=f"{server.url}/search.json",
                page_size=3,
                max_pages=3,
                retries=1,
                rate_limit_per_second=0,
            )

            assert ReplayFetcher(config).discover_ids() == ("gen9stress-2",)

    def test_unsafe_id_cannot_write_outside_the_cache(self, tmp_path: Path) -> None:
        with ReplayHTTPServer({"/../escape.json": ((200, b"{}"),)}) as server:
            config = ScrapeConfig(
                format_id="f",
                cache_dir=tmp_path,
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                retries=1,
                rate_limit_per_second=0,
            )

            with pytest.raises(ReplayFetchError, match="unsafe path"):
                ReplayFetcher(config).acquire(("../escape",))
            assert not (tmp_path / "f/escape.json.gz").exists()

    def test_later_queries_fetch_new_games_without_repeating_cached_games(
        self, tmp_path: Path
    ) -> None:
        with ReplayHTTPServer(
            {
                "/f-1.json": (
                    (200, b'{"id":"f-1","p1":"Alice","p2":"Bob","rating":1500,"uploadtime":120}'),
                ),
                "/f-2.json": (
                    (200, b'{"id":"f-2","p1":"Carol","p2":"Dan","rating":1800,"uploadtime":240}'),
                ),
            }
        ) as server:
            config = ScrapeConfig(
                format_id="f",
                cache_dir=tmp_path,
                replay_url_template=f"{server.url}/{{replay_id}}.json",
                limit_games=1,
                rate_limit_per_second=0,
            )
            first = ReplayFetcher(config).acquire(("f-1",))
            second = ReplayFetcher(config).acquire(("f-1", "f-2"))
            assert [row.replay_id for row in first] == ["f-1"]
            assert [row.replay_id for row in second] == ["f-1", "f-2"]
            assert server.calls == {"/f-1.json": 1, "/f-2.json": 1}
            assert second[1].rating == 1800
            assert second[1].players == ("Carol", "Dan")
