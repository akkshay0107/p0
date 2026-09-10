"""Tests for replay statistics and scraping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.battle.legality import DecisionView, SlotDecision
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.paths import DEFAULT_PATHS
from p0.replays.compile import (
    CompilationResult,
    compile_payloads,
    compile_to_shards,
)
from p0.replays.dataset import (
    LazyReplayDataset,
    assign_series_splits,
)
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.protocol import ReplayInputContractError, parse_replay_payload
from p0.replays.schema import LabelKind
from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ReplayFetchError,
    ReplayUnavailableError,
    ScrapeConfig,
    load_raw_replay,
    read_fetch_index,
)
from tests.unit.replay_fixtures import payload_with_ots_natures, sample_replay_payload


def _numerical_rows(result: CompilationResult) -> list[tuple[float, ...]]:
    builder = ObservationBuilder(default_runtime_resources())
    rows: list[tuple[float, ...]] = []
    for game in result.games:
        for perspective in game.perspectives:
            for snapshot in perspective.snapshots:
                observation = builder.build(snapshot.view)
                rows.extend(
                    tuple(float(value) for value in token) for token in observation.numerical
                )
    return rows


class TestReplayScraping:
    def test_replay_stat_imputation_is_consistent_with_runtime_dex(self) -> None:
        """Verify explicit replay stat estimates are stable when the runtime dex is passed explicitly."""
        dex = json.loads(
            (DEFAULT_PATHS.data_root / "champions_dex.json").read_text(encoding="utf-8")
        )
        without = compile_payloads((payload_with_ots_natures("metrics-off"),))
        with_dex = compile_payloads((payload_with_ots_natures("metrics-on"),), dex=dex)

        assert _numerical_rows(without) == _numerical_rows(with_dex)

        assert (
            without.metrics.counters["imputations"] == with_dex.metrics.counters["imputations"] > 0
        )
        assert (
            without.metrics.counters["imputation_unknown"]
            == with_dex.metrics.counters["imputation_unknown"]
        )
        assert (
            without.metrics.counters["imputation_confidence_sum"]
            == with_dex.metrics.counters["imputation_confidence_sum"]
        )
        assert with_dex.metrics.counters["imputation_confidence_sum"] > 0.0

    def test_replay_stats_are_applied_to_both_projected_sides(self) -> None:
        """Verify replay stat estimates cover both sides and projected natures remain explicit."""
        dex = json.loads(
            (DEFAULT_PATHS.data_root / "champions_dex.json").read_text(encoding="utf-8")
        )
        result = compile_payloads((payload_with_ots_natures("own-side"),), dex=dex)
        assert result.games

        # Estimates cover every stable roster member, and both projected sides retain
        # the corresponding open team sheet nature.
        for game in result.games:
            assert {estimate.member_id.side.side_index for estimate in game.stat_estimates} == {
                0,
                1,
            }
            for perspective in game.perspectives:
                snapshot = perspective.snapshots[0]
                assert all(mon.nature for mon in snapshot.view.team.values())

    def test_candidate_cap_degrades_to_explicit_unknown_evidence(self) -> None:
        """Verify that when candidate action space exceeds max_candidates, the label degrades to LabelKind.UNKNOWN with empty candidates."""
        view = DecisionView(
            slots=(
                SlotDecision(move_targets=((-2, -1),)),
                SlotDecision(move_targets=((-2,),)),
            )
        )
        evidence = extract_action_evidence(
            EvidenceRequest(
                view=view,
                slots=(
                    ObservedAction(alternatives=(7, 8), exact=False),
                    ObservedAction(action=7),
                ),
                max_candidates=1,
            )
        )
        assert evidence.label_kind is LabelKind.UNKNOWN
        assert evidence.candidates == ()
        assert "candidate_cap_or_illegal" in evidence.tags

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

    def test_scrape_soft_limit_completes_the_final_linked_series(self, tmp_path: Path) -> None:
        """Verify scrape limit_games acts as a soft limit that finishes downloading the remaining sibling games of the current series."""
        format_id = FORMAT.bo3_format
        seeds = [f"{format_id}-{number}" for number in (100, 200, 300)]
        siblings = [f"{format_id}-{number}" for number in (101, 201, 301)]
        bodies: dict[str, bytes] = {}
        for seed, sibling in zip(seeds, siblings, strict=True):
            first = sample_replay_payload(seed, parent=f"series-{seed}")
            first["log"] = f'|uhtml|next|<a href="/battle-{sibling}">Game 2</a>\n{first["log"]}'
            bodies[seed] = json.dumps(first).encode()
            bodies[sibling] = json.dumps(
                sample_replay_payload(sibling, parent=f"series-{seed}")
            ).encode()

        def transport(url: str, timeout: float) -> HttpResponse:
            del timeout
            if "search.invalid" in url:
                return HttpResponse(
                    200,
                    json.dumps([{"id": seed, "formatid": format_id} for seed in seeds]).encode(),
                )
            replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
            return HttpResponse(200, bodies[replay_id])

        config = ScrapeConfig(
            format_id=format_id,
            cache_dir=tmp_path,
            search_url="https://search.invalid",
            replay_url_template="https://replay.invalid/{replay_id}.json",
            page_size=50,
            limit_games=3,
            rate_limit_per_second=0,
        )
        entries = ReplayFetcher(config, transport=transport).acquire()

        assert {entry.replay_id for entry in entries} == {
            seeds[0],
            siblings[0],
            seeds[1],
            siblings[1],
        }

    def test_raw_cache_keeps_malformed_bytes_for_later_quality_rejection(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify malformed replay payloads are written to disk cache to prevent repeatedly querying broken replay endpoints."""
        replay_id = f"{FORMAT.bo3_format}-malformed"

        def transport(url: str, timeout: float) -> HttpResponse:
            del url, timeout
            return HttpResponse(200, b"not-json")

        config = ScrapeConfig(
            format_id=FORMAT.bo3_format,
            cache_dir=tmp_path,
            replay_url_template="https://replay.invalid/{replay_id}.json",
            rate_limit_per_second=0,
        )
        ReplayFetcher(config, transport=transport).acquire((replay_id,))

        assert (
            load_raw_replay(tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz")
            == b"not-json"
        )

    def test_fetcher_skips_404_without_writing_cache_entry(self, tmp_path: Path, caplog) -> None:
        """Verify ReplayFetcher logs a warning and returns empty results without creating cache artifacts on 404 Not Found."""
        replay_id = f"{FORMAT.bo3_format}-missing"

        def transport(url: str, timeout: float) -> HttpResponse:
            del timeout
            if "search.invalid" in url:
                return HttpResponse(200, json.dumps([{"id": replay_id}]).encode())
            return HttpResponse(404, b"not found")

        config = ScrapeConfig(
            format_id=FORMAT.bo3_format,
            cache_dir=tmp_path,
            search_url="https://search.invalid",
            replay_url_template="https://replay.invalid/{replay_id}.json",
            rate_limit_per_second=0,
        )
        with caplog.at_level("WARNING", logger="p0.replays.scrape"):
            assert ReplayFetcher(config, transport=transport).acquire() == ()
        assert not (tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz").exists()
        assert not (tmp_path / FORMAT.bo3_format / "metadata" / f"{replay_id}.json").exists()
        assert "skipping unavailable replay" in caplog.text

    def test_cache_build_is_dataset_bound_and_preserves_bo3_series(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify compile_to_shards produces identical deterministic dataset hashes and separates accepted vs rejected games."""
        good_id = f"{FORMAT.bo3_format}-good"
        bad_id = f"{FORMAT.bo3_format}-bad"
        good = sample_replay_payload(good_id, parent="source-series")
        bad = sample_replay_payload(bad_id, parent="source-series")
        bad["log"] = "\n".join(
            line for line in str(bad["log"]).splitlines() if "|showteam|" not in line
        )
        bodies = {good_id: json.dumps(good).encode(), bad_id: json.dumps(bad).encode()}

        def transport(url: str, timeout: float) -> HttpResponse:
            del timeout
            replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
            return HttpResponse(200, bodies[replay_id])

        cache = tmp_path / "replays"
        config = ScrapeConfig(
            format_id=FORMAT.bo3_format,
            cache_dir=cache,
            replay_url_template="https://replay.invalid/{replay_id}.json",
            rate_limit_per_second=0,
        )
        ReplayFetcher(config, transport=transport).acquire((good_id, bad_id))

        docs = [
            parse_replay_payload(load_raw_replay(p))
            for p in (cache / FORMAT.bo3_format / "raw").glob("*.json.gz")
        ]
        with pytest.raises(ReplayInputContractError) as error:
            compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)
        assert error.value.category == "INVALID_INPUT_CONTRACT"
        docs = [
            parse_replay_payload(load_raw_replay(path))
            for path in (cache / FORMAT.bo3_format / "raw").glob("*.json.gz")
            if "bad" not in path.name
        ]
        first = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)
        second = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)

        assert first.manifest_path == second.manifest_path
        assert first.manifest.dataset_hash == second.manifest.dataset_hash
        assert first.manifest.source_games == 1
        assert first.manifest.accepted_games == 1
        assert first.manifest.rejected_games == 0
        chunks = list(LazyReplayDataset(first.manifest_path))
        assert len(chunks) == 2

    def test_split_assignment_populates_all_requested_splits_when_possible(self) -> None:
        """Verify assign_series_splits allocates items to train, validation, and test partitions."""
        manifest = assign_series_splits(
            ("a", "b", "c", "d", "e"),
            global_contract_sha256="a" * 64,
            dataset_hash="b" * 64,
        )

        assert set(manifest.assignments.values()) == {"train", "validation", "test"}

    def test_replay_fetcher_filters_discovery_and_rejects_unsafe_cache_ids(
        self, tmp_path: Path
    ) -> None:
        """Verify ReplayFetcher filters duplicate discovery IDs and prevents directory traversal attacks in cache paths."""
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
            fetcher.acquire(("../escape",))

    @pytest.mark.parametrize(
        "status, error", ((404, ReplayUnavailableError), (429, ReplayFetchError))
    )
    def test_replay_fetcher_handles_http_error_matrix(
        self, tmp_path: Path, status: int, error: type[Exception]
    ) -> None:
        """Verify ReplayFetcher distinguishes 404 Not Found from retryable errors (429/500/503)."""
        config = ScrapeConfig(
            format_id="gen9stress", cache_dir=tmp_path, retries=1, backoff_seconds=0
        )

        def transport(url: str, timeout: float) -> HttpResponse:
            del url, timeout
            return HttpResponse(status, b"missing")

        fetcher = ReplayFetcher(config, transport=transport)
        if status == 404:
            assert fetcher.acquire(("gen9stress-missing",)) == ()
        else:
            with pytest.raises(error):
                fetcher.acquire(("gen9stress-missing",))

    def test_replay_fetcher_recovers_corrupt_raw_cache_and_rejects_bad_index(
        self, tmp_path: Path
    ) -> None:
        """Verify ReplayFetcher re-fetches when disk cache is corrupted and read_fetch_index validates JSONL schema."""
        config = ScrapeConfig(format_id="gen9stress", cache_dir=tmp_path, retries=1)

        def transport(url: str, timeout: float) -> HttpResponse:
            del timeout
            replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
            return HttpResponse(200, json.dumps({"log": [f"|turn|{replay_id}"]}).encode())

        fetcher = ReplayFetcher(config, transport=transport)
        raw = tmp_path / config.format_id / "raw" / "gen9stress-1.json.gz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"not gzip")
        assert fetcher.acquire(("gen9stress-1",))
        fetcher.index_path.write_bytes(b"not-json\n")
        with pytest.raises((ValueError, ReplayFetchError)):
            read_fetch_index(fetcher.index_path)
