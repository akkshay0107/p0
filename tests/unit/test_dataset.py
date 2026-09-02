"""Tests for replay dataset loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p0.format_config import (
    load_runtime_manifest,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.replays.compile import (
    CompilationResult,
    ShardBuildResult,
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.dataset import (
    LazyReplayDataset,
)
from p0.replays.schema import (
    ActionEvidence,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    SeriesRecord,
)
from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ReplayFetchError,
    ReplayUnavailableError,
    ScrapeConfig,
    read_fetch_index,
)
from p0.replays.shards import (
    ShardIndexEntry,
    ShardManifest,
)
from tests.unit.replay_fixtures import golden_replay_payload, sample_replay_payload


def _write_dataset_replay_dataset(
    tmp_path: Path, payloads: tuple[dict[str, object], ...]
) -> ShardBuildResult:
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def _build_dataset_from_payloads(tmp_path, payloads):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path, count: int):
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    return _build_dataset_from_payloads(tmp_path, payloads)


def _payload_with_ots_natures(replay_id: str) -> dict[str, object]:
    """A pipeline payload whose open team sheets declare natures, as real replays do."""
    natures = {"Pikachu": "Jolly", "Eevee": "Adamant", "Bulbasaur": "Bold", "Charmander": "Timid"}
    payload = sample_replay_payload(replay_id)
    lines = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|"):
            head, _, body = line.rpartition("|")
            roster = json.loads(body)
            for mon in roster:
                mon["nature"] = natures.get(mon["species"], "Serious")
            line = f"{head}|{json.dumps(roster, separators=(',', ':'))}"
        lines.append(line)
    payload["log"] = "\n".join(lines)
    return payload


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


def _evidence(kind: LabelKind) -> ActionEvidence:
    candidates = {
        LabelKind.EXACT: ((7, 1),),
        LabelKind.PARTIAL: ((7, 1), (8, 1)),
        LabelKind.UNKNOWN: (),
    }[kind]
    return ActionEvidence(
        label_kind=kind,
        candidates=candidates,
        confidence=0.5 if kind is not LabelKind.UNKNOWN else 0.0,
        mask_provenance=MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tags=("fixture",),
    )


def _series_record() -> SeriesRecord:
    return SeriesRecord(
        series_id="s1",
        format_id="gen9championsvgc2026regmbbo3",
        players=("alice", "bob"),
        game_replay_ids=("r1", "r2"),
        game_player_roles=((0, 1), (1, 0)),
        team_hashes=("a" * 64, "b" * 64),
        is_complete=True,
        score=(2, 0),
        grouping_method=GroupingMethod.PARENT_ROOM,
        grouping_confidence=1.0,
    )


def _shard_manifest_fixture_unit() -> ShardManifest:
    active_contract = load_runtime_manifest().global_sha256
    entry = ShardIndexEntry(
        filename="shard-000.pt", sha256="c" * 64, decisions=10, games=2, series=1, byte_size=1024
    )
    return ShardManifest(
        global_contract_sha256=active_contract,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"max_candidates": 256},
        raw_replays={"game-1": "f" * 64},
        source_series={"series-1": ("game-1",)},
        source_games=1,
        accepted_games=1,
        rejected_games=0,
        artifact_hashes={
            "shard-000.pt": "c" * 64,
        },
    )


class TestReplayDatasets:
    def test_dataset_rejects_tampered_golden_shard(self, tmp_path) -> None:
        """Verify LazyReplayDataset raises ValueError on tampered golden replay shards when verify_hashes=True."""
        built = _build_dataset(tmp_path, 1)
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))

    def test_dataset_rejects_missing_golden_shard(self, tmp_path) -> None:
        """Verify LazyReplayDataset detects missing physical shard files on disk and raises ValueError."""
        built = _build_dataset(tmp_path, 1)
        shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
        shard_path.unlink()
        with pytest.raises(ValueError, match="Shard file is missing"):
            next(iter(LazyReplayDataset(built.manifest_path)))

    def test_dataset_rejects_missing_and_duplicate_source_records(self, tmp_path) -> None:
        """Verify ShardManifest validation checks consistency between raw_replays and source_series records."""
        missing = _build_dataset_from_payloads(
            tmp_path / "missing", (golden_replay_payload("only", series_id="series"),)
        )
        missing_manifest = missing.manifest.to_dict()
        missing_manifest["raw_replays"] = {}
        altered = missing.manifest_path.parent / "missing.json"
        altered.write_text(json.dumps(missing_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="raw_replays|source_games|partition"):
            LazyReplayDataset(altered)

        duplicate_payloads = (
            golden_replay_payload("duplicate", series_id="duplicate-series"),
            golden_replay_payload("duplicate", series_id="duplicate-series"),
        )
        with pytest.raises((ValueError, KeyError), match="duplicate|already|unique|invalid"):
            _build_dataset_from_payloads(tmp_path / "duplicate", duplicate_payloads)

    def test_compiler_retains_exact_partial_unknown_and_rejected_labels(self) -> None:
        """Verify compiler tracks counters for exact, partial, unknown, and rejected labels based on evidence and OTS validity."""
        exact = golden_replay_payload("exact", series_id="label-series")
        partial = golden_replay_payload(
            "partial", series_id="label-series-2", first_move_target=None
        )
        partial["log"] = str(partial["log"]).replace(
            "|move|p1a: Pikachu|Protect\n", "|move|p1a: Pikachu|Tackle\n"
        )
        result = compile_payloads((exact, partial), format_id=exact["formatid"])
        counters = result.metrics.counters
        assert counters["accepted_games"] == 2
        assert counters["label_exact"] == 0
        assert counters["label_partial"] == 8
        assert counters["label_unknown"] == 0

        capped = compile_payloads((partial,), format_id=partial["formatid"], max_candidates=1)
        assert capped.metrics.counters["label_partial"] == 0
        assert capped.metrics.counters["label_exact"] == 0
        assert capped.metrics.counters["label_unknown"] == 4

        rejected = golden_replay_payload("rejected", series_id="rejected-series")
        rejected["log"] = "\n".join(
            line for line in str(rejected["log"]).splitlines() if "|showteam|" not in line
        )
        rejected_result = compile_payloads((rejected,), format_id=rejected["formatid"])
        assert not rejected_result.games
        assert rejected_result.metrics.counters["rejected_games"] == 1

    def test_compiler_is_stable_across_worker_chunk_sizes(self) -> None:
        """Verify public compilation produces the same replay result for serial and parallel chunks."""
        payloads = (
            golden_replay_payload("chunk-a", series_id="chunk-series-a"),
            golden_replay_payload("chunk-b", series_id="chunk-series-b"),
        )

        serial = compile_payloads(payloads, chunksize=0)
        parallel = compile_payloads(payloads, chunksize=1)

        assert serial.to_dict() == parallel.to_dict()

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
        self, tmp_path: Path, status: int, error
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

    def test_reconstructed_views_carry_open_team_sheet_nature(self) -> None:
        """Verify open team sheet natures are preserved for both projected sides."""
        payload = sample_replay_payload("ots-nature")
        natures = {
            "Pikachu": "Jolly",
            "Eevee": "Adamant",
            "Bulbasaur": "Bold",
            "Charmander": "Timid",
        }

        lines = []
        for line in str(payload["log"]).splitlines():
            if line.startswith("|showteam|"):
                head, _, body = line.rpartition("|")
                roster = json.loads(body)
                for mon in roster:
                    mon["nature"] = natures.get(mon["species"], "Serious")
                line = f"{head}|{json.dumps(roster, separators=(',', ':'))}"
            lines.append(line)
        payload["log"] = "\n".join(lines)

        result = compile_payloads((payload,))
        assert result.games

        own: set[str | None] = set()
        opponent: set[str | None] = set()
        for game in result.games:
            for perspective in game.perspectives:
                for snapshot in perspective.snapshots:
                    own.update(mon.nature for mon in snapshot.view.team.values())
                    opponent.update(mon.nature for mon in snapshot.view.opponent_team.values())

        # Every projected roster member comes from the same open team sheet facts.
        assert opponent and None not in opponent
        assert opponent <= {*natures.values(), "Serious"}
        assert own and None not in own
        assert own <= {*natures.values(), "Serious"}
