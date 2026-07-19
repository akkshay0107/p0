"""Deterministic replay pipeline fixtures; network coverage lives in tests/integration."""

from __future__ import annotations

import json

import pytest

from p0.replays.compile import compile_payloads
from p0.replays.group import group_replays
from p0.replays.oracle import OracleCase, OracleExpectation, validate_oracle
from p0.replays.protocol import ReplayParseError, parse_replay_payload
from p0.replays.reconstruct import impute_stat_points
from p0.replays.schema import (
    FetchMetadata,
    GameEndReason,
    OTSData,
    ProtocolLine,
    ReplayMetadata,
    ReplayOutcome,
)
from p0.replays.scrape import HttpResponse, ReplayFetcher, ScrapeConfig, load_raw_replay


def _payload(
    replay_id: str, *, parent: str = "series-1", winner: str = "Alice"
) -> dict[str, object]:
    ots = {
        "p1": [
            {"species": "Pikachu", "moves": ["Protect", "Tackle"]},
            {"species": "Eevee", "moves": ["Tackle", "Helping Hand"]},
        ],
        "p2": [
            {"species": "Bulbasaur", "moves": ["Protect", "Tackle"]},
            {"species": "Charmander", "moves": ["Tackle", "Helping Hand"]},
        ],
    }
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
        f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
        "|switch|p1a: Pikachu|Pikachu, L50",
        "|switch|p1b: Eevee|Eevee, L50",
        "|switch|p2a: Bulbasaur|Bulbasaur, L50",
        "|switch|p2b: Charmander|Charmander, L50",
        "|turn|1",
        "|move|p1a: Pikachu|Protect|p2a: Bulbasaur",
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|move|p2a: Bulbasaur|Protect|p1a: Pikachu",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        f"|win|{winner}",
    ]
    return {
        "id": replay_id,
        "format": "gen9championsvgc2026regmbbo3",
        "p1": "Alice",
        "p2": "Bob",
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": parent,
        "log": "\n".join(lines),
    }


def test_protocol_records_are_strict_and_ordered() -> None:
    document = parse_replay_payload(_payload("g1"))
    assert [line.index for line in document.protocol_lines] == list(
        range(len(document.protocol_lines))
    )
    assert document.ots[0].revealed_species == ("Pikachu", "Eevee")
    assert document.ots[0].revealed_details["Pikachu"]["moves"] == ["Protect", "Tackle"]
    assert document.outcome.winner == 0
    assert ReplayMetadata.from_dict(document.metadata.to_dict()) == document.metadata
    assert (
        ProtocolLine.from_dict(document.protocol_lines[2].to_dict()) == document.protocol_lines[2]
    )
    with pytest.raises(ValueError, match="unknown"):
        ProtocolLine.from_dict({**document.protocol_lines[0].to_dict(), "unknown": 1})
    with pytest.raises(ReplayParseError, match="Malformed protocol line"):
        parse_replay_payload({**_payload("bad"), "log": "not a protocol line"})


def test_packed_open_team_sheet_and_seeded_imputation() -> None:
    payload = _payload("packed")
    payload["log"] = "\n".join(
        [
            "|start",
            "|teampreview",
            "|showteam|p1|Pikachu|||static|protect,tackle|Jolly|||||50",
            "|showteam|p2|Bulbasaur|||overgrow|protect,tackle|Bold|||||50",
            "|switch|p1a: Pikachu|Pikachu, L50",
            "|switch|p1b: Pikachu|Pikachu, L50",
            "|switch|p2a: Bulbasaur|Bulbasaur, L50",
            "|switch|p2b: Bulbasaur|Bulbasaur, L50",
            "|turn|1",
            "|move|p1a: Pikachu|Protect|p2a: Bulbasaur",
        ]
    )
    document = parse_replay_payload(payload)
    assert document.ots[0].revealed_details["Pikachu"]["moves"] == ("protect", "tackle")
    dex = {
        "species": [
            {
                "id": "pikachu",
                "baseStats": {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90},
            },
            {
                "id": "bulbasaur",
                "baseStats": {"hp": 45, "atk": 49, "def": 49, "spa": 65, "spd": 65, "spe": 45},
            },
        ],
        "moves": [
            {"id": "protect", "category": "Status"},
            {"id": "tackle", "category": "Physical"},
        ],
    }
    first = impute_stat_points(document, dex=dex, seed=7)
    second = impute_stat_points(document, dex=dex, seed=7)
    assert first == second and all(item.provenance == "IMPUTED" for item in first)


def test_new_schema_records_round_trip() -> None:
    fetch = FetchMetadata(
        source_url="https://example.invalid/replay.json",
        fetched_at="2026-07-19T00:00:00Z",
        http_status=200,
        attempt=2,
        retry_count=1,
        elapsed_ms=12,
    )
    assert FetchMetadata.from_dict(fetch.to_dict()) == fetch
    ots = OTSData("p1", "", ("pikachu",), {})
    assert OTSData.from_dict(ots.to_dict()) == ots
    outcome = ReplayOutcome(0, GameEndReason.NORMAL, 1, 4)
    assert ReplayOutcome.from_dict(outcome.to_dict()) == outcome


def test_grouping_parent_and_fallback_are_deterministic() -> None:
    first = parse_replay_payload(_payload("g1", parent="series-1"))
    second = parse_replay_payload(_payload("g2", parent="series-1", winner="Bob"))
    parent_result = group_replays((second, first), format_id=first.metadata.format_id)
    assert len(parent_result.series) == 1
    assert parent_result.series[0].record.game_replay_ids == ("g1", "g2")
    assert parent_result.series[0].record.score == (1, 1)
    fallback_first = parse_replay_payload(_payload("fallback-game-1", parent=""))
    fallback_second = parse_replay_payload(_payload("fallback-game-2", parent=""))
    fallback = group_replays((fallback_second, fallback_first))
    assert len(fallback.series) == 1
    assert fallback.series[0].record.grouping_method.name == "FALLBACK_SAME_PLAYERS"


def test_reconstruction_is_causal_symmetric_and_compilable() -> None:
    first = _payload("g1")
    second = _payload("g2", winner="Bob")
    result = compile_payloads((first, second))
    assert result.to_dict() == compile_payloads((second, first)).to_dict()
    game = result.games[0]
    left, right = game.perspectives
    assert left.player == 0 and right.player == 1
    assert left.decisions[0].evidence.label_kind.name == "UNKNOWN"
    assert left.decisions[1].evidence.exact_action == (9, 10)
    assert right.decisions[1].evidence.exact_action == (9, 10)
    assert left.snapshots[1].pre_line_index < left.snapshots[1].post_line_index
    assert result.metrics.counters["illegal_candidates"] == 0


def test_controlled_oracle_requires_candidate_containment() -> None:
    case = OracleCase(
        "normal-move",
        _payload("oracle"),
        (
            OracleExpectation(0, 1, (9, 10)),
            OracleExpectation(1, 1, (9, 10)),
        ),
    )
    result = validate_oracle(case)
    assert result.passed and result.checked == 2


def test_fetcher_retries_and_writes_immutable_raw_cache(tmp_path) -> None:
    payload = _payload("g1")
    body = json.dumps(payload).encode()
    calls: list[str] = []
    failures = {"https://search.invalid?format=f&page=1": 1}

    def transport(url: str, timeout: float) -> HttpResponse:
        calls.append(url)
        if failures.get(url, 0):
            failures[url] -= 1
            return HttpResponse(503, b"retry")
        if "search.invalid" in url:
            return HttpResponse(
                200, json.dumps([{"id": "g1", "uploadtime": 1_750_000_000}]).encode()
            )
        return HttpResponse(200, body)

    config = ScrapeConfig(
        format_id="f",
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        retries=2,
        backoff_seconds=0,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config, transport=transport).acquire()
    assert entries[0].replay_id == "g1"
    assert calls.count("https://search.invalid?format=f&page=1") == 2
    raw_path = tmp_path / "f" / "raw" / "g1.json.gz"
    assert load_raw_replay(raw_path) == body
    assert ReplayFetcher(config, transport=transport).acquire() == entries
