"""Deterministic replay pipeline fixtures; network coverage lives in tests/integration."""

from __future__ import annotations

import json

import pytest

from p0.replays.compile import compile_payloads
from p0.replays.group import group_replays, individual_games, validated_bo3_series
from p0.replays.identity import linked_replay_ids
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
from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ReplayFetchError,
    ScrapeConfig,
    load_raw_replay,
)


def _payload(
    replay_id: str,
    *,
    parent: str = "series-1",
    winner: str = "Alice",
    game_number: int | None = None,
    players: tuple[str, str] = ("Alice", "Bob"),
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
        "p1": players[0],
        "p2": players[1],
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": parent,
        "game_number": game_number,
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


def test_public_bo3_metadata_and_empty_protocol_commands_are_preserved() -> None:
    payload = _payload("gen9championsvgc2026regmbbo3-100")
    payload.pop("p1")
    payload.pop("p2")
    payload.pop("parent")
    payload.pop("game_number")
    payload["players"] = ["Alice", "Bob"]
    payload["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
    payload["formatid"] = "gen9championsvgc2026regmbbo3"
    payload["log"] = "\n".join(
        [
            "|uhtml|bestof|<h2><strong>Game 1</strong> of "
            '<a href="/game-bestof3-gen9championsvgc2026regmbbo3-99">a best-of-3</a></h2>',
            "|",
            "||Alice is ready for game 2.",
            str(payload["log"]),
            "|uhtml|next|Next: "
            '<a href="/battle-gen9championsvgc2026regmbbo3-101">'
            "<strong>Game 2 of 3</strong></a>",
        ]
    )

    document = parse_replay_payload(
        payload,
        format_id="gen9championsvgc2026regmbbo3",
    )

    assert document.metadata.format_id == "gen9championsvgc2026regmbbo3"
    assert document.metadata.parent_room == ("game-bestof3-gen9championsvgc2026regmbbo3-99")
    assert document.metadata.game_number == 1
    assert [line.raw for line in document.protocol_lines[:3]] == [
        str(payload["log"]).splitlines()[0],
        "|",
        "||Alice is ready for game 2.",
    ]


def test_null_parent_is_an_orphan_instead_of_a_literal_series_id() -> None:
    payload = _payload("orphan")
    payload["parent"] = None

    document = parse_replay_payload(payload)

    assert document.metadata.parent_room == ""
    assert group_replays((document,)).series[0].record.grouping_method.name == (
        "FALLBACK_SAME_PLAYERS"
    )


def test_link_extraction_is_same_format_and_model_agnostic() -> None:
    payload = _payload("gen9championsvgc2026regmbbo3-100")
    payload["log"] = "\n".join(
        [
            '|uhtml|bestof|<a href="/game-bestof3-gen9championsvgc2026regmbbo3-99">series</a>',
            '|uhtml|next|<a href="/battle-gen9championsvgc2026regmbbo3-101">Game 2</a>',
            '|uhtml|other|<a href="/battle-gen9otherformat-5">other</a>',
        ]
    )
    body = json.dumps(payload).encode()

    assert linked_replay_ids(
        body,
        format_id="gen9championsvgc2026regmbbo3",
    ) == ("gen9championsvgc2026regmbbo3-101",)


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


def test_grouping_preserves_authoritative_numbers_and_stable_series_id() -> None:
    first = parse_replay_payload(_payload("g1", game_number=1))
    second = parse_replay_payload(_payload("g2", game_number=2))

    incomplete = group_replays((first,)).series[0]
    complete = group_replays((second, first)).series[0]

    assert incomplete.record.series_id == complete.record.series_id
    assert [membership.game_number for membership in complete.memberships] == [1, 2]
    assert complete.record.is_complete


def test_grouping_quarantines_missing_and_duplicate_game_numbers() -> None:
    second = parse_replay_payload(_payload("g2", game_number=2))
    third = parse_replay_payload(_payload("g3", game_number=3))
    missing = group_replays((third, second)).series[0]

    duplicate_a = parse_replay_payload(_payload("dup-a", game_number=1))
    duplicate_b = parse_replay_payload(_payload("dup-b", game_number=1))
    duplicate = group_replays((duplicate_a, duplicate_b)).series[0]

    assert [membership.game_number for membership in missing.memberships] == [2, 3]
    assert not missing.record.is_complete
    assert "non_contiguous_game_numbers" in {diagnostic.code for diagnostic in missing.diagnostics}
    assert [membership.game_number for membership in duplicate.memberships] == [1, 1]
    assert not duplicate.record.is_complete
    assert "duplicate_game_number" in {diagnostic.code for diagnostic in duplicate.diagnostics}


def test_grouping_quarantines_games_after_a_series_clinch() -> None:
    games = tuple(
        parse_replay_payload(_payload(f"g{number}", game_number=number)) for number in (1, 2, 3)
    )

    group = group_replays(games).series[0]

    assert group.record.score == (2, 0)
    assert not group.record.is_complete
    assert "game_after_series_clinch" in {diagnostic.code for diagnostic in group.diagnostics}
    assert validated_bo3_series(games) == ()


def test_grouping_quarantines_missing_outcomes_and_team_conflicts() -> None:
    unresolved_payload = _payload("unresolved", game_number=1, parent="unresolved-series")
    unresolved_payload["log"] = "\n".join(
        line for line in str(unresolved_payload["log"]).splitlines() if not line.startswith("|win|")
    )
    unresolved = group_replays((parse_replay_payload(unresolved_payload),)).series[0]

    first = parse_replay_payload(_payload("team-1", game_number=1, parent="team-series"))
    changed_payload = _payload("team-2", game_number=2, parent="team-series")
    changed_payload["log"] = str(changed_payload["log"]).replace("Pikachu", "Raichu")
    conflicted = group_replays((first, parse_replay_payload(changed_payload))).series[0]

    assert not unresolved.record.is_complete
    assert "missing_outcome" in {diagnostic.code for diagnostic in unresolved.diagnostics}
    assert not conflicted.record.is_complete
    assert "team_identity_conflict" in {diagnostic.code for diagnostic in conflicted.diagnostics}
    assert all(
        "team_identity_conflict" in membership.diagnostics for membership in conflicted.memberships
    )


def test_side_roles_are_canonical_and_bo1_bo3_views_share_games() -> None:
    first = parse_replay_payload(_payload("g1", game_number=1))
    second = parse_replay_payload(
        _payload(
            "g2",
            game_number=2,
            winner="Alice",
            players=("Bob", "Alice"),
        )
    )
    grouping = group_replays((second, first)).series[0]

    assert grouping.record.game_player_roles == ((0, 1), (1, 0))
    assert grouping.record.score == (2, 0)
    assert tuple(game.metadata.replay_id for game in individual_games((second, first))) == (
        "g1",
        "g2",
    )
    assert validated_bo3_series((first,)) == ()
    inferred_first = parse_replay_payload(_payload("inferred-1"))
    inferred_second = parse_replay_payload(_payload("inferred-2"))
    assert validated_bo3_series((inferred_first, inferred_second)) == ()
    same_side_second = parse_replay_payload(_payload("g2", game_number=2))
    assert tuple(
        game.metadata.replay_id for game in validated_bo3_series((same_side_second, first))[0].games
    ) == ("g1", "g2")


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


def test_fetcher_accepts_display_formats_and_follows_sibling_links(tmp_path) -> None:
    format_id = "gen9championsvgc2026regmbbo3"
    first_id = f"{format_id}-100"
    second_id = f"{format_id}-101"
    first = _payload(first_id, game_number=1)
    first["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
    first["formatid"] = format_id
    first["log"] = (
        f"|uhtml|bestof|<strong>Game 1</strong> of "
        f'<a href="/game-bestof3-{format_id}-99">a best-of-3</a>\n'
        f'|uhtml|next|<a href="/battle-{second_id}">Game 2 of 3</a>\n'
        f"{first['log']}"
    )
    second = _payload(second_id, game_number=2)
    second["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
    second["formatid"] = format_id
    bodies = {
        f"https://replay.invalid/{first_id}.json": json.dumps(first).encode(),
        f"https://replay.invalid/{second_id}.json": json.dumps(second).encode(),
    }

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(
                200,
                json.dumps(
                    [
                        {
                            "id": first_id,
                            "format": "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)",
                        }
                    ]
                ).encode(),
            )
        return HttpResponse(200, bodies[url])

    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    fetcher = ReplayFetcher(config, transport=transport)

    assert fetcher.discover_ids() == (first_id,)
    assert tuple(entry.replay_id for entry in fetcher.acquire((first_id,))) == (
        first_id,
        second_id,
    )


def test_fetcher_rejects_malformed_replay_json_before_caching(tmp_path) -> None:
    config = ScrapeConfig(
        format_id="f",
        cache_dir=tmp_path,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(200, b"not-json")

    with pytest.raises(ReplayFetchError, match="malformed JSON"):
        ReplayFetcher(config, transport=transport).acquire(("f-1",))

    assert not (tmp_path / "f" / "raw" / "f-1.json.gz").exists()
