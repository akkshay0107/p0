from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p0.battle.legality import DecisionView, SlotDecision
from p0.format_config import DEFAULT_RUNTIME_MANIFEST, load_active_runtime_manifest
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import (
    LazyReplayDataset,
    SeriesSplitManifest,
    assign_series_splits,
    load_split_manifest,
    write_split_manifest,
)
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.group import group_replays, individual_games, validated_bo3_series
from p0.replays.identity import linked_replay_ids
from p0.replays.oracle import OracleCase, OracleExpectation, validate_oracle
from p0.replays.protocol import ReplayParseError, parse_replay_payload
from p0.replays.reconstruct import (
    _segments,
    impute_stat_points,
    reconstruct_both,
    reconstruct_perspective,
)
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
    ScrapeConfig,
    load_raw_replay,
)
from p0.replays.shards import load_shard_manifest


def _payload_replay_dataset(replay_id: str, parent: str = "series-1") -> dict[str, object]:
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
        "|win|Alice",
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


def _write_dataset_replay_dataset(tmp_path: Path, payloads: tuple[dict[str, object], ...]):
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def test_split_assignment_is_order_independent_and_round_trips(tmp_path: Path) -> None:
    runtime_hash = load_active_runtime_manifest(DEFAULT_RUNTIME_MANIFEST).runtime_contract_sha256
    first = assign_series_splits(
        ("series-b", "series-a"),
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        runtime_contract_sha256=runtime_hash,
        dataset_hash="a" * 64,
    )
    second = assign_series_splits(
        ("series-a", "series-b"),
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        runtime_contract_sha256=runtime_hash,
        dataset_hash="a" * 64,
    )
    assert first.to_dict() == second.to_dict()
    path = tmp_path / "splits.json"
    write_split_manifest(first, path)
    assert load_split_manifest(path).to_dict() == first.to_dict()


def test_lazy_dataset_yields_canonical_bo3_game_perspectives(
    tmp_path: Path,
) -> None:
    built = _write_dataset_replay_dataset(
        tmp_path, (_payload_replay_dataset("game-1"), _payload_replay_dataset("game-2"))
    )
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert [chunk.canonical_player for chunk in chunks] == [0, 1, 0, 1]
    assert [chunk.is_series_end for chunk in chunks] == [False, False, True, True]
    assert all(chunk.length == 2 for chunk in chunks)
    assert chunks[2].candidate_offsets.tolist() == [0, 0, 1]


def test_canonical_player_identity_survives_replay_side_swap(tmp_path: Path) -> None:
    first = _payload_replay_dataset("game-1")
    first["game_number"] = 1
    second = _payload_replay_dataset("game-2")
    second["game_number"] = 2
    second["p1"] = "Bob"
    second["p2"] = "Alice"

    built = _write_dataset_replay_dataset(tmp_path, (first, second))
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player, chunk.canonical_player) for chunk in chunks] == [
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 1),
        (2, 1, 0),
    ]


def test_downstream_shards_preserve_noncontiguous_source_game_numbers(
    tmp_path: Path,
) -> None:
    second = _payload_replay_dataset("game-2")
    second["game_number"] = 2
    third = _payload_replay_dataset("game-3")
    third["game_number"] = 3

    built = _write_dataset_replay_dataset(tmp_path, (third, second))
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]


def test_split_dataset_keeps_series_together(tmp_path: Path) -> None:
    built = _write_dataset_replay_dataset(
        tmp_path,
        (
            _payload_replay_dataset("game-1", "series-1"),
            _payload_replay_dataset("game-2", "series-2"),
        ),
    )
    series_ids = sorted({str(summary["series_id"]) for summary in torch_summaries(built)})
    runtime_hash = built.manifest.runtime_contract_sha256
    split = SeriesSplitManifest(
        runtime_hash,
        0,
        {series_ids[0]: "train", series_ids[1]: "test"},
        dataset_hash=built.manifest.dataset_hash,
    )
    split_path = tmp_path / "splits.json"
    write_split_manifest(split, split_path)
    train = list(LazyReplayDataset(built.manifest_path, split="train", split_manifest=split_path))
    test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split_path))
    assert {chunk.series_id for chunk in train} == {series_ids[0]}
    assert {chunk.series_id for chunk in test} == {series_ids[1]}


def test_dataset_rejects_tampered_shard(tmp_path: Path) -> None:
    built = _write_dataset_replay_dataset(tmp_path, (_payload_replay_dataset("game-1"),))
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def _payload_replay_pipeline(
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
    document = parse_replay_payload(_payload_replay_pipeline("g1"))
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
        parse_replay_payload({**_payload_replay_pipeline("bad"), "log": "not a protocol line"})


def test_protocol_ignores_chat_and_multiline_chat_responses() -> None:
    payload = _payload_replay_pipeline("chat-response")
    payload["log"] = "\n".join(
        [
            str(payload["log"]),
            "|c|☆Alice|!dt sharp break",
            "'sharp break' has no exact match. Approximate match:",
            "|c|☆Alice|/raw <ul>Sharp Beak</ul>",
            "|turn|2",
        ]
    )

    document = parse_replay_payload(payload)

    assert all(line.parts[1] not in {"c", "chatmsg"} for line in document.protocol_lines)
    assert all("sharp break" not in line.raw for line in document.protocol_lines)


def test_public_bo3_metadata_and_empty_protocol_commands_are_preserved() -> None:
    payload = _payload_replay_pipeline("gen9championsvgc2026regmbbo3-100")
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
    payload = _payload_replay_pipeline("orphan")
    payload["parent"] = None

    document = parse_replay_payload(payload)

    assert document.metadata.parent_room == ""
    assert group_replays((document,)).series[0].record.grouping_method.name == (
        "FALLBACK_SAME_PLAYERS"
    )


def test_link_extraction_is_same_format_and_model_agnostic() -> None:
    payload = _payload_replay_pipeline("gen9championsvgc2026regmbbo3-100")
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
    payload = _payload_replay_pipeline("packed")
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
    first = parse_replay_payload(_payload_replay_pipeline("g1", parent="series-1"))
    second = parse_replay_payload(_payload_replay_pipeline("g2", parent="series-1", winner="Bob"))
    parent_result = group_replays((second, first), format_id=first.metadata.format_id)
    assert len(parent_result.series) == 1
    assert parent_result.series[0].record.game_replay_ids == ("g1", "g2")
    assert parent_result.series[0].record.score == (1, 1)
    fallback_first = parse_replay_payload(_payload_replay_pipeline("fallback-game-1", parent=""))
    fallback_second = parse_replay_payload(_payload_replay_pipeline("fallback-game-2", parent=""))
    fallback = group_replays((fallback_second, fallback_first))
    assert len(fallback.series) == 1
    assert fallback.series[0].record.grouping_method.name == "FALLBACK_SAME_PLAYERS"


def test_grouping_preserves_authoritative_numbers_and_stable_series_id() -> None:
    first = parse_replay_payload(_payload_replay_pipeline("g1", game_number=1))
    second = parse_replay_payload(_payload_replay_pipeline("g2", game_number=2))

    incomplete = group_replays((first,)).series[0]
    complete = group_replays((second, first)).series[0]

    assert incomplete.record.series_id == complete.record.series_id
    assert [membership.game_number for membership in complete.memberships] == [1, 2]
    assert complete.record.is_complete


def test_grouping_quarantines_missing_and_duplicate_game_numbers() -> None:
    second = parse_replay_payload(_payload_replay_pipeline("g2", game_number=2))
    third = parse_replay_payload(_payload_replay_pipeline("g3", game_number=3))
    missing = group_replays((third, second)).series[0]

    duplicate_a = parse_replay_payload(_payload_replay_pipeline("dup-a", game_number=1))
    duplicate_b = parse_replay_payload(_payload_replay_pipeline("dup-b", game_number=1))
    duplicate = group_replays((duplicate_a, duplicate_b)).series[0]

    assert [membership.game_number for membership in missing.memberships] == [2, 3]
    assert not missing.record.is_complete
    assert "non_contiguous_game_numbers" in {diagnostic.code for diagnostic in missing.diagnostics}
    assert [membership.game_number for membership in duplicate.memberships] == [1, 1]
    assert not duplicate.record.is_complete
    assert "duplicate_game_number" in {diagnostic.code for diagnostic in duplicate.diagnostics}


def test_grouping_quarantines_games_after_a_series_clinch() -> None:
    games = tuple(
        parse_replay_payload(_payload_replay_pipeline(f"g{number}", game_number=number))
        for number in (1, 2, 3)
    )

    group = group_replays(games).series[0]

    assert group.record.score == (2, 0)
    assert not group.record.is_complete
    assert "game_after_series_clinch" in {diagnostic.code for diagnostic in group.diagnostics}
    assert validated_bo3_series(games) == ()


def test_grouping_quarantines_missing_outcomes_and_team_conflicts() -> None:
    unresolved_payload = _payload_replay_pipeline(
        "unresolved", game_number=1, parent="unresolved-series"
    )
    unresolved_payload["log"] = "\n".join(
        line for line in str(unresolved_payload["log"]).splitlines() if not line.startswith("|win|")
    )
    unresolved = group_replays((parse_replay_payload(unresolved_payload),)).series[0]

    first = parse_replay_payload(
        _payload_replay_pipeline("team-1", game_number=1, parent="team-series")
    )
    changed_payload = _payload_replay_pipeline("team-2", game_number=2, parent="team-series")
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
    first = parse_replay_payload(_payload_replay_pipeline("g1", game_number=1))
    second = parse_replay_payload(
        _payload_replay_pipeline(
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
    inferred_first = parse_replay_payload(_payload_replay_pipeline("inferred-1"))
    inferred_second = parse_replay_payload(_payload_replay_pipeline("inferred-2"))
    assert validated_bo3_series((inferred_first, inferred_second)) == ()
    same_side_second = parse_replay_payload(_payload_replay_pipeline("g2", game_number=2))
    assert tuple(
        game.metadata.replay_id for game in validated_bo3_series((same_side_second, first))[0].games
    ) == ("g1", "g2")


def test_reconstruction_is_causal_symmetric_and_compilable() -> None:
    first = _payload_replay_pipeline("g1")
    second = _payload_replay_pipeline("g2", winner="Bob")
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


def test_replay_request_chunks_ignore_outcome_and_automatic_switch_lines() -> None:
    payload = _payload_replay_pipeline("request-chunks")
    lines = str(payload["log"]).splitlines()
    turn_index = lines.index("|turn|1")
    lines = lines[:turn_index] + [
        "|turn|1",
        "|move|p1a: Pikachu|Protect|p2a: Bulbasaur",
        "|cant|p1a: Pikachu|flinch",
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|switch|p1b: Pikachu|Pikachu, L50|100/100|[from] Parting Shot",
        "|move|p2a: Bulbasaur|Protect|p1a: Pikachu",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        "|switch|p1a: Eevee|Eevee, L50|100/100",
        "|turn|2",
        "|move|p1a: Eevee|Tackle|p2a: Bulbasaur",
        "|win|Alice",
    ]
    payload["log"] = "\n".join(lines)

    document = parse_replay_payload(payload)
    segments = _segments(document)
    segment_tags = [
        [line.parts[1] for line in document.protocol_lines[start:end] if len(line.parts) > 1]
        for start, end, _ in segments
    ]
    assert segment_tags[-3:] == [
        ["turn", "move", "cant", "move", "switch", "move", "move"],
        ["switch"],
        ["turn", "move", "win"],
    ]

    perspective = reconstruct_perspective(document, perspective=0)
    assert perspective.diagnostics.counters.get("observed_illegal_action", 0) == 0
    assert perspective.decisions[1].evidence.exact_action == (9, 10)
    assert "cant" not in perspective.decisions[1].evidence.tags


def test_reconstruction_recovers_target_from_still_animation() -> None:
    ots = {
        "p1": [
            {"species": "Archaludon", "moves": ["Electro Shot", "Protect"]},
            {"species": "Swampert", "moves": ["Protect"]},
        ],
        "p2": [
            {"species": "Whimsicott", "moves": ["Protect"]},
            {"species": "Grimmsnarl", "moves": ["Protect"]},
        ],
    }
    payload = _payload_replay_pipeline("still-animation-target")
    payload["log"] = "\n".join(
        [
            "|start",
            "|teampreview",
            f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
            f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
            "|switch|p1a: Archaludon|Archaludon, L50|100/100",
            "|switch|p1b: Swampert|Swampert, L50|0 fnt",
            "|faint|p1b: Swampert",
            "|switch|p2a: Whimsicott|Whimsicott, L50|100/100",
            "|switch|p2b: Grimmsnarl|Grimmsnarl, L50|100/100",
            "|turn|1",
            "|move|p1a: Archaludon|Electro Shot||[still]",
            "|-prepare|p1a: Archaludon|Electro Shot",
            "|-boost|p1a: Archaludon|spa|1",
            "|-anim|p1a: Archaludon|Electro Shot|p2b: Grimmsnarl",
            "|-damage|p2b: Grimmsnarl|0 fnt",
            "|faint|p2b: Grimmsnarl",
            "|win|Alice",
        ]
    )

    perspective = reconstruct_perspective(parse_replay_payload(payload), perspective=0)

    evidence = perspective.decisions[1].evidence
    assert evidence.label_kind.name == "EXACT"
    assert evidence.exact_action == (10, 0)  # Electro Shot -> p2b, plus implicit pass.
    assert "move_anim_target" in evidence.tags
    assert perspective.diagnostics.counters.get("move_slot_or_target_unknown", 0) == 0


def test_reconstruction_marks_struggle_as_forced_move() -> None:
    payload = _payload_replay_pipeline("forced-move")
    lines = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|p1|"):
            roster = json.loads(line.split("|", 3)[3])
            roster[0]["moves"] = ["Struggle"]
            line = f"|showteam|p1|{json.dumps(roster, separators=(',', ':'))}"
        if line.startswith("|move|p1a: Pikachu|Protect|"):
            line = line.replace("|Protect|", "|Struggle|", 1)
        lines.append(line)
    payload["log"] = "\n".join(lines)

    perspective = reconstruct_perspective(parse_replay_payload(payload), perspective=0)

    assert perspective.snapshots[1].view.decision.slots[0].forced_move
    assert perspective.decisions[1].evidence.exact_action[0] == 48
    assert perspective.diagnostics.counters.get("observed_illegal_action", 0) == 0


def test_reconstruction_restores_illusion_alias_on_replace() -> None:
    ots = {
        "p1": [
            {"species": "Toxapex", "moves": ["Protect"]},
            {"species": "Zoroark-Hisui", "moves": ["Protect"]},
            {"species": "Incineroar", "moves": ["Protect"]},
            {"species": "Grimmsnarl", "moves": ["Protect"]},
        ],
        "p2": [
            {"species": "Bulbasaur", "moves": ["Protect"]},
            {"species": "Charmander", "moves": ["Protect"]},
        ],
    }
    payload = _payload_replay_pipeline("illusion-replace")
    payload["log"] = "\n".join(
        [
            "|start",
            "|teampreview",
            f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
            f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
            "|switch|p1a: Toxapex|Toxapex, L50|100/100",
            "|switch|p1b: Grimmsnarl|Grimmsnarl, L50|100/100",
            "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
            "|switch|p2b: Charmander|Charmander, L50|100/100",
            "|turn|1",
            "|move|p2a: Bulbasaur|Tackle|p1a: Toxapex",
            "|-damage|p1a: Toxapex|0 fnt",
            "|replace|p1a: Zoroark|Zoroark-Hisui, L50",
            "|-end|p1a: Zoroark|Illusion",
            "|faint|p1a: Zoroark",
            "|-damage|p1b: Grimmsnarl|0 fnt",
            "|faint|p1b: Grimmsnarl",
            "|switch|p1a: Incineroar|Incineroar, L50|100/100",
            "|switch|p1b: Toxapex|Toxapex, L50|100/100",
            "|turn|2",
            "|win|Alice",
        ]
    )

    perspective = reconstruct_perspective(parse_replay_payload(payload), perspective=0)

    assert perspective.diagnostics.counters.get("observed_illegal_action", 0) == 0
    active = perspective.snapshots[-1].view.active_pokemon[1]
    assert active is not None
    assert active.species == "Toxapex"
    toxapex = next(
        (
            pokemon
            for pokemon in perspective.snapshots[-1].view.team.values()
            if pokemon is not None and pokemon.species == "Toxapex"
        ),
        None,
    )
    assert toxapex is not None
    assert not toxapex.fainted


def test_imputation_uses_base_form_stats_for_missing_form_entry() -> None:
    payload = _payload_replay_pipeline("florges-blue")
    lines = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|p1|"):
            roster = json.loads(line.split("|", 3)[3])
            roster[0]["species"] = "Florges-Blue"
            line = f"|showteam|p1|{json.dumps(roster, separators=(',', ':'))}"
        line = line.replace("p1a: Pikachu", "p1a: Florges-Blue")
        line = line.replace("|Pikachu, L50", "|Florges-Blue, L50")
        lines.append(line)
    payload["log"] = "\n".join(lines)
    document = parse_replay_payload(payload)
    dex = {
        "species": [
            {
                "id": "florges",
                "name": "Florges",
                "baseSpecies": "Florges",
                "formeOrder": ["Florges", "Florges-Blue"],
                "baseStats": {
                    "hp": 78,
                    "atk": 65,
                    "def": 68,
                    "spa": 112,
                    "spd": 154,
                    "spe": 75,
                },
            }
        ],
        "moves": [],
    }

    estimate = impute_stat_points(document, dex=dex, seed=0)[0]

    assert estimate.species == "Florges-Blue"
    assert estimate.provenance == "IMPUTED"
    assert estimate.precomputed is not None


def test_reconstruction_does_not_share_active_illusion_alias_state() -> None:
    ots = {
        "p1": [
            {"species": "Blaziken", "moves": ["Protect"]},
            {"species": "Toxapex", "moves": ["Protect"]},
            {"species": "Zoroark-Hisui", "moves": ["Protect"]},
            {"species": "Incineroar", "moves": ["Protect"]},
        ],
        "p2": [
            {"species": "Bulbasaur", "moves": ["Protect"]},
            {"species": "Charmander", "moves": ["Protect"]},
        ],
    }
    payload = _payload_replay_pipeline("illusion-duplicate-active")
    payload["log"] = "\n".join(
        [
            "|start",
            "|teampreview",
            f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
            f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
            "|switch|p1a: Blaziken|Blaziken, L50|100/100",
            "|switch|p1b: Toxapex|Toxapex, L50|100/100",
            "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
            "|switch|p2b: Charmander|Charmander, L50|100/100",
            "|turn|1",
            "|move|p2a: Bulbasaur|Tackle|p1b: Toxapex",
            "|-damage|p1b: Toxapex|50/100",
            "|move|p2b: Charmander|Tackle|p1a: Blaziken",
            "|-damage|p1a: Blaziken|0 fnt",
            "|faint|p1a: Blaziken",
            "|switch|p1a: Toxapex|Toxapex, L50|100/100",
            "|turn|2",
            "|move|p2a: Bulbasaur|Tackle|p1a: Toxapex",
            "|-damage|p1a: Toxapex|0 fnt",
            "|replace|p1a: Zoroark|Zoroark-Hisui, L50",
            "|-end|p1a: Zoroark|Illusion",
            "|faint|p1a: Zoroark",
            "|turn|3",
            "|win|Alice",
        ]
    )

    perspective = reconstruct_perspective(parse_replay_payload(payload), perspective=0)

    active = perspective.snapshots[-1].view.active_pokemon
    assert active[0] is None
    assert active[1] is not None and active[1].species == "Toxapex"
    assert active[1].current_hp_fraction == 0.5
    assert perspective.diagnostics.counters.get("observed_illegal_action", 0) == 0


def test_forced_switch_makes_other_slot_an_exact_pass() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(switch_slots=(1,), active=False, force_switch=True),
            SlotDecision(active=True),
        )
    )
    evidence = extract_action_evidence(
        EvidenceRequest(view, (ObservedAction(2, tag="switch"), None))
    )

    assert evidence.label_kind.name == "EXACT"
    assert evidence.candidates == ((2, 0),)
    assert evidence.tags == ("switch", "implicit_pass")


def test_replay_event_window_is_previous_request_and_is_model_grounded() -> None:
    payload = _payload_replay_pipeline("event-window")
    payload["log"] = "\n".join(
        f"{line}|100/100" if line.startswith("|switch|") else line
        for line in str(payload["log"]).splitlines()
    )
    perspective = reconstruct_perspective(parse_replay_payload(payload), perspective=0)

    assert perspective.snapshots[0].events == ()
    assert perspective.snapshots[1].events
    assert all(event.event_type.name == "SWITCH_IN" for event in perspective.snapshots[1].events)
    assert perspective.snapshots[1].view.events == list(perspective.snapshots[1].events)


def test_reconstruction_resolves_switch_species_not_nicknames() -> None:
    payload = _payload_replay_pipeline("nickname-form")
    rewritten: list[str] = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|p1|"):
            roster = json.loads(line.split("|", 3)[3])
            roster[0]["species"] = "Ninetales-Alola"
            line = f"|showteam|p1|{json.dumps(roster, separators=(',', ':'))}"
        line = line.replace("p1a: Pikachu", "p1a: Snow")
        line = line.replace("p1b: Eevee", "p1b: Mint")
        line = line.replace("|Pikachu, L50", "|Ninetales-Alola, L50")
        rewritten.append(line)
    payload["log"] = "\n".join(rewritten)

    document = parse_replay_payload(payload)
    perspectives = reconstruct_both(document)

    assert all(perspective.decisions for perspective in perspectives)


def test_controlled_oracle_requires_candidate_containment() -> None:
    case = OracleCase(
        "normal-move",
        _payload_replay_pipeline("oracle"),
        (
            OracleExpectation(0, 1, (9, 10)),
            OracleExpectation(1, 1, (9, 10)),
        ),
    )
    result = validate_oracle(case)
    assert result.passed and result.checked == 2


def test_fetcher_retries_and_writes_immutable_raw_cache(tmp_path) -> None:
    payload = _payload_replay_pipeline("g1")
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
    first = _payload_replay_pipeline(first_id, game_number=1)
    first["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
    first["formatid"] = format_id
    first["log"] = (
        f"|uhtml|bestof|<strong>Game 1</strong> of "
        f'<a href="/game-bestof3-{format_id}-99">a best-of-3</a>\n'
        f'|uhtml|next|<a href="/battle-{second_id}">Game 2 of 3</a>\n'
        f"{first['log']}"
    )
    second = _payload_replay_pipeline(second_id, game_number=2)
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


def test_fetcher_preserves_malformed_replay_json_for_compilation_audit(tmp_path) -> None:
    config = ScrapeConfig(
        format_id="f",
        cache_dir=tmp_path,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(200, b"not-json")

    entries = ReplayFetcher(config, transport=transport).acquire(("f-1",))

    assert [entry.replay_id for entry in entries] == ["f-1"]
    assert load_raw_replay(tmp_path / "f" / "raw" / "f-1.json.gz") == b"not-json"


def _payload_replay_shards(replay_id: str) -> dict[str, object]:
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
        "|switch|p1a: Pikachu|Pikachu, L50|100/100",
        "|switch|p1b: Eevee|Eevee, L50|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
        "|switch|p2b: Charmander|Charmander, L50|100/100",
        "|turn|1",
        "|move|p1a: Pikachu|Protect|p2a: Bulbasaur",
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|move|p2a: Bulbasaur|Protect|p1a: Pikachu",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        "|win|Alice",
    ]
    return {
        "id": replay_id,
        "format": "gen9championsvgc2026regmbbo3",
        "p1": "Alice",
        "p2": "Bob",
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": "series-1",
        "log": "\n".join(lines),
    }


def test_replay_fixture_compiles_to_runtime_bound_schema_v4_shard(tmp_path: Path) -> None:
    result = compile_payloads((_payload_replay_shards("shard-fixture"),))
    built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")

    manifest = load_shard_manifest(
        json.loads(built.manifest_path.read_text(encoding="utf-8")), DEFAULT_RUNTIME_MANIFEST
    )
    assert manifest.decisions == 4
    assert manifest.games == 2
    assert manifest.series == 1
    assert manifest.diagnostics["label_unknown"] == 2

    shard_path = built.manifest_path.parent / manifest.shards[0].filename
    payload = torch.load(shard_path, weights_only=True, map_location="cpu")
    tensors = payload["tensors"]
    assert all(
        torch.isfinite(tensor).all() for tensor in tensors.values() if tensor.is_floating_point()
    )
    assert tensors["categorical"].shape[0] == manifest.decisions
    assert tensors["action_mask"].shape == (4, 2, 49)
    assert tensors["candidate_offsets"].tolist() == [0, 0, 1, 1, 2]
    assert tensors["game_offsets"].tolist() == [0, 2, 4]
    assert tensors["series_offsets"].tolist() == [0, 4]
    assert len(payload["series_summaries"]) == manifest.games
    assert [item["canonical_player"] for item in payload["series_summaries"]] == [0, 1]
    assert torch.count_nonzero(tensors["events_cat"]) > 0
    assert torch.count_nonzero(tensors["events_metadata"]) > 0


def test_shard_bytes_are_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    result = compile_payloads((_payload_replay_shards("shard-fixture"),))
    first = write_tensor_shards(result, tmp_path / "first", created_at="2026-01-01T00:00:00Z")
    second = write_tensor_shards(result, tmp_path / "second", created_at="2026-01-01T00:00:00Z")
    first_path = first.manifest_path.parent / first.manifest.shards[0].filename
    second_path = second.manifest_path.parent / second.manifest.shards[0].filename
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.manifest.to_dict() == second.manifest.to_dict()


def test_shard_manifest_rejects_runtime_contract_mismatch(tmp_path: Path) -> None:
    result = compile_payloads((_payload_replay_shards("shard-fixture"),))
    built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")
    value = json.loads(built.manifest_path.read_text(encoding="utf-8"))
    value["runtime_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime contract"):
        load_shard_manifest(value, DEFAULT_RUNTIME_MANIFEST)
