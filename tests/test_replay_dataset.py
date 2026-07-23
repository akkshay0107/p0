import json
from pathlib import Path

import pytest
import torch

from p0.format_config import DEFAULT_RUNTIME_MANIFEST, load_active_runtime_manifest
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import (
    LazyReplayDataset,
    SeriesSplitManifest,
    assign_series_splits,
    load_split_manifest,
    write_split_manifest,
)


def _payload(replay_id: str, parent: str = "series-1") -> dict[str, object]:
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


def _write_dataset(tmp_path: Path, payloads: tuple[dict[str, object], ...]):
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
    )
    second = assign_series_splits(
        ("series-a", "series-b"),
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        runtime_contract_sha256=runtime_hash,
    )
    assert first.to_dict() == second.to_dict()
    path = tmp_path / "splits.json"
    write_split_manifest(first, path)
    assert load_split_manifest(path).to_dict() == first.to_dict()


def test_lazy_dataset_yields_complete_game_perspectives_and_causal_history(
    tmp_path: Path,
) -> None:
    built = _write_dataset(tmp_path, (_payload("game-1"), _payload("game-2")))
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert all(chunk.length == 2 for chunk in chunks)
    assert chunks[0].summary_inputs == () and chunks[1].summary_inputs == ()
    assert len(chunks[2].summary_inputs) == 1
    assert len(chunks[3].summary_inputs) == 1
    assert chunks[2].candidate_offsets.tolist() == [0, 0, 1]


def test_downstream_shards_preserve_noncontiguous_source_game_numbers(
    tmp_path: Path,
) -> None:
    second = _payload("game-2")
    second["game_number"] = 2
    third = _payload("game-3")
    third["game_number"] = 3

    built = _write_dataset(tmp_path, (third, second))
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    assert chunks[0].summary_inputs == ()
    assert [summary.game_number for summary in chunks[2].summary_inputs] == [2]


def test_split_dataset_keeps_series_together(tmp_path: Path) -> None:
    built = _write_dataset(
        tmp_path,
        (_payload("game-1", "series-1"), _payload("game-2", "series-2")),
    )
    series_ids = sorted({str(summary["series_id"]) for summary in torch_summaries(built)})
    runtime_hash = built.manifest.runtime_contract_sha256
    split = SeriesSplitManifest(
        runtime_hash,
        0,
        {series_ids[0]: "train", series_ids[1]: "test"},
    )
    split_path = tmp_path / "splits.json"
    write_split_manifest(split, split_path)
    train = list(LazyReplayDataset(built.manifest_path, split="train", split_manifest=split_path))
    test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split_path))
    assert {chunk.series_id for chunk in train} == {series_ids[0]}
    assert {chunk.series_id for chunk in test} == {series_ids[1]}


def test_dataset_rejects_tampered_shard(tmp_path: Path) -> None:
    built = _write_dataset(tmp_path, (_payload("game-1"),))
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        next(iter(LazyReplayDataset(built.manifest_path)))


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]
