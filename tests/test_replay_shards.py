import json
from pathlib import Path

import pytest
import torch

from p0.format_config import DEFAULT_RUNTIME_MANIFEST
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.shards import load_shard_manifest


def _payload(replay_id: str) -> dict[str, object]:
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
        "parent": "series-1",
        "log": "\n".join(lines),
    }


def test_replay_fixture_compiles_to_runtime_bound_schema_v3_shard(tmp_path: Path) -> None:
    result = compile_payloads((_payload("shard-fixture"),))
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


def test_shard_bytes_are_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    result = compile_payloads((_payload("shard-fixture"),))
    first = write_tensor_shards(result, tmp_path / "first", created_at="2026-01-01T00:00:00Z")
    second = write_tensor_shards(result, tmp_path / "second", created_at="2026-01-01T00:00:00Z")
    first_path = first.manifest_path.parent / first.manifest.shards[0].filename
    second_path = second.manifest_path.parent / second.manifest.shards[0].filename
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.manifest.to_dict() == second.manifest.to_dict()


def test_shard_manifest_rejects_runtime_contract_mismatch(tmp_path: Path) -> None:
    result = compile_payloads((_payload("shard-fixture"),))
    built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")
    value = json.loads(built.manifest_path.read_text(encoding="utf-8"))
    value["runtime_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime contract"):
        load_shard_manifest(value, DEFAULT_RUNTIME_MANIFEST)
