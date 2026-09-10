"""Small deterministic replay fixtures used by unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from p0.format_config import FORMAT
from p0.replays.compile import ShardBuildResult, compile_payloads, write_tensor_shards


def golden_series_id(parent: str) -> str:
    """Return the independently calculated series identity for this fixture family."""
    value = "\n".join((FORMAT.bo3_format, parent, "alice", "bob"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def golden_replay_payload(
    replay_id: str,
    *,
    series_id: str = "series-1",
    game_number: int = 1,
    winner: str = "Alice",
    players: tuple[str, str] = ("Alice", "Bob"),
    first_move_target: str | None = "p1a: Pikachu",
) -> dict[str, Any]:
    """Return a replay payload with the pinned protocol shape."""
    p1_team = [
        {"species": "Pikachu", "ability": "Static", "moves": ["Protect", "Tackle"]},
        {"species": "Eevee", "ability": "Run Away", "moves": ["Tackle", "Helping Hand"]},
        {"species": "Raichu", "ability": "Static", "moves": ["Protect", "Thunderbolt"]},
        {"species": "Jolteon", "ability": "Volt Absorb", "moves": ["Protect", "Thunderbolt"]},
        {"species": "Vaporeon", "ability": "Water Absorb", "moves": ["Protect", "Surf"]},
        {"species": "Flareon", "ability": "Flash Fire", "moves": ["Protect", "Flare Blitz"]},
    ]
    p2_team = [
        {"species": "Bulbasaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
        {"species": "Charmander", "ability": "Blaze", "moves": ["Tackle", "Helping Hand"]},
        {"species": "Squirtle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
        {"species": "Ivysaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
        {"species": "Charmeleon", "ability": "Blaze", "moves": ["Protect", "Ember"]},
        {"species": "Wartortle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
    ]
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{json.dumps(p1_team, separators=(',', ':'))}",
        f"|showteam|p2|{json.dumps(p2_team, separators=(',', ':'))}",
        "|",
        "|switch|p1a: Pikachu|Pikachu, L50|100/100",
        "|switch|p1b: Eevee|Eevee, L50|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
        "|switch|p2b: Charmander|Charmander, L50|100/100",
        "|turn|1",
        "|",
        "|move|p1a: Pikachu|Protect"
        + (f"|{first_move_target}" if first_move_target is not None else ""),
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        "|",
        f"|win|{winner}",
    ]
    return {
        "id": replay_id,
        "formatid": FORMAT.bo3_format,
        "p1": players[0],
        "p2": players[1],
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": series_id,
        "game_number": game_number,
        "log": "\n".join(lines),
    }


def sample_replay_payload(
    replay_id: str,
    parent: str = "series-1",
    *,
    winner: str = "Alice",
    game_number: int | None = None,
    players: tuple[str, str] = ("Alice", "Bob"),
) -> dict[str, object]:
    """Return a compact replay payload for grouping and shard tests."""
    ots = {
        "p1": [
            {"species": "Pikachu", "ability": "Static", "moves": ["Protect", "Tackle"]},
            {"species": "Eevee", "ability": "Run Away", "moves": ["Tackle", "Helping Hand"]},
            {"species": "Raichu", "ability": "Static", "moves": ["Protect", "Thunderbolt"]},
            {"species": "Jolteon", "ability": "Volt Absorb", "moves": ["Protect", "Thunderbolt"]},
            {"species": "Vaporeon", "ability": "Water Absorb", "moves": ["Protect", "Surf"]},
            {"species": "Flareon", "ability": "Flash Fire", "moves": ["Protect", "Flare Blitz"]},
        ],
        "p2": [
            {"species": "Bulbasaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
            {"species": "Charmander", "ability": "Blaze", "moves": ["Tackle", "Helping Hand"]},
            {"species": "Squirtle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
            {"species": "Ivysaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
            {"species": "Charmeleon", "ability": "Blaze", "moves": ["Protect", "Ember"]},
            {"species": "Wartortle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
        ],
    }
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
        f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
        "|",
        "|switch|p1a: Pikachu|Pikachu, L50|100/100",
        "|switch|p1b: Eevee|Eevee, L50|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
        "|switch|p2b: Charmander|Charmander, L50|100/100",
        "|turn|1",
        "|",
        "|move|p1a: Pikachu|Protect|p1a: Pikachu",
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        "|",
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


def decision_payload() -> dict[str, Any]:
    """Return a replay payload with complete OTS entries for decision tests."""
    payload = golden_replay_payload("decision-test")
    lines = str(payload["log"]).splitlines()
    teams = {
        "p1": ("Pikachu", "Eevee", "Raichu", "Jolteon", "Vaporeon", "Flareon"),
        "p2": (
            "Bulbasaur",
            "Charmander",
            "Squirtle",
            "Ivysaur",
            "Charmeleon",
            "Wartortle",
        ),
    }
    for index, line in enumerate(lines):
        if not line.startswith("|showteam|"):
            continue
        side = line.split("|", 3)[2]
        entries = [
            {
                "species": species,
                "name": species,
                "ability": "Static" if side == "p1" else "Overgrow",
                "moves": ["Protect", "Tackle"],
                "item": "Leftovers",
            }
            for species in teams[side]
        ]
        lines[index] = f"|showteam|{side}|{json.dumps(entries, separators=(',', ':'))}"
    payload["log"] = "\n".join(lines)
    return payload


def torch_summaries(built: ShardBuildResult) -> list[dict[str, Any]]:
    """Return the series summaries from the first shard file of a built dataset."""
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def build_dataset_from_payloads(
    tmp_path: Path,
    payloads: tuple[dict[str, Any], ...],
    max_decisions_per_shard: int = 1,
) -> ShardBuildResult:
    """Compile payloads and write shards to a temporary directory."""
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=max_decisions_per_shard,
        created_at="2026-01-01T00:00:00Z",
    )


def build_dataset(tmp_path: Path, count: int) -> ShardBuildResult:
    """Build a dataset of golden replays for shard testing."""
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    return build_dataset_from_payloads(tmp_path, payloads)


def write_dataset_replay_dataset(
    tmp_path: Path,
    payloads: tuple[dict[str, Any], ...],
) -> ShardBuildResult:
    """Compile and write shards from payloads with fixed timestamp."""
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def payload_with_ots_natures(replay_id: str) -> dict[str, Any]:
    """A replay payload whose open team sheets declare natures."""
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
