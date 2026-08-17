"""Small deterministic replay fixtures used by unit tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from p0.format_config import FORMAT


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
        {"species": "Pikachu", "moves": ["Protect", "Tackle"]},
        {"species": "Eevee", "moves": ["Tackle", "Helping Hand"]},
    ]
    p2_team = [
        {"species": "Bulbasaur", "moves": ["Protect", "Tackle"]},
        {"species": "Charmander", "moves": ["Tackle", "Helping Hand"]},
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
