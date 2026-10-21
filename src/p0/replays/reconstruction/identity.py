"""Stable replay identities and unresolved protocol Pokémon references."""

from __future__ import annotations

import re
from typing import NamedTuple

from p0.replays.identity import ReplayMemberId, ReplaySide


class ProtocolPokemonReference(NamedTuple):
    """An unresolved side or active-slot Pokémon reference from a protocol argument."""

    side: ReplaySide
    active_slot: int | None
    displayed_name: str


# The displayed name must contain a visible character.
_ENDPOINT_PATTERN = re.compile(r"^(p[12])([ab])?:\s*(\S.*)$")


def parse_protocol_pokemon_reference(value: str) -> ProtocolPokemonReference:
    """Parse a protocol Pokémon reference without resolving its roster member."""
    match = _ENDPOINT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid protocol Pokémon reference {value!r}")

    side = ReplaySide(match.group(1))
    slot_letter = match.group(2)
    slot = None if slot_letter is None else ord(slot_letter) - ord("a")
    return ProtocolPokemonReference(side, slot, match.group(3).strip())


def looks_like_protocol_pokemon_reference(value: str) -> bool:
    """Return whether a value begins with a protocol side or active-slot prefix."""
    return value.startswith(("p1:", "p2:", "p1a:", "p1b:", "p2a:", "p2b:"))


__all__ = [
    "ProtocolPokemonReference",
    "ReplayMemberId",
    "ReplaySide",
    "looks_like_protocol_pokemon_reference",
    "parse_protocol_pokemon_reference",
]
