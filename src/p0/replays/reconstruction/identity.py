"""Stable replay identities and unresolved protocol Pokémon references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from p0.replays.identity import ReplayMemberId, ReplaySide


@dataclass(frozen=True, slots=True)
class ProtocolPokemonReference:
    """An unresolved side or active-slot Pokémon reference from a protocol argument."""

    side: ReplaySide
    active_slot: int | None
    displayed_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.side, ReplaySide):
            raise TypeError("ProtocolPokemonReference.side must be a ReplaySide")
        if self.active_slot is not None and (
            type(self.active_slot) is not int or not 0 <= self.active_slot < 4
        ):
            raise ValueError("ProtocolPokemonReference.active_slot must be in [0, 4) when present")
        if not self.displayed_name:
            raise ValueError("ProtocolPokemonReference.displayed_name must not be empty")


_ENDPOINT_PATTERN = re.compile(r"^(p[12])([a-d])?:\s*(.+)$")


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
    return value.startswith(
        ("p1:", "p2:", "p1a:", "p1b:", "p1c:", "p1d:", "p2a:", "p2b:", "p2c:", "p2d:")
    )


__all__ = [
    "ProtocolPokemonReference",
    "ReplayMemberId",
    "ReplaySide",
    "looks_like_protocol_pokemon_reference",
    "parse_protocol_pokemon_reference",
]
