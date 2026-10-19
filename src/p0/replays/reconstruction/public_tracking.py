"""Immutable public displays from the same chronological tracker used during play."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache

from poke_env.battle import DoubleBattle, Pokemon
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

from p0.battle.views import PokemonView
from p0.replays.identity import normalize_showdown_id
from p0.replays.reconstruction.resolution import ResolvedProtocolEvent
from p0.replays.reconstruction.state import (
    AbilityState,
    MoveState,
    PublicSideState,
    ReplayPokemonState,
    TransformSnapshot,
    _dex_indexes,
    _move_state,
)
from p0.replays.schema import OTSData
from p0.runtime.poke_env_battle_adapter import PokeEnvBattleView
from p0.runtime.poke_env_patches import install

_LOGGER = logging.getLogger(__name__)


class PublicReplayTracker:
    """Use the owner's known identities and chronological public opponent displays."""

    def __init__(
        self,
        replay_id: str,
        ots: tuple[OTSData, OTSData],
        initial: tuple[ReplayPokemonState, ...],
        *,
        perspective: int,
    ) -> None:
        install()
        self._battle = DoubleBattle(replay_id, "public-reconstruction", _LOGGER, gen=9)
        for side in (0, 1):
            username = "public-reconstruction" if side == perspective else "public-opponent"
            self._battle.parse_message(["", "player", f"p{side + 1}", username, "", ""])
        self._view = PokeEnvBattleView(self._battle)
        self._perspective = perspective
        self._opponent = 1 - perspective
        self._ots = ots
        self._initial = initial
        self._bases: list[tuple[str, ...]] = []
        for sheet in ots:
            role = sheet.side.value
            bases = []
            for member in sheet.members:
                key = f"{role}: {member.nickname}"
                mon = self._battle.get_pokemon(key, details=f"{member.species}, L{member.level}")
                team_member = TeambuilderPokemon(
                    nickname=member.nickname,
                    species=member.species,
                    item=member.item,
                    ability=member.ability,
                    moves=list(member.moves),
                    nature=member.nature,
                    level=member.level,
                )
                self._battle.apply_teambuilder_team(role, [team_member], [mon])
                bases.append(mon.base_species)
            self._bases.append(tuple(bases))

    def apply(self, resolved: ResolvedProtocolEvent) -> None:
        event = resolved.event
        parts = event.raw_line.split("|")
        for reference in resolved.pokemon_refs:
            member_id = reference.member_id
            if member_id is None or member_id.side.side_index != self._perspective:
                continue
            member = self._ots[self._perspective].members[member_id.roster_index]
            index = reference.argument_index + 2
            prefix = parts[index].split(":", 1)[0]
            parts[index] = f"{prefix}: {member.nickname}"
            if (
                reference.argument_index == 0
                and event.tag in {"switch", "drag"}
                and normalize_showdown_id(member.ability) == "illusion"
            ):
                parts[3] = f"{member.species}, L{member.level}"
        self._battle.parse_message(parts)

    def snapshot(self) -> PublicSideState:
        team = self._battle.opponent_team
        active = self._battle.opponent_active_pokemon
        shown = self._view.opponent_active_pokemon
        by_base = {mon.base_species: mon for mon in team.values()}
        entries = tuple(by_base[base] for base in self._bases[self._opponent])
        snapshots: dict[Pokemon, ReplayPokemonState] = {}
        for index, mon in enumerate(entries):
            view = next((view for raw, view in zip(active, shown, strict=True) if raw is mon), mon)
            snapshots[mon] = _snapshot(mon, view, self._initial[self._opponent * 6 + index])
        active_states = []
        for mon, view in zip(active, shown, strict=True):
            if mon is None:
                active_states.append(None)
                continue
            if mon not in snapshots:
                index = next(
                    index
                    for index, entry in enumerate(entries)
                    if entry.base_species == mon.base_species
                )
                snapshots[mon] = _snapshot(mon, view, self._initial[self._opponent * 6 + index])
            active_states.append(snapshots[mon])
        return PublicSideState(
            tuple(snapshots[mon] for mon in entries), (active_states[0], active_states[1])
        )


def _snapshot(raw: Pokemon, view: PokemonView, initial: ReplayPokemonState) -> ReplayPokemonState:
    moves = tuple(
        _public_move(move.id, move.current_pp, move.max_pp) for move in view.moves.values()
    )
    types = tuple(value.name.title() for value in view.types)
    base_stats = tuple(view.base_stats.items())
    ability = view.ability or "unknown"
    species = _dex_indexes()[0]
    form = species[normalize_showdown_id(view.species or initial.current_form)].species
    transform = (
        TransformSnapshot(
            initial.member_id,
            form,
            types,
            view.weight,
            tuple((name, value) for name, value in base_stats if name != "hp"),
            ability,
            tuple(view.boosts.items()),
            moves,
        )
        if raw.transformed
        else None
    )
    return replace(
        initial,
        original_species=species[normalize_showdown_id(view.base_species)].base_species,
        current_form=form,
        nature=view.nature or "",
        hp_fraction=view.current_hp_fraction,
        status=None if view.status is None else view.status.name.lower(),
        item=view.item,
        ability=AbilityState(ability),
        current_types=types,
        base_stats=base_stats,
        weight=view.weight,
        moves=moves,
        boosts=tuple(view.boosts.items()),
        effects=tuple(
            (normalize_showdown_id(effect.name), count) for effect, count in view.effects.items()
        ),
        transform=transform,
        revealed=view.revealed,
        fainted=view.fainted,
        active_turns=int(view.first_turn),
        status_counter=view.status_counter,
        protect_counter=view.protect_counter,
        preparing="preparing" if view.preparing else None,
        last_move=None if view.last_move is None else view.last_move.id,
    )


@lru_cache(maxsize=1 << 14)
def _public_move(move_id: str, current_pp: int, max_pp: int) -> MoveState:
    return replace(_move_state(move_id), current_pp=current_pp, max_pp=max_pp)
