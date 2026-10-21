"""Immutable public displays from the same chronological tracker used during play."""

from __future__ import annotations

import logging
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
    dex_indexes,
    move_state,
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
        self._ots = ots
        initial_by_member = {member.member_id: member for member in initial}
        self._initial: dict[str, ReplayPokemonState] = {}
        self._snapshots: dict[Pokemon, tuple[tuple[object, ...], ReplayPokemonState]] = {}
        for sheet in ots:
            role = sheet.side.value
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
                if sheet.side.side_index != perspective:
                    self._initial[mon.base_species] = initial_by_member[member.member_id]

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
        entries = tuple(by_base[base] for base in self._initial)
        views = dict(zip(active, shown, strict=True))
        snapshots = {
            mon: self._snapshot(mon, views.get(mon, mon))
            for mon in dict.fromkeys((*entries, *active))
            if mon is not None
        }
        active_states = tuple(None if mon is None else snapshots[mon] for mon in active)
        self._snapshots = {mon: self._snapshots[mon] for mon in snapshots}
        return PublicSideState(
            tuple(snapshots[mon] for mon in entries), (active_states[0], active_states[1])
        )

    def _snapshot(self, raw: Pokemon, view: PokemonView) -> ReplayPokemonState:
        """
        Copy changed public values, retaining an earlier immutable state otherwise.

        Arguments:
          raw: tracked Pokemon, including separate copies of the same disguise
          view: live values, with the live adapter's Transform handling

        Returns:
          an immutable member state shared while its tracked values stay equal
        """
        form_name, base_name = view.species, view.base_species
        nature, hp, status = view.nature, view.current_hp_fraction, view.status
        item, ability = view.item, view.ability or "unknown"
        types, base_stats, weight = tuple(view.types), tuple(view.base_stats.items()), view.weight
        moves = tuple((move.id, move.current_pp, move.max_pp) for move in view.moves.values())
        boosts, effects = tuple(view.boosts.items()), tuple(view.effects.items())
        transformed, revealed, fainted = raw.transformed, view.revealed, view.fainted
        first_turn, status_counter = view.first_turn, view.status_counter
        protect_counter, preparing = view.protect_counter, view.preparing
        last_move = None if view.last_move is None else view.last_move.id
        values = (
            form_name,
            base_name,
            nature,
            hp,
            status,
            item,
            ability,
            types,
            base_stats,
            weight,
            moves,
            boosts,
            effects,
            transformed,
            revealed,
            fainted,
            first_turn,
            status_counter,
            protect_counter,
            preparing,
            last_move,
        )
        previous = self._snapshots.get(raw)
        if previous is not None and previous[0] == values:
            return previous[1]

        initial = self._initial[raw.base_species]
        species = dex_indexes()[0]
        form = species[normalize_showdown_id(form_name or initial.current_form)].species
        type_names = tuple(value.name.title() for value in types)
        move_states = tuple(_public_move(*move) for move in moves)
        transform = (
            TransformSnapshot(
                form,
                weight,
                tuple((name, value) for name, value in base_stats if name != "hp"),
                ability,
                move_states,
            )
            if transformed
            else None
        )
        state = ReplayPokemonState(
            member_id=initial.member_id,
            nickname=initial.nickname,
            original_species=species[normalize_showdown_id(base_name)].base_species,
            current_form=form,
            nature=nature or "",
            level=initial.level,
            hp_fraction=hp,
            status=None if status is None else status.name.lower(),
            item=item,
            ability=AbilityState(ability),
            current_types=type_names,
            base_stats=base_stats,
            weight=weight,
            moves=move_states,
            boosts=boosts,
            effects=tuple((normalize_showdown_id(effect.name), count) for effect, count in effects),
            transform=transform,
            revealed=revealed,
            selected=initial.selected,
            fainted=fainted,
            active_turns=int(first_turn),
            status_counter=status_counter,
            protect_counter=protect_counter,
            preparing="preparing" if preparing else None,
            last_move=last_move,
        )
        self._snapshots[raw] = (values, state)
        return state


@lru_cache(maxsize=1 << 14)
def _public_move(move_id: str, current_pp: int, max_pp: int) -> MoveState:
    move = move_state(move_id)
    return MoveState(
        move.move_id, move.name, move.move_type, move.category, move.target, current_pp, max_pp
    )
