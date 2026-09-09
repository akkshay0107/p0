"""Fast player-relative facade over poke-env battle state."""

from __future__ import annotations

from typing import Any, cast
from weakref import ReferenceType, WeakKeyDictionary, ref

from poke_env.battle import DoubleBattle

from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import TransformedPokemonView
from p0.runtime.live_event_capture import captured_protocol_lines, last_move


class PokeEnvBattleView:
    """Cached facade with explicit properties and no copied per-decision graph."""

    __slots__ = ("_battle_ref", "_decision", "stat_cache")

    def __init__(self, battle: DoubleBattle):
        self._battle_ref: ReferenceType[DoubleBattle] = ref(battle)
        self._decision: DecisionView | None = None
        self.stat_cache: dict[object, tuple[int, int, int, int, int, int]] = {}

    @property
    def _battle(self) -> DoubleBattle:
        battle = self._battle_ref()
        if battle is None:
            raise ReferenceError("The underlying battle has been garbage-collected")
        return battle

    def refresh(self) -> PokeEnvBattleView:
        self._decision = None
        return self

    @property
    def team(self):
        return self._battle.team

    @property
    def opponent_team(self):
        return self._battle.opponent_team

    @property
    def active_pokemon(self):
        return _transformed_active_pokemon(self._battle, include_available_moves=True)

    @property
    def opponent_active_pokemon(self):
        return _transformed_active_pokemon(
            self._battle, include_available_moves=False, opponent=True
        )

    @property
    def available_moves(self):
        return self._battle.available_moves

    @property
    def available_switches(self):
        return self._battle.available_switches

    @property
    def can_mega_evolve(self):
        return self._battle.can_mega_evolve

    @property
    def force_switch(self):
        return self._battle.force_switch

    @property
    def trapped(self):
        return self._battle.trapped

    @property
    def maybe_trapped(self):
        return self._battle.maybe_trapped

    @property
    def teampreview(self):
        return self._battle.teampreview

    @property
    def player_role(self):
        return self._battle.player_role

    @property
    def wait(self):
        # poke-env exposes wait as _wait (asserted integer reason code). This is a
        # version-pinned access point: poke-env is locked to 0.15.0 in pyproject.toml.
        return self._battle._wait

    @property
    def protocol_lines(self) -> tuple[str, ...]:
        """Return raw battle protocol lines when runtime capture is enabled."""
        return captured_protocol_lines(self._battle)

    def replay_events(self) -> tuple[str, ...]:
        """Return poke-env's public replay serialization for this battle."""
        return tuple(self._battle._build_replay_events())

    @property
    def weather(self):
        return self._battle.weather

    @property
    def fields(self):
        return self._battle.fields

    @property
    def side_conditions(self):
        return self._battle.side_conditions

    @property
    def opponent_side_conditions(self):
        return self._battle.opponent_side_conditions

    @property
    def turn(self):
        return self._battle.turn

    @property
    def used_mega_evolve(self):
        return self._battle.used_mega_evolve

    @property
    def opponent_used_mega_evolve(self):
        return self._battle.opponent_used_mega_evolve

    @property
    def decision(self) -> DecisionView:
        if self._decision is None:
            self._decision = decision_view(self._battle)
        return self._decision

    @property
    def spatial_turn(self):
        try:
            return self._battle._p0_spatial_turn  # type: ignore[attr-defined]
        except AttributeError:
            from p0.battle.events import SpatialSlotRecord

            return tuple(SpatialSlotRecord() for _ in range(4))

    def get_pokemon(self, identifier: str):
        return self._battle.get_pokemon(identifier)

    def last_move(self, pokemon):
        return last_move(pokemon)


_VIEWS: WeakKeyDictionary[DoubleBattle, PokeEnvBattleView] = WeakKeyDictionary()


def battle_view(battle: DoubleBattle) -> PokeEnvBattleView:
    """Return refreshed PokeEnvBattleView for the specified battle instance."""
    view = current_battle_view(battle)
    return view.refresh()


def current_battle_view(battle: DoubleBattle) -> PokeEnvBattleView:
    """Return the decision's existing view, creating it only when necessary."""
    view = _VIEWS.get(battle)
    if view is None:
        view = PokeEnvBattleView(battle)
        _VIEWS[battle] = view
    return view


def _transformed_active_pokemon(
    battle: DoubleBattle,
    *,
    include_available_moves: bool,
    opponent: bool = False,
) -> list[Any]:
    active = battle.opponent_active_pokemon if opponent else battle.active_pokemon
    active_list = cast(list[Any], list(active))
    targets = getattr(battle, "_p0_transform_targets", {})
    available = getattr(battle, "available_moves", ()) if include_available_moves else ()
    for index, pokemon in enumerate(active_list):
        if pokemon is None or id(pokemon) not in targets:
            continue
        target = targets[id(pokemon)]
        if include_available_moves:
            moves = dict(target.moves)
            if index < len(available) and available[index]:
                moves.update({move.id: move for move in available[index]})
            active_list[index] = TransformedPokemonView(pokemon, target, moves=moves)
        else:
            active_list[index] = TransformedPokemonView(pokemon, target)
    return active_list


def decision_view(battle: DoubleBattle) -> DecisionView:
    """Extract a lightweight DecisionView from live battle state."""
    available_moves = battle.available_moves
    active_pokemon = _transformed_active_pokemon(battle, include_available_moves=True)

    available_switches = battle.available_switches
    team = tuple(battle.team.values())
    trapped = battle.trapped
    maybe_trapped = battle.maybe_trapped
    force_switch = battle.force_switch
    can_mega_evolve = battle.can_mega_evolve
    slots: list[SlotDecision] = []

    for position in (0, 1):
        active = active_pokemon[position]
        position_moves = available_moves[position]
        available_ids = {move.id for move in position_moves}

        move_targets = (
            ()
            if active is None
            else tuple(
                tuple(battle.get_possible_showdown_targets(move, active))
                if move.id in available_ids
                else ()
                for move in active.moves.values()
            )
        )

        # poke-env draws available switches out of battle.team itself, so
        # roster identity is the real relation; the species name was only ever a
        # proxy for it. Identity is matched by id() because poke-env's Pokemon
        # defines __eq__ and callers may pass unhashable stand-ins.
        switches = {id(pokemon) for pokemon in available_switches[position]}
        switch_slots = tuple(index for index, pokemon in enumerate(team) if id(pokemon) in switches)

        forced_move = (
            not any(move_targets)
            and len(position_moves) == 1
            and position_moves[0].id in {"struggle", "recharge"}
        )

        slots.append(
            SlotDecision(
                switch_slots=switch_slots,
                move_targets=move_targets,
                active=active is not None and not active.fainted,
                # Open team sheets make every trapping ability public, so a
                # request that reports maybe-trapped is in practice trapped.
                # Offering the switch anyway produced invalid choices live.
                trapped=trapped[position] or maybe_trapped[position],
                force_switch=force_switch[position],
                can_mega=can_mega_evolve[position],
                forced_move=forced_move,
            )
        )

    return DecisionView(
        slots=(slots[0], slots[1]),
        wait=battle._wait,
        team_preview=battle.teampreview,
        team_size=len(team),
    )
