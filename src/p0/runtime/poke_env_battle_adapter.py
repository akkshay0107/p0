"""Fast player-relative facade over poke-env battle state."""

from __future__ import annotations

from typing import Any, cast
from weakref import WeakKeyDictionary

from poke_env.battle import DoubleBattle

from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import TransformedPokemonView
from p0.runtime.live_event_capture import last_move


class PokeEnvBattleView:
    """Cached facade with explicit properties and no copied per-decision graph."""

    __slots__ = ("_battle", "_decision", "stat_cache")

    def __init__(self, battle: DoubleBattle):
        self._battle = battle
        self._decision: DecisionView | None = None
        self.stat_cache: dict[object, tuple[int, int, int, int, int, int]] = {}

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
        from typing import Any, cast

        active_list = cast(list[Any], list(self._battle.active_pokemon))
        targets = getattr(self._battle, "_p0_transform_targets", {})
        available = getattr(self._battle, "available_moves", [])
        for i, mon in enumerate(active_list):
            if mon is not None and id(mon) in targets:
                target_mon = targets[id(mon)]
                moves = dict(target_mon.moves)
                if i < len(available) and available[i]:
                    for move in available[i]:
                        if move.id not in moves:
                            moves[move.id] = move
                active_list[i] = TransformedPokemonView(mon, target_mon, moves=moves)
        return active_list

    @property
    def opponent_active_pokemon(self):
        from typing import Any, cast

        active_list = cast(list[Any], list(self._battle.opponent_active_pokemon))
        targets = getattr(self._battle, "_p0_transform_targets", {})
        for i, mon in enumerate(active_list):
            if mon is not None and id(mon) in targets:
                active_list[i] = TransformedPokemonView(mon, targets[id(mon)])
        return active_list

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


def decision_view(battle: DoubleBattle) -> DecisionView:
    """Extract a lightweight DecisionView from live battle state."""
    targets = getattr(battle, "_p0_transform_targets", {})
    available_moves = battle.available_moves
    active_pokemon = cast(list[Any], list(battle.active_pokemon))
    for i, mon in enumerate(active_pokemon):
        if mon is not None and id(mon) in targets:
            target_mon = targets[id(mon)]
            moves = dict(target_mon.moves)
            if i < len(available_moves) and available_moves[i]:
                for move in available_moves[i]:
                    if move.id not in moves:
                        moves[move.id] = move
            active_pokemon[i] = TransformedPokemonView(mon, target_mon, moves=moves)

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
