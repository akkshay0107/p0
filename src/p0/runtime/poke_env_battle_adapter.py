"""Fast player-relative facade over poke-env battle state."""

from __future__ import annotations

from weakref import WeakKeyDictionary

from poke_env.battle import DoubleBattle

from p0.battle.events import BattleEvent, parse_events
from p0.battle.legality import DecisionView, SlotDecision
from p0.model.tokenizer import tokenizer
from p0.runtime.live_event_capture import consume_raw_events, last_move
from p0.teams.stat_points import PrecomputedStats


class PokeEnvBattleView:
    """Cached facade with explicit properties and no copied per-decision graph."""

    __slots__ = ("_battle", "_decision", "_events", "_events_key", "stat_cache")

    def __init__(self, battle: DoubleBattle):
        self._battle = battle
        self._decision: DecisionView | None = None
        self._events: list[BattleEvent] = []
        self._events_key: int = -1  # id() is never negative, so -1 forces the first drain
        self.stat_cache: dict[object, PrecomputedStats] = {}

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
        return self._battle.active_pokemon

    @property
    def opponent_active_pokemon(self):
        return self._battle.opponent_active_pokemon

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

    def get_pokemon(self, identifier: str):
        return self._battle.get_pokemon(identifier)

    def consume_events(self):
        # The raw buffer drains once per decision; repeated observation builds
        # for the same decision (value/policy passes, candidate scoring) see the
        # identical event window. Consecutive requests are distinct objects, so
        # object identity is a safe per-decision key.
        key = id(self._battle.last_request)
        if key != self._events_key:
            self._events = parse_events(consume_raw_events(self._battle), tokenizer)
            self._events_key = key
        return self._events

    def last_move(self, pokemon):
        return last_move(pokemon)


_VIEWS: WeakKeyDictionary[DoubleBattle, PokeEnvBattleView] = WeakKeyDictionary()


def battle_view(battle: DoubleBattle) -> PokeEnvBattleView:
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
    active_pokemon = battle.active_pokemon
    available_moves = battle.available_moves
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
        switches = {pokemon.base_species for pokemon in available_switches[position]}
        switch_slots = tuple(
            index for index, pokemon in enumerate(team) if pokemon.base_species in switches
        )
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
