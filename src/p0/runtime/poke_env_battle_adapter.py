"""Fast player-relative facade over poke-env battle state."""

from __future__ import annotations

from weakref import WeakKeyDictionary

from poke_env.battle import DoubleBattle

from p0.battle.events import ProtocolEventParser
from p0.battle.legality import DecisionView, SlotDecision
from p0.model.tokenizer import tokenizer
from p0.runtime.live_event_capture import consume_raw_events, last_move
from p0.teams.stat_points import PrecomputedStats


class PokeEnvBattleView:
    """Cached facade with explicit properties and no copied per-decision graph."""

    __slots__ = ("_battle", "_decision", "stat_cache")

    def __init__(self, battle: DoubleBattle):
        self._battle = battle
        self._decision: DecisionView | None = None
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
            self._decision = PokeEnvBattleAdapter.decision_view(self._battle)
        return self._decision

    def get_pokemon(self, identifier: str):
        return self._battle.get_pokemon(identifier)

    def consume_events(self):
        return ProtocolEventParser.parse_events(consume_raw_events(self._battle), tokenizer)

    def last_move(self, pokemon):
        return last_move(pokemon)


class PokeEnvBattleAdapter:
    _views: WeakKeyDictionary[DoubleBattle, PokeEnvBattleView] = WeakKeyDictionary()

    @classmethod
    def view(cls, battle: DoubleBattle) -> PokeEnvBattleView:
        view = cls._views.get(battle)
        if view is None:
            view = PokeEnvBattleView(battle)
            cls._views[battle] = view
        return view.refresh()

    @classmethod
    def current_view(cls, battle: DoubleBattle) -> PokeEnvBattleView:
        """Return the decision's existing view, creating it only when necessary."""
        view = cls._views.get(battle)
        if view is None:
            view = PokeEnvBattleView(battle)
            cls._views[battle] = view
        return view

    @staticmethod
    def decision_view(battle: DoubleBattle) -> DecisionView:
        slots: list[SlotDecision] = []
        for position in (0, 1):
            active = battle.active_pokemon[position]
            available_ids = {move.id for move in battle.available_moves[position]}
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
            switches = {pokemon.base_species for pokemon in battle.available_switches[position]}
            switch_slots = tuple(
                index
                for index, pokemon in enumerate(battle.team.values())
                if pokemon.base_species in switches
            )
            forced_move = (
                not any(move_targets)
                and len(battle.available_moves[position]) == 1
                and battle.available_moves[position][0].id in {"struggle", "recharge"}
            )
            slots.append(
                SlotDecision(
                    switch_slots=switch_slots,
                    move_targets=move_targets,
                    active=active is not None and not active.fainted,
                    trapped=battle.trapped[position] or battle.maybe_trapped[position],
                    force_switch=battle.force_switch[position],
                    can_mega=battle.can_mega_evolve[position],
                    forced_move=forced_move,
                )
            )
        return DecisionView(
            slots=(slots[0], slots[1]),
            wait=battle._wait,
            team_preview=battle.teampreview,
            team_size=len(battle.team),
        )
