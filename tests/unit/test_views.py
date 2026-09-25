"""Tests for battle view structural protocols and transformed view wrappers."""

from __future__ import annotations

from poke_env.battle import Pokemon
from poke_env.battle.move import Move
from poke_env.teambuilder.teambuilder import TeambuilderPokemon

from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import (
    BattleView,
    FieldView,
    FixtureBattleView,
    MoveView,
    PokemonView,
    TransformedMoveView,
    TransformedPokemonView,
)


def _pokemon(
    species: str,
    *,
    ability: str,
    item: str,
    moves: list[str],
) -> Pokemon:
    return Pokemon(
        gen=9,
        teambuilder=TeambuilderPokemon(
            species=species,
            ability=ability,
            item=item,
            moves=moves,
        ),
    )


def _type_check_views(
    battle: BattleView, field: FieldView, pokemon: PokemonView, move: MoveView
) -> bool:
    """Type-level validation that objects satisfy Protocol contracts."""
    return (
        bool(battle.turn >= 0)
        and bool(field.turn >= 0)
        and bool(pokemon.base_species)
        and bool(move.id)
    )


class TestTransformedMoveView:
    def test_transformed_move_view_clamps_pp_and_delegates(self) -> None:
        """Verify that TransformedMoveView clamps PP to 5 and delegates other attributes."""
        base_move = Move("surf", 9)
        transformed = TransformedMoveView(base_move)

        assert transformed.id == "surf"
        assert transformed.type == base_move.type
        assert transformed.category == base_move.category
        assert transformed.current_pp == 5
        assert transformed.max_pp == 5
        assert transformed.non_ghost_target is base_move.non_ghost_target
        assert transformed.deduced_target == base_move.deduced_target


class TestTransformedPokemonView:
    def test_transformed_pokemon_view_equality_and_hashing(self) -> None:
        """Verify equality symmetry, hashing, and comparison between transformed views and base."""
        base1 = _pokemon("ditto", ability="imposter", item="choicescarf", moves=["transform"])
        base2 = _pokemon("smeargle", ability="own tempo", item="choicescarf", moves=["transform"])
        target = _pokemon(
            "koraidon", ability="orichalcum pulse", item="rusted sword", moves=["collision course"]
        )

        t1 = TransformedPokemonView(base1, target)
        t2 = TransformedPokemonView(base1, target)
        t3 = TransformedPokemonView(base2, target)

        assert t1 == t2
        assert t2 == t1
        assert t1 == base1
        assert base1 == t1
        assert t1 != t3
        assert hash(t1) == hash(base1)
        assert hash(t1) == hash(t2)


class TestFixtureBattleView:
    def test_fixture_battle_view_initialization_and_protocol(self) -> None:
        """Verify FixtureBattleView implements BattleView protocol and helper methods."""
        p1 = _pokemon("incineroar", ability="intimidate", item="choicescarf", moves=["fake out"])
        decision = DecisionView(slots=(SlotDecision(), SlotDecision()))
        view = FixtureBattleView(
            team={"p1: Incineroar": p1},
            opponent_team={},
            active_pokemon=(p1, None),
            opponent_active_pokemon=(None, None),
            available_moves=((), ()),
            available_switches=((), ()),
            can_mega_evolve=(False, False),
            force_switch=(False, False),
            trapped=(False, False),
            maybe_trapped=(False, False),
            teampreview=False,
            player_role="p1",
            wait=False,
            weather={},
            fields={},
            side_conditions={},
            opponent_side_conditions={},
            turn=1,
            used_mega_evolve=False,
            opponent_used_mega_evolve=False,
            decision=decision,
            identifiers={"p1: Incineroar": p1},
        )

        assert _type_check_views(view, view, p1, p1.moves["fakeout"])
        assert view.get_pokemon("p1: Incineroar") is p1
        assert view.last_move(p1) is None
        assert view.spatial_events == ()
