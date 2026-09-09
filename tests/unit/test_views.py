"""Tests for battle view structural protocols and transformed view wrappers."""

from __future__ import annotations

from typing import Any

from p0.battle.events import SpatialSlotRecord
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


class DummyMove:
    def __init__(
        self,
        move_id: str = "thunderbolt",
        move_type: str = "electric",
        category: str = "special",
        current_pp: int = 15,
        max_pp: int = 24,
        non_ghost_target: bool = False,
        deduced_target: str = "normal",
    ) -> None:
        self.id = move_id
        self.type = move_type
        self.category = category
        self.current_pp = current_pp
        self.max_pp = max_pp
        self.non_ghost_target = non_ghost_target
        self.deduced_target = deduced_target


class DummyPokemon:
    def __init__(
        self,
        species: str = "Pikachu",
        base_species: str = "Pikachu",
        ability: str = "static",
        item: str = "lightball",
        nature: str = "timid",
        types: tuple[str, ...] = ("electric",),
        current_hp_fraction: float = 1.0,
        moves: dict[str, Any] | None = None,
        level: int = 50,
    ) -> None:
        self.species = species
        self.base_species = base_species
        self.ability = ability
        self.item = item
        self.nature = nature
        self.types = types
        self.type_1 = types[0] if types else "typeless"
        self.type_2 = types[1] if len(types) > 1 else None
        self.current_hp_fraction = current_hp_fraction
        self.moves = moves or {"thunderbolt": DummyMove()}
        self.level = level
        self.status = None
        self.base_stats = {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90}
        self.stats = {"hp": 110, "atk": 75, "def": 60, "spa": 70, "spd": 70, "spe": 110}
        self.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        self.protect_counter = 0
        self.first_turn = False
        self.weight = 6.0
        self.fainted = False
        self.revealed = True
        self.selected_in_teampreview = True
        self.effects: dict[Any, int] = {}
        self.status_counter = 0
        self.preparing = None
        self.last_move = DummyMove()
        self.is_dynamaxed = False
        self.is_terastallized = False
        self.tera_type = "electric"


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
        base_move = DummyMove(
            move_id="surf",
            move_type="water",
            category="special",
            current_pp=15,
            max_pp=24,
            non_ghost_target=True,
            deduced_target="allAdjacentFoes",
        )
        transformed = TransformedMoveView(base_move)

        assert transformed.id == "surf"
        assert transformed.type == "water"
        assert transformed.category == "special"
        assert transformed.current_pp == 5
        assert transformed.max_pp == 5
        assert transformed.non_ghost_target is True
        assert transformed.deduced_target == "allAdjacentFoes"


class TestTransformedPokemonView:
    def test_transformed_pokemon_view_combines_base_and_target(self) -> None:
        """Verify that TransformedPokemonView overlays target stats/species while preserving base identity."""
        base = DummyPokemon(
            species="Ditto",
            base_species="Ditto",
            ability="imposter",
            item="choicescarf",
            nature="jolly",
            types=("normal",),
            current_hp_fraction=0.85,
        )
        target = DummyPokemon(
            species="Miraidon",
            base_species="Miraidon",
            ability="hadronengine",
            item="lifeorb",
            nature="modest",
            types=("electric", "dragon"),
            current_hp_fraction=0.40,
            moves={"electrodrift": DummyMove(move_id="electrodrift", move_type="electric")},
        )
        transformed = TransformedPokemonView(base, target)

        # Target properties
        assert transformed.species == "Miraidon"
        assert transformed.base_species == "Miraidon"
        assert transformed.ability == "hadronengine"
        assert transformed.type_1 == "electric"
        assert transformed.type_2 == "dragon"
        assert transformed.types == ("electric", "dragon")
        assert transformed.base_stats == target.base_stats
        assert transformed.stats == target.stats
        assert transformed.weight == 6.0

        # Base properties preserved
        assert transformed.item == "choicescarf"
        assert transformed.nature == "jolly"
        assert transformed.current_hp_fraction == 0.85
        assert transformed.level == 50
        assert transformed.fainted is False
        assert transformed.revealed is True
        assert transformed.selected_in_teampreview is True
        assert transformed.status is None
        assert transformed.protect_counter == 0
        assert transformed.first_turn is False
        assert transformed.effects == {}
        assert transformed.status_counter == 0
        assert transformed.preparing is None
        assert transformed.last_move is not None and transformed.last_move.id == "thunderbolt"
        assert transformed.is_dynamaxed is False
        assert transformed.is_terastallized is False
        assert transformed.tera_type == "electric"
        assert transformed.boosts == {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}

        # Transformed moves have PP clamped to 5
        assert "electrodrift" in transformed.moves
        assert transformed.moves["electrodrift"].current_pp == 5
        assert transformed.moves["electrodrift"].max_pp == 5

    def test_transformed_pokemon_view_equality_and_hashing(self) -> None:
        """Verify equality symmetry, hashing, and comparison between transformed views and base."""
        base1 = DummyPokemon(species="Ditto")
        base2 = DummyPokemon(species="Smeargle")
        target = DummyPokemon(species="Koraidon")

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
        p1 = DummyPokemon(species="Incineroar")
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

        assert _type_check_views(view, view, p1, p1.moves["thunderbolt"])
        assert view.get_pokemon("p1: Incineroar") is p1
        assert view.last_move(p1) == "thunderbolt"
        assert len(view.spatial_turn) == 4
        assert all(isinstance(rec, SpatialSlotRecord) for rec in view.spatial_turn)
        assert view.last_move(DummyPokemon(moves={})) is not None
