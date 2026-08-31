from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
import torch
from poke_env.battle import Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from p0.battle.events import (
    SPATIAL_CATEGORICAL_WIDTH,
    SPATIAL_NUMERICAL_WIDTH,
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialSlotRecord,
    SpatialTargetSlot,
)
from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import FixtureBattleView
from p0.model.observation_builder import (
    ObservationBuilder,
)
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_MECHANIC_STATE,
    CAT_IDX_NATURE,
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    EFFECT_CATEGORICAL_WIDTH,
    MAX_EFFECTS,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_CAN_SWITCH_OUT,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_PREPARING,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_STAT_PROVENANCE,
    NUM_IDX_TEAM_PREVIEW,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    TOKEN_IDX_GLOBAL_FIELD,
    TOKEN_IDX_OPPONENT_SIDE,
    EffectNamespace,
    IdentityKnownness,
    MechanicState,
    PresenceStatus,
    SideId,
    StatProvenance,
    StructuredObservation,
    TokenType,
)
from p0.model.tokenizer import tokenizer


@dataclass(frozen=True, slots=True)
class ObservationMove:
    """Concrete move value used to exercise the public MoveView protocol."""

    id: str
    type: Any
    category: Any
    current_pp: int
    max_pp: int
    non_ghost_target: bool = True
    deduced_target: Any = None


@dataclass(eq=False, slots=True)
class ObservationPokemon:
    """Concrete Pokémon value used to exercise the public PokemonView protocol."""

    species: str | None
    base_species: str
    ability: str | None
    item: str | None
    nature: str | None
    moves: dict[str, ObservationMove] = field(default_factory=dict)
    type_1: Any = None
    type_2: Any = None
    status: Any = None
    base_stats: dict[str, int] = field(default_factory=dict)
    stats: dict[str, int | None] | None = None
    boosts: dict[str, int] = field(default_factory=dict)
    current_hp_fraction: float = 0.0
    protect_counter: int = 0
    first_turn: bool = False
    weight: float = 0.0
    fainted: bool = False
    revealed: bool = False
    selected_in_teampreview: bool | None = False
    effects: dict[Any, int] = field(default_factory=dict)
    status_counter: int = 0
    preparing: Any = False
    last_move: ObservationMove | None = None
    level: int | None = 50
    is_dynamaxed: bool = False
    is_terastallized: bool = False
    tera_type: Any = None
    types: tuple[Any, ...] = ()


def make_pokemon_view(
    species: str = "charizard",
    ability: str = "blaze",
    item: str = "charizarditey",
    type_1: str | None = None,
    type_2: str | None = None,
    moves: dict[str, int] | None = None,
    effects: dict[Effect, int] | None = None,
    status: Status | None = None,
    current_hp: int = 100,
    max_hp: int = 100,
    boosts: dict[str, int] | None = None,
    protect_counter: int = 0,
    active_turns: int = 0,
    weightkg: float | None = None,
    status_counter: int = 0,
    preparing_move: str | None = None,
    last_move_id: str | None = None,
    nature: str | None = None,
    reported_species: str | None = None,
) -> ObservationPokemon:
    """Build a concrete value object implementing the public PokemonView protocol."""
    source = Pokemon(gen=9, species=species)
    move_views: dict[str, ObservationMove] = {}
    for move_id, move_pp in (moves or {}).items():
        source_move = Move(move_id, 9)
        move_views[move_id] = ObservationMove(
            id=source_move.id,
            type=source_move.type,
            category=source_move.category,
            current_pp=move_pp,
            max_pp=source_move.max_pp or move_pp,
            non_ghost_target=source_move.non_ghost_target,
            deduced_target=source_move.deduced_target,
        )

    base_stats = dict(source.base_stats)
    boost_values = {
        "accuracy": 0,
        "atk": 0,
        "def": 0,
        "evasion": 0,
        "spa": 0,
        "spd": 0,
        "spe": 0,
    }
    boost_values.update(boosts or {})
    first_type = PokemonType.from_name(type_1) if type_1 else source.type_1
    second_type = PokemonType.from_name(type_2) if type_2 else source.type_2
    effective_species = species if reported_species is None else reported_species
    effective_base_species = source.base_species if reported_species is None else reported_species
    return ObservationPokemon(
        species=effective_species,
        base_species=effective_base_species,
        ability=ability,
        item=item,
        nature=nature,
        moves=move_views,
        type_1=first_type,
        type_2=second_type,
        status=status,
        base_stats=base_stats,
        boosts=boost_values,
        current_hp_fraction=current_hp / max_hp if max_hp else 0.0,
        protect_counter=protect_counter,
        first_turn=active_turns == 1,
        weight=source.weight if weightkg is None else weightkg,
        fainted=status is Status.FNT or current_hp <= 0,
        effects=effects or {},
        status_counter=status_counter,
        preparing=preparing_move is not None,
        last_move=move_views.get(last_move_id) if last_move_id else None,
        types=tuple(value for value in (first_type, second_type) if value is not None),
    )


def make_battle_view(
    active_pokemon: Sequence[ObservationPokemon | None] | None = None,
    opponent_active_pokemon: Sequence[ObservationPokemon | None] | None = None,
    team: Sequence[ObservationPokemon] | None = None,
    opponent_team: Sequence[ObservationPokemon] | None = None,
    teampreview: bool = False,
    available_switches: Sequence[Sequence[ObservationPokemon]] | None = None,
    weather: dict[Weather, int] | None = None,
    fields: dict[Field, int] | None = None,
    turn: int = 0,
    can_mega_evolve: list[bool] | None = None,
    side_conditions: dict[SideCondition, int] | None = None,
    opponent_side_conditions: dict[SideCondition, int] | None = None,
    decision: DecisionView | None = None,
) -> FixtureBattleView:
    """Build a concrete BattleView value object from public observation inputs."""
    active = (tuple(active_pokemon or ()) + (None, None))[:2]
    opponent_active = (tuple(opponent_active_pokemon or ()) + (None, None))[:2]
    allies = tuple(team or ())
    opponents = tuple(opponent_team or ())
    selected_decision = decision or DecisionView(slots=(SlotDecision(), SlotDecision()))
    return FixtureBattleView(
        team={f"p1:{index}": pokemon for index, pokemon in enumerate(allies)},
        opponent_team={f"p2:{index}": pokemon for index, pokemon in enumerate(opponents)},
        active_pokemon=active,
        opponent_active_pokemon=opponent_active,
        available_moves=tuple(
            tuple(pokemon.moves.values()) if pokemon is not None else () for pokemon in active
        ),
        available_switches=available_switches or [[], []],
        can_mega_evolve=can_mega_evolve or [False, False],
        force_switch=(False, False),
        trapped=(False, False),
        maybe_trapped=(False, False),
        teampreview=teampreview,
        player_role="p1",
        wait=False,
        weather=weather or {},
        fields=fields or {},
        side_conditions=side_conditions or {},
        opponent_side_conditions=opponent_side_conditions or {},
        turn=turn,
        used_mega_evolve=False,
        opponent_used_mega_evolve=False,
        decision=selected_decision,
    )


_OBSERVATION_BUILDER = ObservationBuilder(default_runtime_resources())


def from_battle(battle, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    return _OBSERVATION_BUILDER.build(battle, stat_overrides)


def from_battle_into(battle, out, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    _OBSERVATION_BUILDER.build_into(battle, out, stat_overrides)


def test_observation_builder_serializes_pokemon_features() -> None:
    """
    Verify ObservationBuilder converts Pokemon attributes into correctly scaled and indexed categorical/numerical tensors.

    Verifies:
    - Categorical columns: species, ability, item, typing (type 1 & 2), move slots (0..3), status condition, nature.
    - Numerical columns: HP fraction, base stats (scaled by 160), stat boosts (scaled by 6), move PP ratios, protect counter,
      weight bracket, status duration counter, move preparation flag, mega evolution availability.
    """
    builder = _OBSERVATION_BUILDER

    mon = make_pokemon_view(
        species="charizard",
        ability="blaze",
        item="charizarditey",
        type_1="Fire",
        type_2="Flying",
        moves={"closecombat": 4, "protect": 8},
        effects={Effect.CONFUSION: 2, Effect.DISABLE: 1},
        status=Status.BRN,
        current_hp=80,
        max_hp=100,
        boosts={"atk": 3, "def": -1},
        protect_counter=2,
        active_turns=1,
        weightkg=75.0,
        status_counter=3,
        preparing_move="closecombat",
    )
    mon.nature = "Jolly"
    battle = make_battle_view(active_pokemon=[mon, None], team=[mon], can_mega_evolve=[True, False])

    obs = builder.build(battle)

    # Validate Categoricals
    cat = obs.categorical[0]
    assert cat[0] == tokenizer.species_id(mon)
    assert cat[1] == tokenizer.ability_id(mon)
    assert cat[2] == tokenizer.item_id(mon)
    assert cat[3] == tokenizer.type_id(PokemonType.from_name("Fire"))
    assert cat[4] == tokenizer.type_id(PokemonType.from_name("Flying"))
    assert cat[5] == tokenizer.move_id(Move("closecombat", 9))
    assert cat[6] == tokenizer.move_id(Move("protect", 9))
    assert cat[7] == 0 and cat[8] == 0  # padded move slots
    assert cat[17] == tokenizer.status_id(Status.BRN)
    assert cat[CAT_IDX_NATURE] == tokenizer.nature_id(mon)

    # Validate Numericals
    num = obs.numerical[0]
    assert num[5] == pytest.approx(0.8)  # HP fraction (80 / 100)
    assert abs(num[6] - 78.0 / 160.0) < 1e-5  # Base HP normalized by 160
    assert abs(num[7] - 84.0 / 160.0) < 1e-5  # Base Atk normalized by 160
    assert num[12] == 3.0 / 6.0  # Atk Boost (+3 / 6)
    assert num[13] == -1.0 / 6.0  # Def Boost (-1 / 6)
    assert num[19] == 4.0 / 8.0  # Move 1 PP ratio (4 / 8 max PP)
    assert num[20] == 8.0 / 16.0  # Move 2 PP ratio (8 / 16 max PP)
    assert num[23] == 2.0 / 4.0  # Protect counter normalized by 4
    assert num[24] == 1.0  # First turn active
    assert num[25] == pytest.approx(0.6)  # Low kick weight bracket for 75kg
    assert num[36] == 3.0 / 5.0  # Status counter normalized by 5
    assert num[NUM_IDX_PREPARING] == 1.0  # Preparing move flag
    assert num[NUM_IDX_CAN_MEGA] == 1.0  # Can mega evolve flag


def test_ordered_pokemon_and_slot_conditions_real() -> None:
    """Verify public observations preserve roster order and active-slot placement."""
    p1 = make_pokemon_view(species="aerodactyl")
    p2 = make_pokemon_view(species="archaludon")
    p3 = make_pokemon_view(species="azumarill")
    p4 = make_pokemon_view(species="basculegion")
    p5 = make_pokemon_view(species="camerupt", status=Status.FNT)
    p6 = make_pokemon_view(species="dragonite")
    team = [p1, p2, p3, p4, p5, p6]

    # Teampreview ordering
    battle_tp = make_battle_view(team=team, teampreview=True)
    preview = _OBSERVATION_BUILDER.build(battle_tp)
    assert preview.categorical[:6, 0].tolist() == [
        tokenizer.species_id(pokemon) for pokemon in team
    ]

    # Regular battle ordering (active slots first, then bench)
    battle_reg = make_battle_view(
        active_pokemon=[p1, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p4], [p3, p4]],
    )
    regular = _OBSERVATION_BUILDER.build(battle_reg)
    assert regular.categorical[:6, 0].tolist() == [
        tokenizer.species_id(pokemon) for pokemon in (p1, p2, p3, p4, p5, p6)
    ]

    # Empty left active slot placeholder
    battle_left_empty = make_battle_view(
        active_pokemon=[None, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p4], [p3, p4]],
    )
    left_empty = _OBSERVATION_BUILDER.build(battle_left_empty)
    assert left_empty.categorical[0, 0] == 0
    assert left_empty.categorical[1, 0] == tokenizer.species_id(p2)
    assert left_empty.numerical[0, 1] == 1.0
    assert left_empty.numerical[1, 2] == 1.0


def test_global_and_side_field_tokens_include_mega_availability() -> None:
    """Verify global field and side tokens include turn fraction scaling, team preview indicator, and mega availability."""
    ally_mega = make_pokemon_view(species="charizard", item="charizarditey")
    battle = make_battle_view(
        active_pokemon=[ally_mega, None],
        team=[ally_mega],
        weather={Weather.SUNNYDAY: 3},
        fields={Field.ELECTRIC_TERRAIN: 4},
        side_conditions={SideCondition.REFLECT: 2},
        opponent_side_conditions={SideCondition.LIGHT_SCREEN: 1},
        turn=5,
        teampreview=False,
        can_mega_evolve=[True, False],
    )

    obs = _OBSERVATION_BUILDER.build(battle)

    # Global Field Token (index 12)
    assert obs.token_type_ids[TOKEN_IDX_GLOBAL_FIELD] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, 3] == pytest.approx(5 / 24.0)  # turn / 24
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] == 0.0

    # Ally Side Token (index 13)
    assert obs.token_type_ids[TOKEN_IDX_ALLY_SIDE] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_ALLY_SIDE, 4] == 1.0  # Ally mega available

    # Opponent Side Token (index 14)
    assert obs.token_type_ids[TOKEN_IDX_OPPONENT_SIDE] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_OPPONENT_SIDE, 4] == 0.0


def test_effect_overflow_is_counted_and_enforced() -> None:
    """Verify effect capacity capping: slots record exact effect count, overflow count, and truncate to MAX_EFFECTS."""
    effects = {
        Effect.TAUNT: 1,
        Effect.LEECH_SEED: 1,
        Effect.SUBSTITUTE: 1,
        Effect.SALT_CURE: 1,
        Effect.ENCORE: 1,
        Effect.DISABLE: 1,
        Effect.YAWN: 2,
        Effect.PERISH3: 3,
        Effect.TRAPPED: 1,
        Effect.CONFUSION: 1,
        Effect.HEAL_BLOCK: 1,
        Effect.TORMENT: 1,
        Effect.INFESTATION: 1,
        Effect.OCTOLOCK: 1,
    }
    mon = make_pokemon_view(effects=effects)
    battle = make_battle_view(active_pokemon=[mon, None], team=[mon])

    obs = _OBSERVATION_BUILDER.build(battle)
    # Total effects: 14, overflow over MAX_EFFECTS (12) is 2
    assert obs.numerical[0, NUM_IDX_EFFECT_COUNT] == 14
    assert obs.numerical[0, NUM_IDX_EFFECT_OVERFLOW] == 2.0
    pokemon_effects = obs.categorical[0, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == MAX_EFFECTS


def test_concurrent_universal_effect_stress_state() -> None:
    """Verify simultaneous active effects across Pokémon, side conditions, weather, and terrain tokens."""
    mon = make_pokemon_view(
        effects={
            Effect.TAUNT: 1,
            Effect.LEECH_SEED: 1,
            Effect.SUBSTITUTE: 1,
            Effect.SALT_CURE: 1,
            Effect.ENCORE: 1,
            Effect.DISABLE: 1,
            Effect.YAWN: 2,
            Effect.PERISH3: 3,
            Effect.TRAPPED: 1,
        }
    )
    battle = make_battle_view(
        active_pokemon=[mon, None],
        team=[mon],
        turn=3,
        side_conditions={
            SideCondition.REFLECT: 1,
            SideCondition.LIGHT_SCREEN: 1,
            SideCondition.AURORA_VEIL: 1,
            SideCondition.TAILWIND: 1,
            SideCondition.SAFEGUARD: 1,
            SideCondition.SPIKES: 3,
            SideCondition.TOXIC_SPIKES: 2,
        },
        weather={Weather.RAINDANCE: 1},
        fields={
            Field.GRASSY_TERRAIN: 1,
            Field.TRICK_ROOM: 1,
            Field.WONDER_ROOM: 1,
            Field.MAGIC_ROOM: 1,
            Field.GRAVITY: 1,
        },
    )

    obs = _OBSERVATION_BUILDER.build(battle)

    assert obs.numerical[0, NUM_IDX_EFFECT_COUNT] == 9
    assert obs.numerical[13, NUM_IDX_EFFECT_COUNT] == 7
    assert obs.numerical[12, NUM_IDX_EFFECT_COUNT] == 6
    assert obs.numerical[:, NUM_IDX_EFFECT_OVERFLOW].sum() == 0
    pokemon_effects = obs.categorical[0, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == 9
    namespaces = obs.categorical[12, CAT_EFFECT_START + 2 :: EFFECT_CATEGORICAL_WIDTH]
    assert EffectNamespace.FIELD in namespaces
    assert EffectNamespace.WEATHER in namespaces


def test_spatial_events_ground_to_slots() -> None:
    """Verify spatial turn records write to spatial_cat and spatial_num tensors accurately."""
    turn_records = (
        SpatialSlotRecord(
            action_type=int(SpatialActionType.MOVE),
            move_id=15,
            target_slot=int(SpatialTargetSlot.OPP_LEFT),
            order_rank=0.25,
            damage_dealt=0.55,
            landed_crit=1.0,
        ),
        SpatialSlotRecord(action_type=int(SpatialActionType.SWITCH)),
        SpatialSlotRecord(
            action_type=int(SpatialActionType.MOVE),
            move_id=42,
            target_slot=int(SpatialTargetSlot.ALLY_LEFT),
            order_rank=0.50,
            hp_delta=-0.55,
            took_crit=1.0,
        ),
        SpatialSlotRecord(action_type=int(SpatialActionType.NONE)),
    )
    fixture = _legality_fixture_view(DecisionView(slots=(SlotDecision(), SlotDecision())))
    fixture.spatial_turn = turn_records
    builder = ObservationBuilder(default_runtime_resources())
    obs = builder.build(fixture)

    assert obs.spatial_cat.shape == (SPATIAL_SLOT_COUNT, SPATIAL_CATEGORICAL_WIDTH)
    assert obs.spatial_num.shape == (SPATIAL_SLOT_COUNT, SPATIAL_NUMERICAL_WIDTH)
    assert obs.spatial_cat[0, 0].item() == int(SpatialActionType.MOVE)
    assert obs.spatial_cat[0, 1].item() == 15
    assert obs.spatial_cat[0, 2].item() == int(SpatialTargetSlot.OPP_LEFT)
    assert obs.spatial_num[0, 4].item() == 1.0  # landed_crit
    assert obs.spatial_num[2, 5].item() == 1.0  # took_crit


def test_from_battle_into_overwrites_and_validates_output_buffer() -> None:
    """Verify from_battle_into performs in-place tensor writing into pre-allocated memory buffers without stale artifact leakage."""
    ally = make_pokemon_view(
        species="charizard",
        moves={"airslash": 10, "protect": 8},
        effects={Effect.CONFUSION: 2},
        current_hp=73,
        max_hp=100,
    )
    opponent = make_pokemon_view(
        species="venusaur",
        moves={"gigadrain": 6},
        status=Status.BRN,
    )
    battle = make_battle_view(
        active_pokemon=[ally, None],
        opponent_active_pokemon=[opponent, None],
        team=[ally],
        opponent_team=[opponent],
        weather={Weather.SUNNYDAY: 1},
        fields={Field.TRICK_ROOM: 2},
        turn=3,
    )

    expected = from_battle(battle, tokenizer)
    out = StructuredObservation.empty_batch(1)[0]
    # Poison buffer with sentinel 99 values to ensure everything is cleanly overwritten
    out.token_type_ids.fill_(99)
    out.side_ids.fill_(99)
    out.slot_ids.fill_(99)
    out.categorical.fill_(99)
    out.numerical.fill_(99.0)
    out.spatial_cat.fill_(99)
    out.spatial_num.fill_(99.0)

    from_battle_into(battle, out, tokenizer)

    assert torch.equal(out.token_type_ids, expected.token_type_ids)
    assert torch.equal(out.side_ids, expected.side_ids)
    assert torch.equal(out.slot_ids, expected.slot_ids)
    assert torch.equal(out.categorical, expected.categorical)
    assert torch.equal(out.numerical, expected.numerical)
    assert torch.equal(out.spatial_cat, expected.spatial_cat)
    assert torch.equal(out.spatial_num, expected.spatial_num)
    assert not torch.any(out.categorical == 99)
    assert not torch.any(out.numerical == 99)

    # Validate dtype validation rejection
    invalid = StructuredObservation.empty_batch(1)[0]
    invalid.numerical = invalid.numerical.to(torch.float64)
    with pytest.raises(ValueError, match="Invalid numerical"):
        from_battle_into(battle, invalid)


def test_observation_builder_identity_knownness_and_stat_provenance() -> None:
    """Verify IdentityKnownness (KNOWN/OOV/UNKNOWN/PAD) and StatProvenance (KNOWN/IMPUTED/UNKNOWN/PAD)."""
    builder = ObservationBuilder(default_runtime_resources())

    # 1. Standard valid species (KNOWN)
    valid_ally = make_pokemon_view(species="charizard")
    valid_ally.stats = {"hp": 153, "atk": 104, "def": 98, "spa": 177, "spd": 105, "spe": 152}

    # 2. Out-of-vocabulary species (OOV)
    oov_mon = make_pokemon_view(species="pikachu", reported_species="nonexistentspecies123")

    # 3. Missing species (UNKNOWN)
    unknown_mon = make_pokemon_view(species="pikachu", reported_species="")

    # 4. Standard valid opponent with imputed stats (IMPUTED)
    valid_opp = make_pokemon_view(species="pikachu", nature="timid")

    battle = make_battle_view(
        active_pokemon=[valid_ally, oov_mon],
        opponent_active_pokemon=[valid_opp, unknown_mon],
        team=[valid_ally, oov_mon],
        opponent_team=[valid_opp, unknown_mon],
    )

    obs = builder.build(battle)

    # Token 0: Ally Charizard with exact stats -> KNOWN identity, KNOWN stats, ACTIVE presence, NORMAL mechanic
    assert obs.categorical[0, CAT_IDX_IDENTITY_KNOWNNESS] == IdentityKnownness.KNOWN
    assert obs.categorical[0, CAT_IDX_STAT_PROVENANCE] == StatProvenance.KNOWN
    assert obs.categorical[0, CAT_IDX_PRESENCE_STATUS] == PresenceStatus.ACTIVE
    assert obs.categorical[0, CAT_IDX_MECHANIC_STATE] == MechanicState.NORMAL
    assert obs.numerical[0, NUM_IDX_STAT_PROVENANCE] == 1.0

    # Token 1: Ally with OOV species -> OOV identity, UNKNOWN stats, ACTIVE presence
    assert obs.categorical[1, CAT_IDX_IDENTITY_KNOWNNESS] == IdentityKnownness.OOV
    assert obs.categorical[1, CAT_IDX_STAT_PROVENANCE] == StatProvenance.UNKNOWN
    assert obs.categorical[1, CAT_IDX_PRESENCE_STATUS] == PresenceStatus.ACTIVE
    assert obs.categorical[1, CAT_IDX_MECHANIC_STATE] == MechanicState.NORMAL
    assert obs.numerical[1, NUM_IDX_STAT_PROVENANCE] == 0.0

    # Token 2..5: Empty ally slots -> PAD
    for i in range(2, 6):
        assert obs.categorical[i, CAT_IDX_IDENTITY_KNOWNNESS] == IdentityKnownness.PAD
        assert obs.categorical[i, CAT_IDX_STAT_PROVENANCE] == StatProvenance.PAD
        assert obs.categorical[i, CAT_IDX_PRESENCE_STATUS] == PresenceStatus.PAD
        assert obs.categorical[i, CAT_IDX_MECHANIC_STATE] == MechanicState.NORMAL
        assert obs.numerical[i, NUM_IDX_STAT_PROVENANCE] == 0.0

    # Token 6: Opponent Pikachu -> KNOWN identity, IMPUTED stats, ACTIVE presence
    assert obs.categorical[6, CAT_IDX_IDENTITY_KNOWNNESS] == IdentityKnownness.KNOWN
    assert obs.categorical[6, CAT_IDX_STAT_PROVENANCE] == StatProvenance.IMPUTED
    assert obs.categorical[6, CAT_IDX_PRESENCE_STATUS] == PresenceStatus.ACTIVE
    assert obs.categorical[6, CAT_IDX_MECHANIC_STATE] == MechanicState.NORMAL
    assert obs.numerical[6, NUM_IDX_STAT_PROVENANCE] == 0.0

    # Token 7: Opponent with missing species -> UNKNOWN identity, UNKNOWN stats, ACTIVE presence
    assert obs.categorical[7, CAT_IDX_IDENTITY_KNOWNNESS] == IdentityKnownness.UNKNOWN
    assert obs.categorical[7, CAT_IDX_STAT_PROVENANCE] == StatProvenance.UNKNOWN
    assert obs.categorical[7, CAT_IDX_PRESENCE_STATUS] == PresenceStatus.ACTIVE
    assert obs.categorical[7, CAT_IDX_MECHANIC_STATE] == MechanicState.NORMAL
    assert obs.numerical[7, NUM_IDX_STAT_PROVENANCE] == 0.0


def test_observation_overflow_contract_holds_at_capacity_boundaries() -> None:
    """Verify validate_overflow_contract verifies effect overflow totals."""
    observation = StructuredObservation.empty_batch(1)[0]
    observation.numerical[:, NUM_IDX_EFFECT_COUNT] = torch.tensor(
        (0,) * 12 + (MAX_EFFECTS, MAX_EFFECTS + 2, 0),
        dtype=torch.float32,
    )
    observation.numerical[:, NUM_IDX_EFFECT_OVERFLOW] = torch.tensor(
        (0.0,) * 12 + (0.0, 2.0, 0.0),
        dtype=torch.float32,
    )
    observation.validate_overflow_contract()
    assert observation.overflow_totals() == (2, 0)


def _legality_fixture_view(decision: DecisionView) -> FixtureBattleView:
    allies = [make_pokemon_view(species=species) for species in ("charizard", "blastoise")]
    bench = make_pokemon_view(species="pikachu")
    opponent = make_pokemon_view(species="venusaur")
    return make_battle_view(
        active_pokemon=allies,
        opponent_active_pokemon=[opponent, None],
        team=[*allies, bench],
        opponent_team=[opponent],
        available_switches=[[bench], [bench]],
        decision=decision,
    )


def test_unknown_legality_is_gated_rather_than_written_as_illegal() -> None:
    """Verify that when decision legality is unknown, legality columns are 0 and unknown legality gate flags are raised (1.0)."""
    builder = ObservationBuilder(default_runtime_resources())
    slot = SlotDecision(switch_slots=(2,), move_targets=((-2, 1), (), (), ()), can_mega=True)
    proven = builder.build(_legality_fixture_view(DecisionView(slots=(slot, slot))))

    unknown_slot = replace(slot, legality_known=False)
    unknown = builder.build(
        _legality_fixture_view(DecisionView(slots=(unknown_slot, unknown_slot)))
    )

    legality_columns = slice(NUM_IDX_MOVE_LEGAL, NUM_IDX_CAN_SWITCH_OUT + 1)
    assert proven.numerical[0, legality_columns].any()
    assert not unknown.numerical[:, legality_columns].any()
    assert unknown.numerical[0, NUM_IDX_CAN_MEGA] == 0.0

    # Gate columns must indicate unknown status on active slots
    assert proven.numerical[:, NUM_IDX_LEGALITY_UNKNOWN].tolist() == [0.0] * SEQUENCE_LENGTH
    assert unknown.numerical[:2, NUM_IDX_LEGALITY_UNKNOWN].tolist() == [1.0, 1.0]
    assert unknown.numerical[2:, NUM_IDX_LEGALITY_UNKNOWN].sum() == 0.0

    gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
    assert unknown.numerical[TOKEN_IDX_ALLY_SIDE, gates].tolist() == [1.0, 1.0]
    assert proven.numerical[TOKEN_IDX_ALLY_SIDE, gates].tolist() == [0.0, 0.0]
    assert unknown.numerical[TOKEN_IDX_OPPONENT_SIDE, gates].tolist() == [0.0, 0.0]


def test_reconstructed_observations_clear_reused_buffer_state() -> None:
    """Verify build_into on reconstructed replay snapshots clears and overwrites observation buffer tensors across turns."""
    from p0.replays.compile import compile_documents
    from p0.replays.protocol import parse_replay_payload
    from tests.unit.replay_fixtures import golden_replay_payload

    document = parse_replay_payload(golden_replay_payload("buffer-reuse"))
    perspective = compile_documents((document,), chunksize=0).games[0].perspectives[0]
    assert len(perspective.snapshots) >= 2
    builder = ObservationBuilder(default_runtime_resources())
    output = StructuredObservation.empty_batch(1)[0]
    for snapshot in perspective.snapshots:
        output.token_type_ids.fill_(99)
        output.categorical.fill_(99)
        output.numerical.fill_(99.0)
        output.spatial_cat.fill_(99)
        output.spatial_num.fill_(99.0)
        builder.build_into(snapshot.view, output)
        output.validate(batch_rank=0)
        output.validate_overflow_contract()
        assert all(torch.isfinite(tensor).all() for tensor in output.tensors())
        assert output.spatial_cat.shape == (SPATIAL_SLOT_COUNT, SPATIAL_CATEGORICAL_WIDTH)
        assert output.spatial_num.shape == (SPATIAL_SLOT_COUNT, SPATIAL_NUMERICAL_WIDTH)
        assert not torch.any(output.spatial_cat == 99)
        assert not torch.any(output.spatial_num == 99.0)
        assert output.token_type_ids[0].item() == int(TokenType.POKEMON)
        assert output.side_ids[0].item() == int(SideId.ALLY)
        assert tuple(output.slot_ids[:6].tolist()) == (1, 2, 3, 4, 5, 6)
        assert tuple(output.slot_ids[6:12].tolist()) == (1, 2, 3, 4, 5, 6)
        assert output.numerical[12, NUM_IDX_TEAM_PREVIEW].item() == float(snapshot.view.teampreview)
        assert output.numerical[12, 3].item() == pytest.approx(snapshot.view.turn / 24.0)


def test_empty_slot_and_fainted_pokemon_zero_padding() -> None:
    """Verify empty/unrevealed bench slots write empty slot condition (1.0) and zero out stat/move numerical columns."""
    mon = make_pokemon_view(species="pikachu")
    battle = make_battle_view(active_pokemon=[mon, None], team=[mon])

    obs = _OBSERVATION_BUILDER.build(battle)

    # Empty ally bench slot (slot index 2): condition 0 (empty) is encoded at numerical[1]
    assert obs.token_type_ids[2] == TokenType.POKEMON
    assert obs.categorical[2, 0].item() == 0  # No species
    assert obs.numerical[2, 1].item() == 1.0  # Slot condition empty
    assert not obs.numerical[2, 5:].any()  # All Pokemon stats/HP/boosts are zero

    # Opponent slots (slots 6..11) all empty
    for slot_idx in range(6, 12):
        assert obs.categorical[slot_idx, 0].item() == 0
        assert obs.numerical[slot_idx, 1].item() == 1.0
        assert not obs.numerical[slot_idx, 5:].any()


def test_field_and_weather_turn_fraction_scaling() -> None:
    """Verify turn numbers are scaled by 1/24 on the global field token and all output tensors contain finite numbers."""
    mon = make_pokemon_view(species="charizard")
    battle = make_battle_view(
        active_pokemon=[mon, None],
        team=[mon],
        weather={Weather.SUNNYDAY: 3},
        fields={Field.ELECTRIC_TERRAIN: 5},
        turn=12,
    )
    obs = _OBSERVATION_BUILDER.build(battle)

    # Global field token turn count is normalized by 24 (12 / 24 = 0.5)
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, 3].item() == pytest.approx(12.0 / 24.0)
    assert all(torch.isfinite(t).all() for t in obs.tensors())
