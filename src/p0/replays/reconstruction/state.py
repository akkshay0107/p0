"""Owned battle state and immutable snapshots for replay reconstruction v2."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from p0.replays.identity import ReplayMemberId, ReplaySide, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.classification import EventClassification
from p0.replays.reconstruction.diagnostics import (
    ReplayEventDiagnostic,
    ReplayEventParseError,
)
from p0.replays.reconstruction.events import EffectReference
from p0.replays.reconstruction.resolution import (
    ResolvedPokemonRefArgument,
    ResolvedProtocolEvent,
    resolve_replay_events,
)
from p0.replays.schema import OTSData, OTSMember

_BOOST_NAMES = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_STATUS_NAMES = frozenset({"brn", "frz", "par", "psn", "slp", "tox"})
_STACKING_SIDE_CONDITIONS = frozenset({"spikes", "toxicspikes"})
# These four are emitted through -singleturn but are side conditions in the
# simulator. Other ally-side conditions remain governed by the generated dex.
_SIDE_GUARDS = frozenset({"craftyshield", "matblock", "quickguard", "wideguard"})
_BREAKING_PROTECT_MOVES = frozenset(
    {"feint", "hyperspacefury", "hyperspacehole", "phantomforce", "shadowforce"}
)
_BREAKABLE_MEMBER_PROTECT_EFFECTS = frozenset(
    {
        "banefulbunker",
        "burningbulwark",
        "kingsshield",
        "obstruct",
        "protect",
        "silktrap",
        "spikyshield",
    }
)
_PROTECT_COUNTER_MOVES = frozenset(
    {
        "banefulbunker",
        "burningbulwark",
        "detect",
        "endure",
        "kingsshield",
        "matblock",
        "maxguard",
        "obstruct",
        "protect",
        "quickguard",
        "silktrap",
        "spikyshield",
        "wideguard",
    }
)
_IMPLICIT_ACTION_MOVES = frozenset({"recharge", "struggle"})
_SLOT_CONDITION_MOVES = frozenset({"healingwish", "wish"})
_DELAYED_MOVE_IDS = frozenset({"doomdesire", "futuresight"})
_DYNAMIC_EFFECTS: dict[str, dict[str, int | str]] = {
    "perishsong": {f"perish{count}": count for count in range(4)},
    "stockpile": {f"stockpile{layer}": layer for layer in range(1, 4)},
    "protosynthesis": {f"protosynthesis{stat}": stat for stat in _BOOST_NAMES[:5]},
    "quarkdrive": {f"quarkdrive{stat}": stat for stat in _BOOST_NAMES[:5]},
    "supremeoverlord": {f"fallen{count}": count for count in range(1, 6)},
}
_DYNAMIC_PREFIXES = (
    ("perish", "perishsong"),
    ("stockpile", "stockpile"),
    ("protosynthesis", "protosynthesis"),
    ("quarkdrive", "quarkdrive"),
    ("fallen", "supremeoverlord"),
)
_DYNAMIC_CANONICAL_EFFECTS = frozenset(_DYNAMIC_EFFECTS)
_VARIANT_EFFECTS = frozenset(_DYNAMIC_CANONICAL_EFFECTS - {"perishsong"})
_TRANSIENT_ACTION_TAGS = frozenset(
    {
        "-anim",
        "-block",
        "-combine",
        "-eat",
        "-fieldactivate",
        "-hitcount",
        "-immune",
        "-miss",
        "-nothing",
        "-notarget",
        "-waiting",
    }
)
_INITIALIZATION_TAGS = frozenset({"player", "clearpoke", "poke", "showteam", "teamsize", "start"})


@dataclass(frozen=True, slots=True)
class AbilityState:
    """Original, form-native, and temporary ability components."""

    base: str
    forme: str | None = None
    temporary: str | None = None

    def __post_init__(self) -> None:
        if not self.base:
            raise ValueError("AbilityState.base must not be empty")
        if self.forme == "" or self.temporary == "":
            raise ValueError("Optional ability components must be absent or nonempty")

    @property
    def current(self) -> str:
        """Return the effective named ability without modelling suppression."""
        return self.temporary or self.forme or self.base


@dataclass(frozen=True, slots=True)
class DynamicEffectVariant:
    """Validated wire suffix retained alongside a canonical effect ID."""

    canonical_id: str
    wire_id: str
    value: int | str

    def __post_init__(self) -> None:
        if self.canonical_id not in _DYNAMIC_EFFECTS:
            raise ValueError(f"Unknown dynamic effect {self.canonical_id!r}")
        if _DYNAMIC_EFFECTS[self.canonical_id].get(self.wire_id) != self.value:
            raise ValueError("DynamicEffectVariant does not match its canonical effect")


def normalize_dynamic_effect(effect: str) -> DynamicEffectVariant | None:
    """Return validated dynamic-effect metadata, or None for a static effect ID."""
    normalized = normalize_showdown_id(effect)
    for prefix, canonical_id in _DYNAMIC_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        if normalized == canonical_id:
            return None
        try:
            value = _DYNAMIC_EFFECTS[canonical_id][normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported dynamic effect variant {effect!r}") from exc
        return DynamicEffectVariant(canonical_id, normalized, value)
    return None


@dataclass(frozen=True, slots=True)
class DelayedMoveState:
    """A validated Future Sight or Doom Desire condition attached to a side slot."""

    source_member_id: ReplayMemberId
    target_side: ReplaySide
    target_slot: int
    move_id: str
    move_name: str
    move_type: str
    category: str
    base_power: int
    scheduled_turn: int
    announced: bool

    def __post_init__(self) -> None:
        if not isinstance(self.target_side, ReplaySide):
            raise ValueError("DelayedMoveState has an invalid target side")
        if not 0 <= self.target_slot < 2:
            raise ValueError("DelayedMoveState.target_slot must be a doubles slot")
        if self.move_id not in _DELAYED_MOVE_IDS:
            raise ValueError("DelayedMoveState.move_id must be Doom Desire or Future Sight")
        if not self.move_name or not self.move_type or not self.category:
            raise ValueError("DelayedMoveState requires move metadata")
        if self.base_power < 0 or self.scheduled_turn < 0:
            raise ValueError("DelayedMoveState numeric metadata must be nonnegative")


@dataclass(frozen=True, slots=True)
class SlotConditionState:
    """A validated Wish or Healing Wish condition attached to a side slot."""

    source_member_id: ReplayMemberId
    target_side: ReplaySide
    target_slot: int
    move_id: str
    created_turn: int
    expiration_turn: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.target_side, ReplaySide):
            raise ValueError("SlotConditionState has an invalid target side")
        if not 0 <= self.target_slot < 2:
            raise ValueError("SlotConditionState.target_slot must be a doubles slot")
        if self.move_id not in _SLOT_CONDITION_MOVES:
            raise ValueError("SlotConditionState.move_id must be Wish or Healing Wish")
        if self.created_turn < 0:
            raise ValueError("SlotConditionState.created_turn must be nonnegative")
        if self.move_id == "wish":
            if self.expiration_turn is None or self.expiration_turn <= self.created_turn:
                raise ValueError("Wish must have a later expiration turn")
        elif self.expiration_turn is not None:
            raise ValueError("Healing Wish must persist until it resolves")


@dataclass(frozen=True, slots=True)
class MoveState:
    """Immutable move-slot state used by base members and Transform overlays."""

    move_id: str
    name: str
    move_type: str
    category: str
    target: str
    current_pp: int
    max_pp: int

    def __post_init__(self) -> None:
        if not self.move_id or not self.name:
            raise ValueError("MoveState requires a nonempty id and name")
        if not 0 <= self.current_pp <= self.max_pp:
            raise ValueError("MoveState.current_pp must be within its PP range")
        if self.max_pp <= 0:
            raise ValueError("MoveState.max_pp must be positive")


@dataclass(frozen=True, slots=True)
class TransformSnapshot:
    """Values copied atomically from a Transform target."""

    source_member_id: ReplayMemberId
    species: str
    types: tuple[str, ...]
    weight: float
    non_hp_base_stats: tuple[tuple[str, int], ...]
    ability: str
    boosts: tuple[tuple[str, int], ...]
    moves: tuple[MoveState, ...]

    def __post_init__(self) -> None:
        if not self.species or not self.ability:
            raise ValueError("TransformSnapshot requires a species and ability")
        if not 1 <= len(self.types) <= 3:
            raise ValueError("TransformSnapshot.types must contain one to three types")
        if tuple(name for name, _ in self.non_hp_base_stats) != _BOOST_NAMES[:5]:
            raise ValueError("TransformSnapshot.non_hp_base_stats must use canonical stat order")
        if tuple(name for name, _ in self.boosts) != _BOOST_NAMES:
            raise ValueError("TransformSnapshot.boosts must use canonical stat order")


@dataclass(frozen=True, slots=True)
class ReplayPokemonState:
    """One stable roster member at a specific public replay cursor."""

    member_id: ReplayMemberId
    nickname: str
    original_species: str
    current_form: str
    displayed_species: str
    nature: str
    level: int
    hp_fraction: float | None
    status: str | None
    item: str | None
    ability: AbilityState
    base_types: tuple[str, ...]
    current_types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    moves: tuple[MoveState, ...]
    boosts: tuple[tuple[str, int], ...]
    effects: tuple[tuple[str, int], ...]
    perish_count: int | None
    tera_type: str | None
    terastallized: bool
    transform: TransformSnapshot | None
    revealed: bool
    selected: bool | None
    fainted: bool
    active_turns: int
    status_counter: int
    protect_counter: int
    preparing: str | None
    last_move: str | None
    effect_variants: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.hp_fraction is not None and not 0.0 <= self.hp_fraction <= 1.0:
            raise ValueError("ReplayPokemonState.hp_fraction must be in [0, 1]")
        if self.fainted and self.hp_fraction != 0.0:
            raise ValueError("A fainted ReplayPokemonState must have zero HP")
        if self.hp_fraction == 0.0 and not self.fainted:
            raise ValueError("ReplayPokemonState fainted and HP state disagree")
        if tuple(name for name, _ in self.boosts) != _BOOST_NAMES:
            raise ValueError("ReplayPokemonState.boosts must use canonical stat order")
        variant_effects = dict(self.effect_variants)
        variant_effect_names = tuple(variant_effects)
        if variant_effect_names != tuple(sorted(variant_effect_names)):
            raise ValueError("ReplayPokemonState.effect_variants must be sorted")
        if len(self.effect_variants) != len(variant_effects):
            raise ValueError("ReplayPokemonState.effect_variants must be unique")
        effects = dict(self.effects)
        for effect, wire_id in variant_effects.items():
            if effect not in _VARIANT_EFFECTS or effect not in effects:
                raise ValueError("Effect variant must belong to an active canonical effect")
            try:
                variant = normalize_dynamic_effect(wire_id)
            except ValueError as exc:
                raise ValueError("ReplayPokemonState has an invalid effect variant") from exc
            if variant is None or variant.canonical_id != effect:
                raise ValueError("ReplayPokemonState effect variant has the wrong canonical ID")
        if self.perish_count is not None and not 0 <= self.perish_count <= 3:
            raise ValueError("ReplayPokemonState.perish_count must be in [0, 3]")
        if self.perish_count is not None and not any(
            effect == "perishsong" for effect, _ in self.effects
        ):
            raise ValueError("A Perish count requires the canonical Perish Song effect")


@dataclass(frozen=True, slots=True)
class ReplaySideState:
    """Immutable side state and active-slot bindings."""

    side: ReplaySide
    active: tuple[ReplayMemberId | None, ReplayMemberId | None]
    conditions: tuple[tuple[str, int], ...]
    used_mega: bool
    used_z_move: bool

    def __post_init__(self) -> None:
        if any(member is not None and member.side is not self.side for member in self.active):
            raise ValueError("Active members must belong to ReplaySideState.side")


@dataclass(frozen=True, slots=True)
class ReplayBattleState:
    """Deeply immutable public battle state after one protocol line."""

    replay_id: str
    line_index: int
    turn: int
    members: tuple[ReplayPokemonState, ...]
    sides: tuple[ReplaySideState, ReplaySideState]
    weather: tuple[tuple[str, int], ...]
    fields: tuple[tuple[str, int], ...]
    delayed_moves: tuple[DelayedMoveState, ...] = ()
    slot_conditions: tuple[SlotConditionState, ...] = ()
    team_sizes: tuple[int, int] = (4, 4)

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ReplayBattleState.replay_id must not be empty")
        if self.line_index < 0 or self.turn < 0:
            raise ValueError("ReplayBattleState indices must be nonnegative")
        if tuple(side.side for side in self.sides) != (ReplaySide.P1, ReplaySide.P2):
            raise ValueError("ReplayBattleState sides must be ordered as p1 and p2")
        if len(self.team_sizes) != 2 or any(
            type(size) is not int or not 1 <= size <= 6 for size in self.team_sizes
        ):
            raise ValueError("ReplayBattleState.team_sizes must contain two values in [1, 6]")
        delayed_keys = tuple(
            (delayed.target_side.side_index, delayed.target_slot) for delayed in self.delayed_moves
        )
        if delayed_keys != tuple(sorted(delayed_keys)) or len(delayed_keys) != len(
            set(delayed_keys)
        ):
            raise ValueError("ReplayBattleState delayed moves must have unique sorted slots")
        slot_keys = tuple(
            (condition.target_side.side_index, condition.target_slot, condition.move_id)
            for condition in self.slot_conditions
        )
        if slot_keys != tuple(sorted(slot_keys)) or len(slot_keys) != len(set(slot_keys)):
            raise ValueError("ReplayBattleState slot conditions must be uniquely sorted")

    def member(self, member_id: ReplayMemberId) -> ReplayPokemonState:
        """Return one member from its stable side and roster index."""
        offset = member_id.side.side_index * 6 + member_id.roster_index
        try:
            member = self.members[offset]
        except IndexError as exc:
            raise KeyError(member_id) from exc
        if member.member_id != member_id:
            raise KeyError(member_id)
        return member


@dataclass(frozen=True, slots=True)
class ReconstructedReplayState:
    """All line-indexed snapshots or a whole-replay rejection."""

    replay_id: str
    snapshots: tuple[ReplayBattleState, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ReconstructedReplayState.replay_id must not be empty")
        if self.snapshots and self.diagnostics:
            raise ValueError("Rejected state reconstruction cannot retain snapshots")
        if any(snapshot.replay_id != self.replay_id for snapshot in self.snapshots):
            raise ValueError("State reconstruction cannot contain another replay")

    def require_accepted(self) -> tuple[ReplayBattleState, ...]:
        """Return snapshots or raise the structured whole-replay rejection."""
        if self.diagnostics:
            raise ReplayEventParseError(self.diagnostics)
        return self.snapshots


@dataclass(frozen=True, slots=True)
class _SpeciesData:
    species: str
    base_species: str
    types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    native_ability: str | None


@dataclass(slots=True)
class _MutableMove:
    move_id: str
    name: str
    move_type: str
    category: str
    target: str
    current_pp: int
    max_pp: int

    def snapshot(self) -> MoveState:
        return MoveState(
            self.move_id,
            self.name,
            self.move_type,
            self.category,
            self.target,
            self.current_pp,
            self.max_pp,
        )


@dataclass(slots=True)
class _MutablePokemon:
    member_id: ReplayMemberId
    nickname: str
    original_species: str
    current_form: str
    displayed_species: str
    nature: str
    level: int
    hp_fraction: float | None
    status: str | None
    item: str | None
    ability: AbilityState
    base_types: tuple[str, ...]
    current_types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    moves: dict[str, _MutableMove]
    mimic_move: _MutableMove | None = None
    boosts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_BOOST_NAMES, 0))
    effects: dict[str, int] = field(default_factory=dict)
    effect_variants: dict[str, str] = field(default_factory=dict)
    perish_count: int | None = None
    single_turn_effects: set[str] = field(default_factory=set)
    single_move_effects: set[str] = field(default_factory=set)
    tera_type: str | None = None
    terastallized: bool = False
    transform: TransformSnapshot | None = None
    revealed: bool = False
    selected: bool | None = None
    fainted: bool = False
    active_turns: int = 0
    status_counter: int = 0
    protect_counter: int = 0
    preparing: str | None = None
    last_move: str | None = None

    def move_snapshots(self) -> tuple[MoveState, ...]:
        """Return effective move slots without exposing mutable move objects."""
        if self.transform is not None:
            return self.transform.moves
        return tuple(
            self.mimic_move.snapshot()
            if move_id == "mimic" and self.mimic_move is not None
            else move.snapshot()
            for move_id, move in self.moves.items()
        )

    def snapshot(self) -> ReplayPokemonState:
        return ReplayPokemonState(
            member_id=self.member_id,
            nickname=self.nickname,
            original_species=self.original_species,
            current_form=self.current_form,
            displayed_species=self.displayed_species,
            nature=self.nature,
            level=self.level,
            hp_fraction=self.hp_fraction,
            status=self.status,
            item=self.item,
            ability=self.ability,
            base_types=self.base_types,
            current_types=self.current_types,
            base_stats=self.base_stats,
            weight=self.weight,
            moves=self.move_snapshots(),
            boosts=tuple((name, self.boosts[name]) for name in _BOOST_NAMES),
            effects=tuple(sorted(self.effects.items())),
            effect_variants=tuple(sorted(self.effect_variants.items())),
            perish_count=self.perish_count,
            tera_type=self.tera_type,
            terastallized=self.terastallized,
            transform=self.transform,
            revealed=self.revealed,
            selected=self.selected,
            fainted=self.fainted,
            active_turns=self.active_turns,
            status_counter=self.status_counter,
            protect_counter=self.protect_counter,
            preparing=self.preparing,
            last_move=self.last_move,
        )


class _StateTransitionError(ValueError):
    pass


class _StateReducer:
    """Single owner of mutable replay state and all supported transitions."""

    def __init__(
        self,
        replay_id: str,
        ots: tuple[OTSData, OTSData],
        dex: Mapping[str, Any],
    ) -> None:
        self.replay_id = replay_id
        self.turn = 0
        self._species = _species_index(dex)
        self._moves = _move_index(dex)
        self._legal_effects = _legal_effect_index(dex)
        self.members = {
            member.member_id: self._member_from_ots(member)
            for sheet in ots
            for member in sheet.members
        }
        self.active: dict[tuple[ReplaySide, int], ReplayMemberId] = {}
        self.team_sizes = {sheet.side: min(4, len(sheet.members)) for sheet in ots}
        self.side_conditions = {ReplaySide.P1: {}, ReplaySide.P2: {}}
        self.side_single_turn_effects = {ReplaySide.P1: set(), ReplaySide.P2: set()}
        self.used_mega = {ReplaySide.P1: False, ReplaySide.P2: False}
        self.used_z_move = {ReplaySide.P1: False, ReplaySide.P2: False}
        self.weather: dict[str, int] = {}
        self.fields: dict[str, int] = {}
        self.delayed_moves: dict[tuple[ReplaySide, int], DelayedMoveState] = {}
        self._ignored_delayed_starts: set[tuple[ReplayMemberId, str]] = set()
        self.slot_conditions: dict[tuple[ReplaySide, int, str], SlotConditionState] = {}
        self._handlers: dict[str, Callable[[ResolvedProtocolEvent], None]] = {
            "switch": self._handle_switch,
            "drag": self._handle_drag,
            "replace": self._handle_replace,
            "swap": self._handle_swap,
            "faint": self._handle_faint,
            "move": self._handle_move,
            "cant": self._handle_cant,
            "-damage": self._handle_minus_damage,
            "-heal": self._handle_minus_heal,
            "-sethp": self._handle_minus_sethp,
            "-status": self._handle_minus_status,
            "-curestatus": self._handle_minus_curestatus,
            "-cureteam": self._handle_minus_cureteam,
            "-boost": self._handle_minus_boost,
            "-unboost": self._handle_minus_unboost,
            "-setboost": self._handle_minus_setboost,
            "-clearboost": self._handle_minus_clearboost,
            "-clearallboost": self._handle_minus_clearallboost,
            "-clearpositiveboost": self._handle_minus_clearpositiveboost,
            "-clearnegativeboost": self._handle_minus_clearnegativeboost,
            "-invertboost": self._handle_minus_invertboost,
            "-copyboost": self._handle_minus_copyboost,
            "-swapboost": self._handle_minus_swapboost,
            "-item": self._handle_minus_item,
            "-enditem": self._handle_minus_enditem,
            "-ability": self._handle_minus_ability,
            "-endability": self._handle_minus_endability,
            "detailschange": self._handle_detailschange,
            "-formechange": self._handle_minus_formechange,
            "-mega": self._handle_minus_mega,
            "-primal": self._handle_minus_primal,
            "-burst": self._handle_minus_burst,
            "-zpower": self._handle_minus_zpower,
            "-zbroken": self._handle_minus_zbroken,
            "-terastallize": self._handle_minus_terastallize,
            "-transform": self._handle_minus_transform,
            "-typechange": self._handle_minus_typechange,
            "-typeadd": self._handle_minus_typeadd,
            "-start": self._handle_minus_start,
            "-end": self._handle_minus_end,
            "-singleturn": self._handle_minus_singleturn,
            "-singlemove": self._handle_minus_singlemove,
            "-activate": self._handle_minus_activate,
            "-prepare": self._handle_minus_prepare,
            "-fail": self._handle_minus_fail,
            "-mustrecharge": self._handle_minus_mustrecharge,
            "-weather": self._handle_minus_weather,
            "-fieldstart": self._handle_minus_fieldstart,
            "-fieldend": self._handle_minus_fieldend,
            "-sidestart": self._handle_minus_sidestart,
            "-sideend": self._handle_minus_sideend,
            "-swapsideconditions": self._handle_minus_swapsideconditions,
        }

    def _member_from_ots(self, member: OTSMember) -> _MutablePokemon:
        species = self._species_data(member.species)
        if not member.ability:
            raise _StateTransitionError(
                f"OTS member {member.member_id!r} has no recoverable base ability"
            )
        moves = {
            move.move_id: move for move in (self._move_from_name(name) for name in member.moves)
        }
        return _MutablePokemon(
            member_id=member.member_id,
            nickname=member.nickname,
            original_species=member.species,
            current_form=member.species,
            displayed_species=member.species,
            nature=member.nature,
            level=member.level,
            hp_fraction=None,
            status=None,
            item=member.item or None,
            ability=AbilityState(member.ability),
            base_types=species.types,
            current_types=species.types,
            base_stats=species.base_stats,
            weight=species.weight,
            moves=moves,
        )

    def _species_data(self, name: str) -> _SpeciesData:
        try:
            return self._species[normalize_showdown_id(name)]
        except KeyError as exc:
            raise _StateTransitionError(f"species {name!r} is absent from the pinned dex") from exc

    def _move_from_name(self, name: str, *, transform: bool = False) -> _MutableMove:
        move_id = normalize_showdown_id(name)
        try:
            data = self._moves[move_id]
        except KeyError as exc:
            raise _StateTransitionError(f"move {name!r} is absent from the pinned dex") from exc
        base_pp = int(data.get("pp", 0))
        if base_pp <= 0:
            raise _StateTransitionError(f"move {name!r} has invalid PP data")
        max_pp = min(5, base_pp) if transform else _maximum_pp(data, base_pp)
        return _MutableMove(
            move_id=move_id,
            name=str(data.get("name", name)),
            move_type=str(data.get("type", "unknown")),
            category=str(data.get("category", "unknown")),
            target=str(data.get("target", "normal")),
            current_pp=max_pp,
            max_pp=max_pp,
        )

    def apply(self, resolved: ResolvedProtocolEvent) -> None:
        event = resolved.event
        classification = event.classification
        if classification.rejects_replay:
            raise _StateTransitionError(event.rejection_reason)
        if event.tag == "-hint":
            self._handle_hint(resolved)
            return
        if classification is EventClassification.NO_STATE_CHANGE:
            return
        if classification is EventClassification.BOUNDARY_SIGNAL:
            self._apply_boundary(event.tag, event.arguments)
            return
        if event.tag == "teamsize":
            self._set_team_size(event.arguments)
            return
        if event.tag in _INITIALIZATION_TAGS:
            return

        handler = self._handlers.get(event.tag)
        if handler is None:
            if event.tag in _TRANSIENT_ACTION_TAGS:
                self._apply_provenance(resolved)
                return
            raise _StateTransitionError(f"state transition {event.tag!r} is not implemented")
        handler(resolved)
        self._apply_provenance(resolved)

    def _apply_boundary(self, tag: str, arguments: tuple[str, ...]) -> None:
        if tag == "turn":
            turn = int(arguments[0])
            if turn <= self.turn:
                raise _StateTransitionError(
                    f"turn number {turn} does not advance current turn {self.turn}"
                )
            self._ignored_delayed_starts.clear()
            self._expire_member_single_turn_effects()
            self._expire_side_single_turn_effects()
            self.turn = turn
            for member_id in self.active.values():
                member = self.members[member_id]
                member.active_turns += 1
                if member.status == "tox":
                    member.status_counter += 1
        elif tag == "upkeep":
            self._expire_member_single_turn_effects()
            self._expire_side_single_turn_effects()
            self._expire_slot_conditions()
        elif tag in {"", "teampreview", "win", "tie", "forfeit"}:
            return
        else:
            raise _StateTransitionError(f"boundary signal {tag!r} is not implemented")

    def _set_team_size(self, arguments: tuple[str, ...]) -> None:
        if len(arguments) < 2:
            raise _StateTransitionError("teamsize requires a side and a size")
        side = ReplaySide(arguments[0])
        size = int(arguments[1])
        if not 1 <= size <= len(self.members) // 2:
            raise _StateTransitionError(f"invalid team size {size} for {side.value}")
        self.team_sizes[side] = size

    def _expire_member_single_turn_effects(self) -> None:
        for member_id in self.active.values():
            member = self.members[member_id]
            for effect in member.single_turn_effects:
                member.effects.pop(effect, None)
            member.single_turn_effects.clear()

    def _expire_side_single_turn_effects(self) -> None:
        for side, effects in self.side_single_turn_effects.items():
            for effect in effects:
                self.side_conditions[side].pop(effect, None)
            effects.clear()

    def _handle_switch(self, resolved: ResolvedProtocolEvent) -> None:
        self._switch(resolved)

    def _handle_drag(self, resolved: ResolvedProtocolEvent) -> None:
        self._switch(resolved)

    def _switch(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        slot = _required_slot(reference)
        key = (member_id.side, slot)
        outgoing_id = self.active.get(key)
        if outgoing_id is not None and outgoing_id != member_id:
            self._clear_switch_state(self.members[outgoing_id])

        member = self.members[member_id]
        details_species = _details_species(resolved.event.arguments[1])
        details_data = self._species_data(details_species)
        original_data = self._species_data(member.original_species)
        disguised = normalize_showdown_id(details_data.base_species) != normalize_showdown_id(
            original_data.base_species
        )
        if disguised:
            if normalize_showdown_id(member.ability.current) != "illusion":
                raise _StateTransitionError(
                    "disguised switch resolved to a member without Illusion"
                )
            member.displayed_species = details_species
            member.effects["illusion"] = self.turn
        else:
            self._set_form(member, details_species)
            member.effects.pop("illusion", None)
        hp_fraction, status = _parse_hp_status(resolved.event.arguments[2])
        member.hp_fraction = hp_fraction
        member.status = status
        if status != "slp":
            member.status_counter = 0
        member.fainted = hp_fraction == 0.0
        member.revealed = not disguised
        member.selected = True
        member.active_turns = 0
        self.active[key] = member_id

    def _handle_replace(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        revealed_species = _details_species(resolved.event.arguments[1])
        revealed_data = self._species_data(revealed_species)
        original_data = self._species_data(member.original_species)
        if normalize_showdown_id(revealed_data.base_species) != normalize_showdown_id(
            original_data.base_species
        ):
            raise _StateTransitionError("Illusion reveal does not match the active member")
        if "illusion" not in member.effects:
            raise _StateTransitionError("replace requires an active Illusion history")

        member.displayed_species = revealed_species
        member.revealed = True
        member.effects.pop("illusion")

    def _handle_swap(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        source_slot = _required_slot(reference)
        target_slot = int(resolved.event.arguments[1])
        if not 0 <= target_slot < 2:
            raise _StateTransitionError("swap target must be a doubles active slot")
        source_key = (member_id.side, source_slot)
        target_key = (member_id.side, target_slot)
        if self.active.get(source_key) != member_id:
            raise _StateTransitionError("swap source does not match reducer active state")
        target = self.active.get(target_key)
        self.active[target_key] = member_id
        if target is None:
            self.active.pop(source_key)
        else:
            self.active[source_key] = target

    def _handle_faint(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        slot = _required_slot(reference)
        key = (member_id.side, slot)
        if self.active.get(key) != member_id:
            raise _StateTransitionError("faint member does not match reducer active state")
        member = self.members[member_id]
        member.hp_fraction = 0.0
        member.fainted = True
        member.status = None
        self._clear_switch_state(member)
        self.active.pop(key)

    def _handle_move(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        move_id = normalize_showdown_id(resolved.event.arguments[1])
        move = next((item for item in member.move_snapshots() if item.move_id == move_id), None)
        if move is None and move_id not in _IMPLICIT_ACTION_MOVES:
            raise _StateTransitionError(
                f"move {resolved.event.arguments[1]!r} is not in the member's effective move slots"
            )
        if move_id in _DELAYED_MOVE_IDS:
            target = resolved.event.arguments[2] if len(resolved.event.arguments) > 2 else ""
            if target:
                target_ref = _resolved_reference(resolved, 2)
                if (
                    target_ref.member_id is not None
                    and target_ref.pokemon_ref.active_slot is not None
                ):
                    self._ignored_delayed_starts.discard((member.member_id, move_id))
                    self._schedule_delayed_move(resolved, member, move_id)
                else:
                    self._ignored_delayed_starts.add((member.member_id, move_id))
        elif move_id in _SLOT_CONDITION_MOVES:
            self._schedule_slot_condition(resolved, member, move_id)
        elif move_id == "recharge":
            member.effects.pop("mustrecharge", None)
        elif move_id not in _PROTECT_COUNTER_MOVES:
            member.protect_counter = 0
        if resolved.event.cause is None and move_id not in _IMPLICIT_ACTION_MOVES:
            self._decrement_move(member, move_id)
        for effect in member.single_move_effects:
            member.effects.pop(effect, None)
        member.single_move_effects.clear()
        member.last_move = move_id
        member.preparing = None

    def _schedule_delayed_move(
        self,
        resolved: ResolvedProtocolEvent,
        source: _MutablePokemon,
        move_id: str,
    ) -> None:
        target_ref = _resolved_reference(resolved, 2)
        target_member_id = _required_member(target_ref)
        target_slot = _required_slot(target_ref)
        target_key = (target_member_id.side, target_slot)
        if self.active.get(target_key) != target_member_id:
            raise _StateTransitionError("delayed move target is not active in its referenced slot")
        if target_key in self.delayed_moves:
            raise _StateTransitionError("delayed move target slot already has a pending effect")

        data = self._moves[move_id]
        try:
            move_name = str(data["name"])
            move_type = str(data["type"])
            category = str(data["category"])
            base_power = int(data["basePower"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _StateTransitionError(
                f"delayed move {move_id!r} has incomplete damage metadata"
            ) from exc
        self.delayed_moves[target_key] = DelayedMoveState(
            source_member_id=source.member_id,
            target_side=target_member_id.side,
            target_slot=target_slot,
            move_id=move_id,
            move_name=move_name,
            move_type=move_type,
            category=category,
            base_power=base_power,
            # Showdown resolves delayed moves at the end of the second turn after use.
            scheduled_turn=self.turn + 2,
            announced=False,
        )

    def _schedule_slot_condition(
        self,
        resolved: ResolvedProtocolEvent,
        source: _MutablePokemon,
        move_id: str,
    ) -> None:
        source_slot = _required_slot(_resolved_reference(resolved, 0))
        key = (source.member_id.side, source_slot, move_id)
        if key in self.slot_conditions:
            return
        self.slot_conditions[key] = SlotConditionState(
            source_member_id=source.member_id,
            target_side=source.member_id.side,
            target_slot=source_slot,
            move_id=move_id,
            created_turn=self.turn,
            expiration_turn=self.turn + 1 if move_id == "wish" else None,
        )

    def _expire_slot_conditions(self) -> None:
        for key, condition in tuple(self.slot_conditions.items()):
            if (
                condition.move_id == "wish"
                and condition.expiration_turn is not None
                and condition.expiration_turn <= self.turn
            ):
                del self.slot_conditions[key]

    def _discard_failed_slot_condition(self, member: _MutablePokemon) -> None:
        if member.last_move not in _SLOT_CONDITION_MOVES:
            return
        for key, condition in tuple(self.slot_conditions.items()):
            if (
                condition.source_member_id == member.member_id
                and condition.move_id == member.last_move
                and condition.created_turn == self.turn
            ):
                del self.slot_conditions[key]

    def _handle_delayed_start(
        self,
        resolved: ResolvedProtocolEvent,
        move_id: str,
    ) -> None:
        source = self._member_for(resolved, 0)
        matches = tuple(
            (key, delayed)
            for key, delayed in self.delayed_moves.items()
            if delayed.source_member_id == source.member_id and delayed.move_id == move_id
        )
        pending = tuple((key, delayed) for key, delayed in matches if not delayed.announced)
        if (source.member_id, move_id) in self._ignored_delayed_starts:
            self._ignored_delayed_starts.remove((source.member_id, move_id))
            return
        if len(pending) != 1:
            if not pending and matches:
                raise _StateTransitionError(f"{move_id} start was emitted more than once")
            raise _StateTransitionError(
                f"{move_id} start has an ambiguous pending move from the referenced caster"
            )
        key, delayed = pending[0]
        self.delayed_moves[key] = replace(delayed, announced=True)

    def _handle_delayed_end(
        self,
        resolved: ResolvedProtocolEvent,
        move_id: str,
    ) -> None:
        target_ref = _resolved_reference(resolved, 0)
        target_member_id = _required_member(target_ref)
        target_slot = _required_slot(target_ref)
        target_key = (target_member_id.side, target_slot)
        if self.active.get(target_key) != target_member_id:
            raise _StateTransitionError("delayed move ended for a non-active target slot")
        try:
            delayed = self.delayed_moves[target_key]
        except KeyError as exc:
            raise _StateTransitionError(
                f"{move_id} ended without a pending target-slot condition"
            ) from exc
        if delayed.move_id != move_id:
            raise _StateTransitionError(
                f"delayed move end {move_id!r} does not match {delayed.move_id!r}"
            )
        if not delayed.announced:
            raise _StateTransitionError("delayed move ended before its start message")
        if self.turn > 0 and self.turn != delayed.scheduled_turn:
            timing = "before" if self.turn < delayed.scheduled_turn else "after"
            raise _StateTransitionError(f"delayed move ended {timing} its scheduled turn")
        del self.delayed_moves[target_key]

    def _handle_hint(self, resolved: ResolvedProtocolEvent) -> None:
        hint = resolved.event.arguments[0]
        marker = " did not hit because the target is "
        move_name, separator, outcome = hint.partition(marker)
        if not separator:
            return

        move_id = normalize_showdown_id(move_name)
        if move_id not in _DELAYED_MOVE_IDS:
            return

        outcome = outcome.casefold()
        if outcome == "the user.":
            keys = tuple(
                key
                for key, delayed in self.delayed_moves.items()
                if delayed.move_id == move_id
                and delayed.scheduled_turn == self.turn
                and self.active.get(key) == delayed.source_member_id
            )
        elif outcome == "fainted.":
            keys = tuple(
                key
                for key, delayed in self.delayed_moves.items()
                if delayed.move_id == move_id
                and delayed.scheduled_turn == self.turn
                and self.active.get(key) is None
            )
        else:
            return

        for key in keys:
            del self.delayed_moves[key]

    def _handle_cant(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        effect = _required_effect(resolved).normalized
        if effect == "recharge":
            member.effects.pop("mustrecharge", None)
        else:
            self._discard_failed_slot_condition(member)
        if member.status == "slp":
            member.status_counter += 1
        member.protect_counter = 0

    def _handle_minus_damage(self, resolved: ResolvedProtocolEvent) -> None:
        self._set_hp(resolved)

    def _handle_minus_heal(self, resolved: ResolvedProtocolEvent) -> None:
        cause = resolved.event.cause
        move_id = cause.normalized if cause is not None and cause.namespace == "move" else None
        condition_key = None
        if move_id in _SLOT_CONDITION_MOVES:
            target_reference = _resolved_reference(resolved, 0)
            target_member_id = _required_member(target_reference)
            target_slot = _required_slot(target_reference)
            condition_key = (target_member_id.side, target_slot, move_id)
            if condition_key not in self.slot_conditions:
                raise _StateTransitionError(
                    f"{move_id} healed a slot without a pending slot condition"
                )
        self._set_hp(resolved, clear_status=move_id == "healingwish")
        if condition_key is not None:
            del self.slot_conditions[condition_key]

    def _handle_minus_sethp(self, resolved: ResolvedProtocolEvent) -> None:
        arguments = resolved.event.arguments
        if len(arguments) % 2:
            raise _StateTransitionError("-sethp requires Pokémon/HP argument pairs")
        for argument_index in range(0, len(arguments), 2):
            member = self._member_for(resolved, argument_index)
            hp_fraction, status = _parse_hp_status(arguments[argument_index + 1])
            member.hp_fraction = hp_fraction
            member.fainted = hp_fraction == 0.0
            if status is not None:
                member.status = status

    def _set_hp(self, resolved: ResolvedProtocolEvent, *, clear_status: bool = False) -> None:
        member = self._member_for(resolved, 0)
        hp_fraction, status = _parse_hp_status(resolved.event.arguments[1])
        if clear_status and status is not None:
            raise _StateTransitionError("Healing Wish heal must not retain a status")
        member.hp_fraction = hp_fraction
        member.fainted = hp_fraction == 0.0
        if clear_status:
            member.status = None
            member.status_counter = 0
        elif hp_fraction > 0.0 and status is not None:
            member.status = status

    def _handle_minus_status(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        status = normalize_showdown_id(resolved.event.arguments[1])
        if status not in _STATUS_NAMES:
            raise _StateTransitionError(f"unsupported status {status!r}")
        member.status = status
        member.status_counter = 0

    def _handle_minus_curestatus(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        expected = normalize_showdown_id(resolved.event.arguments[1])
        if member.status is not None and member.status != expected:
            raise _StateTransitionError("curestatus does not match the current status")
        member.status = None
        member.status_counter = 0

    def _handle_minus_cureteam(self, resolved: ResolvedProtocolEvent) -> None:
        side = _resolved_reference(resolved, 0).pokemon_ref.side
        for member_id, member in self.members.items():
            if member_id.side is side:
                member.status = None
                member.status_counter = 0

    def _handle_minus_boost(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_boost(resolved, 1)

    def _handle_minus_unboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_boost(resolved, -1)

    def _handle_minus_setboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_boost(resolved, 0)

    def _change_boost(self, resolved: ResolvedProtocolEvent, direction: int) -> None:
        member = self._member_for(resolved, 0)
        stat = normalize_showdown_id(resolved.event.arguments[1])
        if stat not in _BOOST_NAMES:
            raise _StateTransitionError(f"unsupported boost stat {stat!r}")
        amount = int(resolved.event.arguments[2])
        value = amount if direction == 0 else member.boosts[stat] + direction * amount
        member.boosts[stat] = max(-6, min(6, value))

    def _handle_minus_clearboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._clear_boosts(self._member_for(resolved, 0), lambda _: True)

    def _handle_minus_clearallboost(self, resolved: ResolvedProtocolEvent) -> None:
        for member_id in self.active.values():
            self._clear_boosts(self.members[member_id], lambda _: True)

    def _handle_minus_clearpositiveboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._clear_boosts(self._member_for(resolved, 0), lambda value: value > 0)

    def _handle_minus_clearnegativeboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._clear_boosts(self._member_for(resolved, 0), lambda value: value < 0)

    def _handle_minus_invertboost(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.boosts = {name: -value for name, value in member.boosts.items()}

    def _handle_minus_copyboost(self, resolved: ResolvedProtocolEvent) -> None:
        source = self._member_for(resolved, 0)
        target = self._member_for(resolved, 1)
        target.boosts = dict(source.boosts)

    def _handle_minus_swapboost(self, resolved: ResolvedProtocolEvent) -> None:
        source = self._member_for(resolved, 0)
        target = self._member_for(resolved, 1)
        stats = _boost_argument(resolved.event.arguments[2:])
        for stat in stats:
            source.boosts[stat], target.boosts[stat] = target.boosts[stat], source.boosts[stat]

    @staticmethod
    def _clear_boosts(member: _MutablePokemon, predicate: Callable[[int], bool]) -> None:
        member.boosts = {
            name: 0 if predicate(value) else value for name, value in member.boosts.items()
        }

    def _handle_minus_item(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0).item = resolved.event.arguments[1]

    def _handle_minus_enditem(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        announced = normalize_showdown_id(resolved.event.arguments[1])
        if member.item is not None and normalize_showdown_id(member.item) != announced:
            raise _StateTransitionError("enditem does not match the member's current item")
        member.item = None

    def _handle_minus_ability(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        ability = resolved.event.arguments[1]
        current_ids = {
            normalize_showdown_id(value)
            for value in (member.ability.base, member.ability.forme, member.ability.temporary)
            if value is not None
        }
        temporary = member.ability.temporary
        if resolved.event.cause is not None or normalize_showdown_id(ability) not in current_ids:
            temporary = ability
        if temporary is not None:
            self._set_temporary_ability(member, temporary)

    def _handle_minus_endability(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.effects.setdefault("gastroacid", self.turn)

    def _handle_detailschange(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_form(resolved, resolved.event.arguments[1])

    def _handle_minus_formechange(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_form(resolved, resolved.event.arguments[1])

    def _change_form(self, resolved: ResolvedProtocolEvent, details: str) -> None:
        member = self._member_for(resolved, 0)
        form = _details_species(details)
        self._set_form(member, form)

    def _set_form(self, member: _MutablePokemon, form: str) -> None:
        data = self._species_data(form)
        member.current_form = form
        member.displayed_species = form
        if not member.terastallized:
            member.current_types = data.types
        member.base_stats = data.base_stats
        member.weight = data.weight
        forme_ability = (
            None
            if normalize_showdown_id(form) == normalize_showdown_id(member.original_species)
            else data.native_ability
        )
        member.ability = AbilityState(
            member.ability.base,
            forme_ability,
            member.ability.temporary,
        )

    def _handle_minus_mega(self, resolved: ResolvedProtocolEvent) -> None:
        self.used_mega[self._member_for(resolved, 0).member_id.side] = True

    def _handle_minus_primal(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0)

    def _handle_minus_burst(self, resolved: ResolvedProtocolEvent) -> None:
        self._change_form(resolved, resolved.event.arguments[1])

    def _handle_minus_zpower(self, resolved: ResolvedProtocolEvent) -> None:
        self.used_z_move[self._member_for(resolved, 0).member_id.side] = True

    def _handle_minus_zbroken(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0)

    def _handle_minus_terastallize(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        tera_type = resolved.event.arguments[1]
        if not tera_type:
            raise _StateTransitionError("terastallize requires a nonempty type")
        member.tera_type = tera_type
        member.terastallized = True
        member.current_types = (tera_type,)

    def _handle_minus_transform(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        target = self._transform_target(resolved, member)
        target_species = (
            target.transform.species if target.transform is not None else target.current_form
        )
        target_types = (
            target.transform.types if target.transform is not None else target.current_types
        )
        target_weight = target.transform.weight if target.transform is not None else target.weight
        target_base_stats = (
            target.transform.non_hp_base_stats
            if target.transform is not None
            else tuple((name, dict(target.base_stats)[name]) for name in _BOOST_NAMES[:5])
        )
        target_ability = self._current_ability(target)
        copied_moves = tuple(
            MoveState(
                move.move_id,
                move.name,
                move.move_type,
                move.category,
                move.target,
                min(5, move.max_pp),
                min(5, move.max_pp),
            )
            for move in target.move_snapshots()
        )
        member.transform = TransformSnapshot(
            source_member_id=target.member_id,
            species=target_species,
            types=target_types,
            weight=target_weight,
            non_hp_base_stats=target_base_stats,
            ability=target_ability,
            boosts=tuple((name, target.boosts[name]) for name in _BOOST_NAMES),
            moves=copied_moves,
        )
        member.boosts = dict(target.boosts)

    def _transform_target(
        self,
        resolved: ResolvedProtocolEvent,
        member: _MutablePokemon,
    ) -> _MutablePokemon:
        """Resolve a Transform target from a reference or the emitted species name."""
        target_reference = next(
            (reference for reference in resolved.pokemon_refs if reference.argument_index == 1),
            None,
        )
        if target_reference is not None:
            target_id = _required_member(target_reference)
            if target_id == member.member_id:
                raise _StateTransitionError("Transform cannot target the transforming member")
            return self.members[target_id]

        target_species = normalize_showdown_id(resolved.event.arguments[1])
        candidates = tuple(
            candidate
            for candidate in self.members.values()
            if candidate.member_id != member.member_id
            and candidate.member_id in self.active.values()
            and target_species
            in {
                normalize_showdown_id(candidate.current_form),
                normalize_showdown_id(candidate.displayed_species),
                normalize_showdown_id(
                    candidate.transform.species if candidate.transform is not None else ""
                ),
            }
        )
        if len(candidates) != 1:
            raise _StateTransitionError(
                f"Transform species {resolved.event.arguments[1]!r} resolved to "
                f"{len(candidates)} active targets"
            )
        return candidates[0]

    def _handle_minus_typechange(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0).current_types = _parse_types(resolved.event.arguments[1])

    def _handle_minus_typeadd(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        added = resolved.event.arguments[1].strip()
        if not added:
            raise _StateTransitionError("typeadd requires a nonempty type")
        member.current_types = tuple(dict.fromkeys((*member.current_types, added)))

    def _handle_minus_start(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        variant = _dynamic_effect_or_none(effect.normalized)
        canonical_id = effect.normalized if variant is None else variant.canonical_id
        if canonical_id in _DELAYED_MOVE_IDS:
            self._handle_delayed_start(resolved, canonical_id)
            return

        member = self._member_for(resolved, 0)
        if variant is None and effect.normalized in _DYNAMIC_CANONICAL_EFFECTS:
            raise _StateTransitionError(f"unsupported effect variant {effect.normalized!r}")
        if variant is not None and variant.canonical_id == "perishsong":
            member.effects.setdefault("perishsong", self.turn)
            member.perish_count = int(variant.value)
            return
        self._validate_effect(canonical_id, "effect")
        if variant is not None:
            member.effects.setdefault(canonical_id, self.turn)
            member.effect_variants[canonical_id] = variant.wire_id
            return
        if canonical_id == "typechange":
            if len(resolved.event.arguments) < 3:
                raise _StateTransitionError("typechange start requires resulting types")
            member.current_types = _parse_types(resolved.event.arguments[2])
        elif canonical_id == "mimic":
            if len(resolved.event.arguments) < 3:
                raise _StateTransitionError("Mimic start requires the copied move")
            if "mimic" not in member.moves:
                raise _StateTransitionError(
                    "Mimic effect requires Mimic in the member's move slots"
                )
            member.mimic_move = self._move_from_name(resolved.event.arguments[2])
        member.effects.setdefault(canonical_id, self.turn)

    def _handle_minus_end(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        if len(resolved.event.arguments) > 2:
            secondary_effect = normalize_showdown_id(resolved.event.arguments[2])
            if secondary_effect == "partiallytrapped":
                member = self._member_for(resolved, 0)
                member.effects.pop(secondary_effect, None)
                return
        variant = _dynamic_effect_or_none(effect.normalized)
        canonical_id = effect.normalized if variant is None else variant.canonical_id
        if canonical_id in _DELAYED_MOVE_IDS:
            self._handle_delayed_end(resolved, canonical_id)
            return
        member = self._member_for(resolved, 0)
        self._validate_effect(canonical_id, "effect")
        active_variant = member.effect_variants.get(canonical_id)
        if variant is not None and active_variant is not None and active_variant != variant.wire_id:
            raise _StateTransitionError(
                f"{canonical_id} ended with variant {effect.normalized!r}, "
                f"but active variant is {active_variant!r}"
            )
        member.effects.pop(canonical_id, None)
        member.effect_variants.pop(canonical_id, None)
        if canonical_id == "perishsong":
            member.perish_count = None
        if canonical_id == "typechange" and not member.terastallized:
            member.current_types = self._species_data(member.current_form).types
        elif canonical_id == "mimic":
            member.mimic_move = None

    def _handle_minus_singleturn(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        if effect in _SIDE_GUARDS:
            member = self._member_for(resolved, 0)
            side = member.member_id.side
            if effect in self.side_single_turn_effects[side]:
                raise _StateTransitionError(f"side guard {effect!r} started twice")
            self.side_conditions[side][effect] = self.turn
            self.side_single_turn_effects[side].add(effect)
            self._record_successful_protection(member)
            return

        member = self._member_for(resolved, 0)
        self._validate_effect(effect, "effect")
        member.effects[effect] = self.turn
        member.single_turn_effects.add(effect)
        if effect in _PROTECT_COUNTER_MOVES:
            self._record_successful_protection(member)

    def _handle_minus_singlemove(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "effect")
        member.effects[effect] = self.turn
        member.single_move_effects.add(effect)

    def _handle_minus_activate(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        breaking = (
            "[broken]" in resolved.event.arguments or effect.normalized in _BREAKING_PROTECT_MOVES
        )
        if breaking:
            target = self._member_for(resolved, 0)
            self._break_protect(target.member_id.side, target)
            return
        if effect.normalized != "skillswap":
            return
        references = tuple(
            reference for reference in resolved.pokemon_refs if reference.member_id is not None
        )
        if len(references) != 2:
            raise _StateTransitionError("Skill Swap requires source and target references")
        source = self.members[_required_member(references[0])]
        target = self.members[_required_member(references[1])]
        source_ability = self._current_ability(source)
        target_ability = self._current_ability(target)
        arguments = resolved.event.arguments
        if len(arguments) >= 4 and arguments[2] and arguments[3]:
            target_ability = arguments[2]
            source_ability = arguments[3]
        self._set_temporary_ability(source, target_ability)
        self._set_temporary_ability(target, source_ability)

    def _handle_minus_fail(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.protect_counter = 0
        self._discard_failed_slot_condition(member)
        for key, delayed in tuple(self.delayed_moves.items()):
            if delayed.source_member_id == member.member_id and not delayed.announced:
                del self.delayed_moves[key]

    def _record_successful_protection(self, member: _MutablePokemon) -> None:
        member.protect_counter += 1

    def _break_protect(self, side: ReplaySide, target: _MutablePokemon) -> None:
        active_guards = self.side_single_turn_effects[side]
        for effect in _BREAKABLE_MEMBER_PROTECT_EFFECTS:
            target.effects.pop(effect, None)
            target.single_turn_effects.discard(effect)
        for effect in tuple(active_guards):
            self.side_conditions[side].pop(effect, None)
        active_guards.clear()
        target.protect_counter = 0

    def _handle_minus_prepare(self, resolved: ResolvedProtocolEvent) -> None:
        move_id = normalize_showdown_id(resolved.event.arguments[1])
        if move_id not in self._moves:
            raise _StateTransitionError(f"prepare references unknown move {move_id!r}")
        self._member_for(resolved, 0).preparing = move_id

    def _handle_minus_mustrecharge(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0).effects["mustrecharge"] = self.turn

    def _handle_minus_weather(self, resolved: ResolvedProtocolEvent) -> None:
        weather = _required_effect(resolved).normalized
        if weather == "none":
            self.weather.clear()
            return
        self._validate_effect(weather, "weather")
        if "[upkeep]" in resolved.event.arguments[1:]:
            if weather not in self.weather:
                raise _StateTransitionError(f"weather upkeep has no active {weather!r} weather")
            return
        self.weather = {weather: self.turn}

    def _handle_minus_fieldstart(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "field")
        self.fields.setdefault(effect, self.turn)

    def _handle_minus_fieldend(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "field")
        if effect not in self.fields:
            raise _StateTransitionError(f"field effect {effect!r} ended before it started")
        self.fields.pop(effect)

    def _handle_minus_sidestart(self, resolved: ResolvedProtocolEvent) -> None:
        side = _resolved_reference(resolved, 0).pokemon_ref.side
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "side_condition")
        conditions = self.side_conditions[side]
        if effect in _STACKING_SIDE_CONDITIONS:
            conditions[effect] = conditions.get(effect, 0) + 1
        else:
            conditions.setdefault(effect, self.turn)

    def _handle_minus_sideend(self, resolved: ResolvedProtocolEvent) -> None:
        side = _resolved_reference(resolved, 0).pokemon_ref.side
        effect = _required_effect(resolved).normalized
        if effect in _SIDE_GUARDS:
            if effect not in self.side_conditions[side]:
                raise _StateTransitionError(f"side condition {effect!r} ended before it started")
            self.side_conditions[side].pop(effect)
            self.side_single_turn_effects[side].discard(effect)
            return
        self._validate_effect(effect, "side_condition")
        if effect not in self.side_conditions[side]:
            raise _StateTransitionError(f"side condition {effect!r} ended before it started")
        self.side_conditions[side].pop(effect)
        self.side_single_turn_effects[side].discard(effect)

    def _handle_minus_swapsideconditions(self, resolved: ResolvedProtocolEvent) -> None:
        self.side_conditions[ReplaySide.P1], self.side_conditions[ReplaySide.P2] = (
            self.side_conditions[ReplaySide.P2],
            self.side_conditions[ReplaySide.P1],
        )
        (
            self.side_single_turn_effects[ReplaySide.P1],
            self.side_single_turn_effects[ReplaySide.P2],
        ) = (
            self.side_single_turn_effects[ReplaySide.P2],
            self.side_single_turn_effects[ReplaySide.P1],
        )

    def _apply_provenance(self, resolved: ResolvedProtocolEvent) -> None:
        event = resolved.event
        if event.tag in {"-ability", "-endability"}:
            return
        source = _provenance_member(resolved)
        if source is None:
            return
        member = self.members[source]
        references = tuple(value for value in (event.cause, event.effect) if value is not None)
        for reference in references:
            if reference.namespace == "item":
                member.item = reference.name
            elif reference.namespace == "ability":
                current = member.ability
                if normalize_showdown_id(reference.name) == normalize_showdown_id(current.base):
                    continue
                if current.forme is not None and normalize_showdown_id(
                    reference.name
                ) == normalize_showdown_id(current.forme):
                    continue
                self._set_temporary_ability(member, reference.name)

    def _member_for(self, resolved: ResolvedProtocolEvent, argument_index: int) -> _MutablePokemon:
        return self.members[_required_member(_resolved_reference(resolved, argument_index))]

    def _validate_effect(self, effect: str, category: str) -> None:
        if category == "effect" and (
            effect in {"illusion", "mimic", "typechange"} or effect in _DYNAMIC_CANONICAL_EFFECTS
        ):
            return
        allowed = self._legal_effects.get(category)
        if allowed is not None and effect not in allowed:
            raise _StateTransitionError(
                f"unsupported {category.replace('_', ' ')} effect variant {effect!r}"
            )

    @staticmethod
    def _current_ability(member: _MutablePokemon) -> str:
        return member.transform.ability if member.transform is not None else member.ability.current

    @staticmethod
    def _set_temporary_ability(member: _MutablePokemon, ability: str) -> None:
        if member.transform is not None:
            member.transform = replace(member.transform, ability=ability)
        else:
            member.ability = AbilityState(member.ability.base, member.ability.forme, ability)

    def _clear_switch_state(self, member: _MutablePokemon) -> None:
        member.ability = AbilityState(member.ability.base, member.ability.forme)
        member.boosts = dict.fromkeys(_BOOST_NAMES, 0)
        member.effects.clear()
        member.effect_variants.clear()
        member.perish_count = None
        member.single_turn_effects.clear()
        member.single_move_effects.clear()
        member.active_turns = 0
        member.protect_counter = 0
        member.preparing = None
        member.last_move = None
        member.transform = None
        member.mimic_move = None
        if member.status != "slp":
            member.status_counter = 0
        species = self._species_data(member.current_form)
        if member.terastallized:
            if member.tera_type is None:
                raise _StateTransitionError("terastallized member has no tera type")
            member.current_types = (member.tera_type,)
        else:
            member.current_types = species.types

    @staticmethod
    def _decrement_move(member: _MutablePokemon, move_id: str) -> None:
        if member.transform is not None:
            updated = tuple(
                replace(move, current_pp=max(0, move.current_pp - 1))
                if move.move_id == move_id
                else move
                for move in member.transform.moves
            )
            member.transform = replace(member.transform, moves=updated)
            return
        if member.mimic_move is not None and member.mimic_move.move_id == move_id:
            member.mimic_move.current_pp = max(0, member.mimic_move.current_pp - 1)
            return
        member.moves[move_id].current_pp = max(0, member.moves[move_id].current_pp - 1)

    def snapshot(self, line_index: int) -> ReplayBattleState:
        members = tuple(
            self.members[ReplayMemberId(side, roster_index)].snapshot()
            for side in (ReplaySide.P1, ReplaySide.P2)
            for roster_index in range(6)
        )
        sides = tuple(
            ReplaySideState(
                side=side,
                active=(self.active.get((side, 0)), self.active.get((side, 1))),
                conditions=tuple(sorted(self.side_conditions[side].items())),
                used_mega=self.used_mega[side],
                used_z_move=self.used_z_move[side],
            )
            for side in (ReplaySide.P1, ReplaySide.P2)
        )
        delayed_moves = tuple(
            self.delayed_moves[key]
            for key in sorted(self.delayed_moves, key=lambda item: (item[0].side_index, item[1]))
        )
        slot_conditions = tuple(
            self.slot_conditions[key]
            for key in sorted(
                self.slot_conditions,
                key=lambda item: (item[0].side_index, item[1], item[2]),
            )
        )
        return ReplayBattleState(
            replay_id=self.replay_id,
            line_index=line_index,
            turn=self.turn,
            members=members,
            sides=(sides[0], sides[1]),
            weather=tuple(sorted(self.weather.items())),
            fields=tuple(sorted(self.fields.items())),
            delayed_moves=delayed_moves,
            slot_conditions=slot_conditions,
            team_sizes=(self.team_sizes[ReplaySide.P1], self.team_sizes[ReplaySide.P2]),
        )


def _species_index(dex: Mapping[str, Any]) -> dict[str, _SpeciesData]:
    index: dict[str, _SpeciesData] = {}
    for value in dex.get("species", ()):
        if not isinstance(value, Mapping):
            continue
        name = str(value.get("name", value.get("id", "")))
        species_id = normalize_showdown_id(str(value.get("id", name)))
        types = tuple(str(item) for item in value.get("types", ()) if isinstance(item, str))
        base_stats_value = value.get("baseStats", {})
        if not species_id or not types or not isinstance(base_stats_value, Mapping):
            continue
        try:
            base_stats = tuple(
                (stat, int(base_stats_value[stat]))
                for stat in ("hp", "atk", "def", "spa", "spd", "spe")
            )
        except (KeyError, TypeError, ValueError):
            continue
        abilities = value.get("abilities", {})
        native_ability = (
            str(abilities["0"])
            if isinstance(abilities, Mapping) and isinstance(abilities.get("0"), str)
            else None
        )
        data = _SpeciesData(
            name,
            str(value.get("baseSpecies", name)),
            types,
            base_stats,
            float(value.get("weightkg", 0.0)),
            native_ability,
        )
        index[species_id] = data
        index.setdefault(normalize_showdown_id(name), data)
    return index


def _move_index(dex: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        normalize_showdown_id(str(value.get("id", value.get("name", "")))): value
        for value in dex.get("moves", ())
        if isinstance(value, Mapping) and value.get("id", value.get("name"))
    }


def _legal_effect_index(dex: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    value = dex.get("legalProtocolEffects")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(category): frozenset(
            normalize_showdown_id(str(effect)) for effect in effects if str(effect)
        )
        for category, effects in value.items()
        if isinstance(effects, (list, tuple))
    }


def _maximum_pp(data: Mapping[str, Any], base_pp: int) -> int:
    return base_pp if data.get("noPPBoosts") else base_pp * 8 // 5


def _resolved_reference(
    resolved: ResolvedProtocolEvent,
    argument_index: int,
) -> ResolvedPokemonRefArgument:
    try:
        return next(
            reference
            for reference in resolved.pokemon_refs
            if reference.argument_index == argument_index
        )
    except StopIteration as exc:
        raise _StateTransitionError(
            f"event {resolved.event.tag!r} has no resolved reference at argument {argument_index}"
        ) from exc


def _required_member(reference: ResolvedPokemonRefArgument) -> ReplayMemberId:
    if reference.member_id is None:
        raise _StateTransitionError("state transition requires an active member reference")
    return reference.member_id


def _required_slot(reference: ResolvedPokemonRefArgument) -> int:
    slot = reference.pokemon_ref.active_slot
    if slot is None:
        raise _StateTransitionError("state transition requires an active slot")
    return slot


def _required_effect(resolved: ResolvedProtocolEvent) -> EffectReference:
    if resolved.event.effect is None:
        raise _StateTransitionError(f"event {resolved.event.tag!r} requires a normalized effect")
    return resolved.event.effect


def _dynamic_effect_or_none(effect: str) -> DynamicEffectVariant | None:
    try:
        return normalize_dynamic_effect(effect)
    except ValueError as exc:
        raise _StateTransitionError(f"unsupported effect variant {effect!r}") from exc


def _details_species(details: str) -> str:
    species = details.split(",", 1)[0].strip()
    if not species:
        raise _StateTransitionError("Pokémon details require a species")
    return species


def _parse_hp_status(value: str) -> tuple[float, str | None]:
    parts = value.split()
    if not parts:
        raise _StateTransitionError("HP status must not be empty")
    hp = parts[0]
    if "/" in hp:
        numerator_text, denominator_text = hp.split("/", 1)
        denominator_text = denominator_text.rstrip(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%"
        )
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError as exc:
            raise _StateTransitionError(f"invalid HP status {value!r}") from exc
        if denominator <= 0 or not 0 <= numerator <= denominator:
            raise _StateTransitionError(f"invalid HP range {value!r}")
        fraction = numerator / denominator
    elif hp == "0":
        fraction = 0.0
    else:
        raise _StateTransitionError(f"invalid HP status {value!r}")
    status = normalize_showdown_id(parts[1]) if len(parts) > 1 else None
    if status == "fnt":
        status = None
    elif status is not None and status not in _STATUS_NAMES:
        raise _StateTransitionError(f"invalid HP status suffix {status!r}")
    return fraction, status


def _parse_types(value: str) -> tuple[str, ...]:
    types = tuple(item.strip() for item in value.split("/") if item.strip())
    if not 1 <= len(types) <= 2:
        raise _StateTransitionError(f"invalid type list {value!r}")
    return types


def _boost_argument(arguments: tuple[str, ...]) -> tuple[str, ...]:
    if not arguments or arguments[0].startswith("["):
        return _BOOST_NAMES
    stats = tuple(normalize_showdown_id(value) for value in arguments[0].split(","))
    if not stats or any(stat not in _BOOST_NAMES for stat in stats):
        raise _StateTransitionError(f"invalid boost list {arguments[0]!r}")
    return stats


def _provenance_member(resolved: ResolvedProtocolEvent) -> ReplayMemberId | None:
    annotated = tuple(
        reference.member_id
        for reference in resolved.pokemon_refs
        if resolved.event.arguments[reference.argument_index].startswith(("[of] ", "[from] "))
        and reference.member_id is not None
    )
    if annotated:
        return annotated[-1]
    return next(
        (
            reference.member_id
            for reference in resolved.pokemon_refs
            if reference.member_id is not None
        ),
        None,
    )


def _diagnostic(event: ResolvedProtocolEvent, reason: str) -> ReplayEventDiagnostic:
    parsed = event.event
    return ReplayEventDiagnostic(
        replay_id=parsed.replay_id,
        line_index=parsed.line_index,
        tag=parsed.tag,
        normalized_effect="" if parsed.effect is None else parsed.effect.normalized,
        normalized_cause="" if parsed.cause is None else parsed.cause.normalized,
        raw_line=parsed.raw_line,
        reason=reason,
    )


def reduce_replay_state(
    replay_id: str,
    ots: tuple[OTSData, OTSData],
    events: Iterable[ResolvedProtocolEvent],
    *,
    dex: Mapping[str, Any],
) -> ReconstructedReplayState:
    """Apply resolved events once and emit immutable state at every replay cursor."""
    event_tuple = tuple(events)
    if tuple(sheet.side for sheet in ots) != (ReplaySide.P1, ReplaySide.P2):
        raise ValueError("OTS sheets must be ordered as p1 and p2")
    if any(not sheet.is_complete for sheet in ots):
        raise ValueError("State reconstruction requires complete OTS for both sides")
    if any(event.event.replay_id != replay_id for event in event_tuple):
        raise ValueError("State reconstruction cannot contain events from another replay")
    line_indices = tuple(event.event.line_index for event in event_tuple)
    if line_indices != tuple(sorted(set(line_indices))):
        raise ValueError("Resolved events must have unique ascending line indices")
    if not event_tuple:
        return ReconstructedReplayState(replay_id, ())
    try:
        reducer = _StateReducer(replay_id, ots, dex)
    except _StateTransitionError as exc:
        return ReconstructedReplayState(
            replay_id,
            (),
            (_diagnostic(event_tuple[0], str(exc)),),
        )

    snapshots: list[ReplayBattleState] = []
    for resolved in event_tuple:
        try:
            reducer.apply(resolved)
            snapshots.append(reducer.snapshot(resolved.event.line_index))
        except (KeyError, TypeError, ValueError) as exc:
            return ReconstructedReplayState(
                replay_id,
                (),
                (_diagnostic(resolved, str(exc)),),
            )
    return ReconstructedReplayState(replay_id, tuple(snapshots))


def reconstruct_replay_state(
    document: ReplayDocument,
    *,
    dex: Mapping[str, Any] | None = None,
) -> ReconstructedReplayState:
    """Resolve and reduce one normalized replay document without legacy battle objects."""
    if dex is None:
        from p0.model.resources import default_runtime_resources

        dex = default_runtime_resources().dex
    resolved = resolve_replay_events(document, dex=dex)
    if resolved.diagnostics:
        return ReconstructedReplayState(document.metadata.replay_id, (), resolved.diagnostics)
    return reduce_replay_state(
        document.metadata.replay_id,
        document.ots,
        resolved.events,
        dex=dex,
    )


__all__ = [
    "AbilityState",
    "DelayedMoveState",
    "DynamicEffectVariant",
    "MoveState",
    "ReconstructedReplayState",
    "ReplayBattleState",
    "ReplayPokemonState",
    "ReplaySideState",
    "SlotConditionState",
    "TransformSnapshot",
    "reconstruct_replay_state",
    "normalize_dynamic_effect",
    "reduce_replay_state",
]
