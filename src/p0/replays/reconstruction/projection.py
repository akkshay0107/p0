"""Causal player-relative views and replay stat estimates for reconstruction v2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from p0.battle.events import EventRecord, SpatialEventRecorder
from p0.battle.legality import GAME_END_DECISION, DecisionView
from p0.model.resources import default_runtime_resources
from p0.model.tokenizer import tokenizer
from p0.replays.identity import ReplayMemberId, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.decisions import (
    BoundaryKind,
    DecisionReconstruction,
    DecisionWindow,
    _mega_rules,
    build_decision_view,
)
from p0.replays.reconstruction.resolution import ResolvedProtocolEvent
from p0.replays.reconstruction.state import (
    AbilityState,
    MoveState,
    ReconstructedReplayState,
    ReplayBattleState,
    ReplayPokemonState,
    _maximum_pp,
)
from p0.replays.schema import DecisionRecord, DecisionType, OTSData, ReplayDiagnostics
from p0.teams.spread_usage import cosmetic_forme_aliases, load_spread_table_file
from p0.teams.stat_points import BaseStats, calculate_stats

_STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
_IMPUTATION_SOURCE_VERSION = 1
_DEX_INDEX_CACHE: (
    tuple[
        Mapping[str, Any],
        dict[str, Mapping[str, Any]],
        dict[str, Mapping[str, Any]],
        dict[str, str],
    ]
    | None
) = None


def _enum_name(value: str) -> str:
    """Return the enum-like spelling expected by the observation tokenizer."""
    return normalize_showdown_id(value).upper()


class ReplayNamedValue(NamedTuple):
    """Small immutable enum-like value accepted by the observation contracts."""

    name: str


class _FrozenMapping(Mapping[Any, Any]):
    """Picklable read-only mapping for projected observation data."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[Any, Any] | Iterable[tuple[Any, Any]] = ()) -> None:
        self._values = dict(values)

    def __getitem__(self, key: Any) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


_EMPTY_STATS: Mapping[str, int | None] = _FrozenMapping()


@dataclass(frozen=True, slots=True)
class ReplayMoveView:
    """Observation-facing move metadata copied from an immutable move state."""

    id: str
    type: ReplayNamedValue
    category: ReplayNamedValue
    target: str
    current_pp: int
    max_pp: int

    @property
    def non_ghost_target(self) -> bool:
        return False

    @property
    def deduced_target(self) -> int:
        return 0


@dataclass(frozen=True, slots=True, eq=False)
class ReplayPokemonView:
    """Causal observation view for one stable replay member."""

    member_id: ReplayMemberId
    state: ReplayPokemonState
    opponent: bool
    active: bool
    _species: str
    _base_species: str
    _ability: str
    _types: tuple[ReplayNamedValue, ...]
    _base_stats: Mapping[str, int]
    _moves: Mapping[str, ReplayMoveView]
    _effects: Mapping[ReplayNamedValue, int]
    _transformed: bool
    _boosts: Mapping[str, int] = field(default_factory=_FrozenMapping, repr=False, compare=False)
    identity_uncertain: bool = False

    @property
    def species(self) -> str:
        return self._species

    @property
    def base_species(self) -> str:
        return self._base_species

    @property
    def ability(self) -> str:
        return self._ability

    @property
    def item(self) -> str | None:
        return self.state.item

    @property
    def nature(self) -> str:
        return self.state.nature

    @property
    def moves(self) -> Mapping[str, ReplayMoveView]:
        return self._moves

    @property
    def type_1(self) -> ReplayNamedValue | None:
        return self._types[0] if self._types else None

    @property
    def type_2(self) -> ReplayNamedValue | None:
        return self._types[1] if len(self._types) > 1 else None

    @property
    def status(self) -> ReplayNamedValue | None:
        return None if self.state.status is None else ReplayNamedValue(self.state.status.upper())

    @property
    def base_stats(self) -> Mapping[str, int]:
        return self._base_stats

    @property
    def stats(self) -> Mapping[str, int | None]:
        """Replay stats are never exact; the pipeline supplies explicit overrides."""
        return _EMPTY_STATS

    @property
    def boosts(self) -> Mapping[str, int]:
        return self._boosts

    @property
    def current_hp_fraction(self) -> float:
        return 1.0 if self.state.hp_fraction is None else self.state.hp_fraction

    @property
    def protect_counter(self) -> int:
        return self.state.protect_counter

    @property
    def first_turn(self) -> bool:
        return self.state.active_turns == 1

    @property
    def weight(self) -> float:
        return (
            self.state.transform.weight if self.state.transform is not None else self.state.weight
        )

    @property
    def fainted(self) -> bool:
        return self.state.fainted

    @property
    def revealed(self) -> bool:
        return self.state.revealed

    @property
    def selected_in_teampreview(self) -> bool | None:
        return self.state.selected

    @property
    def effects(self) -> Mapping[ReplayNamedValue, int]:
        return self._effects

    @property
    def status_counter(self) -> int:
        return self.state.status_counter

    @property
    def preparing(self) -> bool:
        return self.state.preparing is not None

    @property
    def last_move(self) -> ReplayMoveView | None:
        return None if self.state.last_move is None else self._moves.get(self.state.last_move)

    @property
    def level(self) -> int:
        return self.state.level

    @property
    def is_dynamaxed(self) -> bool:
        return "dynamax" in dict(self.state.effects)

    @property
    def is_terastallized(self) -> bool:
        return self.state.terastallized

    @property
    def tera_type(self) -> ReplayNamedValue | None:
        return None if self.state.tera_type is None else ReplayNamedValue(self.state.tera_type)

    @property
    def types(self) -> tuple[ReplayNamedValue, ...]:
        return self._types

    @property
    def is_transformed(self) -> bool:
        return self._transformed


@dataclass(frozen=True, slots=True)
class ReplayBattleView:
    """Immutable player-relative battle view consumed by ObservationBuilder."""

    team: Mapping[str, ReplayPokemonView]
    opponent_team: Mapping[str, ReplayPokemonView]
    active_pokemon: tuple[ReplayPokemonView | None, ReplayPokemonView | None]
    opponent_active_pokemon: tuple[ReplayPokemonView | None, ReplayPokemonView | None]
    available_moves: tuple[tuple[ReplayMoveView, ...], tuple[ReplayMoveView, ...]]
    available_switches: tuple[tuple[ReplayPokemonView, ...], tuple[ReplayPokemonView, ...]]
    can_mega_evolve: tuple[bool, bool]
    force_switch: tuple[bool, bool]
    trapped: tuple[bool, bool]
    maybe_trapped: tuple[bool, bool]
    teampreview: bool
    player_role: str
    wait: bool
    weather: Mapping[ReplayNamedValue, int]
    fields: Mapping[ReplayNamedValue, int]
    side_conditions: Mapping[ReplayNamedValue, int]
    opponent_side_conditions: Mapping[ReplayNamedValue, int]
    turn: int
    used_mega_evolve: bool
    opponent_used_mega_evolve: bool
    decision: DecisionView
    identifiers: Mapping[str, ReplayPokemonView]
    spatial_events: tuple[EventRecord, ...] = ()
    stat_cache: dict[Any, Any] = field(default_factory=dict, compare=False, repr=False)

    def get_pokemon(self, identifier: str) -> ReplayPokemonView:
        return self.identifiers[identifier]

    def last_move(self, pokemon: ReplayPokemonView) -> str | None:
        return None if pokemon.last_move is None else pokemon.last_move.id


@dataclass(frozen=True, slots=True)
class ReplayStatValue:
    """One explicit replay stat estimate keyed by stable roster identity."""

    member_id: ReplayMemberId
    values: tuple[int, int, int, int, int, int] | None
    provenance: str
    confidence: float
    source_version: int = _IMPUTATION_SOURCE_VERSION

    def __post_init__(self) -> None:
        if self.provenance not in {"IMPUTED", "UNKNOWN"}:
            raise ValueError("Replay stat provenance must be IMPUTED or UNKNOWN")
        if self.values is not None and len(self.values) != len(_STAT_NAMES):
            raise ValueError("Replay stat values must use canonical stat order")
        if self.provenance == "UNKNOWN" and self.values is not None:
            raise ValueError("UNKNOWN replay stats cannot carry values")
        if self.provenance == "IMPUTED" and self.values is None:
            raise ValueError("IMPUTED replay stats require values")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Replay stat confidence must be in [0, 1]")
        if type(self.source_version) is not int or self.source_version < 1:
            raise ValueError("Replay stat source version must be positive")


@dataclass(frozen=True, slots=True)
class ProjectedSnapshot:
    """One pre-decision causal view, including the events since the previous decision."""

    decision_index: int
    turn: int
    pre_line_index: int
    post_line_index: int
    view: ReplayBattleView
    raw_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectedPerspective:
    """All projected decision views for one replay player."""

    game_id: str
    player: int
    snapshots: tuple[ProjectedSnapshot, ...]
    decisions: tuple[DecisionRecord, ...]
    diagnostics: ReplayDiagnostics
    # The board after the last line with the events since this player's last
    # decision; it feeds series memory and is never a labeled decision.
    final_view: ReplayBattleView

    def __post_init__(self) -> None:
        if not self.game_id or self.player not in (0, 1):
            raise ValueError("ProjectedPerspective requires a replay id and player index")
        if len(self.snapshots) != len(self.decisions):
            raise ValueError("Projected snapshots and decisions must have equal lengths")
        if any(decision.player != self.player for decision in self.decisions):
            raise ValueError("Projected decisions must belong to the perspective player")


def _species_index(dex: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = tuple(entry for entry in dex.get("species", ()) if isinstance(entry, Mapping))
    by_id = {
        normalize_showdown_id(str(entry.get("id", entry.get("name", "")))): entry
        for entry in entries
    }
    aliases = cosmetic_forme_aliases(dex)
    for alias, target in aliases.items():
        if target in by_id:
            by_id.setdefault(alias, by_id[target])
    for entry in entries:
        base = entry.get("baseSpecies", entry.get("name", entry.get("id", "")))
        fallback = by_id.get(normalize_showdown_id(str(base)), entry)
        for value in (*entry.get("formeOrder", ()), *entry.get("otherFormes", ())):
            if isinstance(value, str):
                by_id.setdefault(normalize_showdown_id(value), fallback)
    return by_id


def _cached_dex_indexes(
    dex: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, str]]:
    global _DEX_INDEX_CACHE
    if _DEX_INDEX_CACHE is None or _DEX_INDEX_CACHE[0] is not dex:
        moves = {
            normalize_showdown_id(str(entry.get("id", entry.get("name", "")))): entry
            for entry in dex.get("moves", ())
            if isinstance(entry, Mapping)
        }
        move_categories = {
            move_id: str(entry.get("category", "")) for move_id, entry in moves.items()
        }
        _DEX_INDEX_CACHE = (dex, _species_index(dex), moves, move_categories)
    return _DEX_INDEX_CACHE[1], _DEX_INDEX_CACHE[2], _DEX_INDEX_CACHE[3]


def impute_replay_stats(
    document: ReplayDocument,
    *,
    dex: Mapping[str, Any],
) -> tuple[ReplayStatValue, ...]:
    """Estimate every OTS member exactly once, retaining explicit unknowns."""
    table = load_spread_table_file()
    species, _, move_categories = _cached_dex_indexes(dex)
    estimates: list[ReplayStatValue] = []
    for sheet in document.ots:
        for member in sheet.members:
            entry = species.get(normalize_showdown_id(member.species))
            base_stats = entry.get("baseStats") if entry is not None else None
            if not isinstance(base_stats, Mapping):
                estimates.append(ReplayStatValue(member.member_id, None, "UNKNOWN", 0.0))
                continue

            categories = tuple(
                move_categories.get(normalize_showdown_id(move), "") for move in member.moves
            )
            nature = member.nature or "serious"
            estimate = table.resolve(member.species, nature, categories)
            if estimate is None:
                estimates.append(ReplayStatValue(member.member_id, None, "UNKNOWN", 0.0))
                continue

            values = calculate_stats(
                BaseStats.from_mapping(base_stats),
                estimate.points,
                nature,
                member.level,
            )
            estimates.append(
                ReplayStatValue(
                    member.member_id,
                    values,
                    "IMPUTED",
                    estimate.confidence,
                )
            )
    return tuple(estimates)


def _move_view(move: MoveState) -> ReplayMoveView:
    return ReplayMoveView(
        move.move_id,
        ReplayNamedValue(move.move_type),
        ReplayNamedValue(move.category),
        move.target,
        move.current_pp,
        move.max_pp,
    )


def _pokemon_view(
    member: ReplayPokemonState,
    *,
    perspective: int,
    active: bool,
) -> ReplayPokemonView:
    transform = member.transform
    moves = transform.moves if transform is not None else member.moves
    move_views = _FrozenMapping({move.move_id: _move_view(move) for move in moves})
    if transform is not None:
        base_stats = {**dict(member.base_stats), **dict(transform.non_hp_base_stats)}
        species = base_species = transform.species
        ability = transform.ability
    else:
        base_stats = member.base_stats
        own_or_public = (
            perspective == member.member_id.side.side_index or not active or member.revealed
        )
        species = member.current_form if own_or_public else member.displayed_species
        base_species = member.original_species
        ability = member.ability.current
    effects = _FrozenMapping(
        {ReplayNamedValue(_enum_name(name)): value for name, value in member.effects}
    )
    return ReplayPokemonView(
        member.member_id,
        member,
        member.member_id.side.side_index != perspective,
        active,
        species,
        base_species,
        ability,
        tuple(ReplayNamedValue(value) for value in member.current_types),
        _FrozenMapping(base_stats),
        move_views,
        effects,
        transform is not None,
        _boosts=_FrozenMapping(member.boosts),
    )


def _mask_hidden_illusion(
    state: ReplayBattleState,
    ots: tuple[OTSData, OTSData],
    perspective: int,
    views: dict[ReplayMemberId, ReplayPokemonView],
    dex: Mapping[str, Any],
) -> dict[ReplayMemberId, ReplayPokemonView]:
    """
    Keep later identity resolution out of an opponent's earlier view.

    Arguments:
      state: replay state at the current decision boundary
      ots: public open team sheets for both sides
      perspective: observing player's side index
      views: roster views to replace when a disguise has a public match
      dex: pinned species and move data for public roster fields

    Returns:
      active views that show the disguise without exposing the true member
    """
    opponent = 1 - perspective
    active_ids = state.sides[opponent].active
    hidden = tuple(
        member_id
        for member_id in active_ids
        if member_id is not None
        and not (member := state.member(member_id)).revealed
        and normalize_showdown_id(member.displayed_species)
        != normalize_showdown_id(member.current_form)
    )
    if not hidden:
        return {}

    species_index, move_index, _ = _cached_dex_indexes(dex)
    hidden_ids = set(hidden)
    assigned = {member_id for member_id in active_ids if member_id is not None} - hidden_ids
    active_overrides: dict[ReplayMemberId, ReplayPokemonView] = {}
    for member_id in hidden:
        member = state.member(member_id)
        displayed_id = normalize_showdown_id(member.displayed_species)
        candidate = next(
            (
                ots_member
                for ots_member in ots[opponent].members
                if normalize_showdown_id(ots_member.species) == displayed_id
                and ots_member.member_id not in assigned
                and ots_member.member_id not in hidden_ids
                and not state.member(ots_member.member_id).revealed
            ),
            None,
        )
        displayed_species = species_index.get(displayed_id)
        if displayed_species is None:
            raise ValueError("Illusion display species is absent from the pinned dex")
        true_sheet = ots[opponent].members[member_id.roster_index]
        true_species = species_index.get(normalize_showdown_id(true_sheet.species))
        if true_species is None:
            raise ValueError("Illusion member species is absent from the pinned dex")

        display_stats = tuple((k, int(displayed_species["baseStats"][k])) for k in _STAT_NAMES)
        true_stats = tuple((k, int(true_species["baseStats"][k])) for k in _STAT_NAMES)
        display_types = tuple(str(v) for v in displayed_species["types"])
        true_types = tuple(str(v) for v in true_species["types"])
        public_moves = tuple(
            MoveState(
                (m_id := normalize_showdown_id(name)),
                str((data := move_index[m_id]).get("name", name)),
                str(data["type"]),
                str(data["category"]),
                str(data["target"]),
                (max_pp := _maximum_pp(data, int(data["pp"]))),
                max_pp,
            )
            for name in true_sheet.moves
        )
        public_effects = tuple(
            (name, count) for name, count in member.effects if name != "illusion"
        )
        public_effect_names = {name for name, _ in public_effects}
        displayed_state = replace(
            member,
            member_id=member_id if candidate is None else candidate.member_id,
            nickname=member.displayed_species if candidate is None else candidate.nickname,
            original_species=str(displayed_species.get("baseSpecies", member.displayed_species)),
            current_form=member.displayed_species,
            displayed_species=member.displayed_species,
            nature="",
            item=None,
            ability=AbilityState("unknown"),
            base_types=display_types,
            current_types=display_types,
            base_stats=display_stats,
            weight=float(displayed_species["weightkg"]),
            moves=(),
            effects=public_effects,
            effect_variants=tuple(p for p in member.effect_variants if p[0] in public_effect_names),
            effect_sources=tuple(p for p in member.effect_sources if p[0] in public_effect_names),
            tera_type=None,
            transform=None,
            revealed=True,
            selected=None,
            last_move=None,
        )
        static_state = replace(
            member,
            nickname=true_sheet.nickname,
            original_species=str(true_species.get("baseSpecies", true_sheet.species)),
            current_form=true_sheet.species,
            displayed_species=true_sheet.species,
            nature=true_sheet.nature,
            hp_fraction=None,
            status=None,
            item=true_sheet.item,
            ability=AbilityState(true_sheet.ability or member.ability.base),
            base_types=true_types,
            current_types=true_types,
            base_stats=true_stats,
            weight=float(true_species["weightkg"]),
            moves=public_moves,
            boosts=tuple((name, 0) for name, _ in member.boosts),
            effects=(),
            effect_variants=(),
            effect_sources=(),
            perish_count=None,
            tera_type=None,
            terastallized=False,
            transform=None,
            revealed=False,
            selected=None,
            fainted=False,
            active_turns=0,
            status_counter=0,
            protect_counter=0,
            preparing=None,
            last_move=None,
            added_type=None,
            dragoncheer_has_dragon_type=False,
        )
        displayed_view = replace(
            _pokemon_view(displayed_state, perspective=perspective, active=True),
            identity_uncertain=True,
        )
        if candidate is not None:
            views[candidate.member_id] = displayed_view
            assigned.add(candidate.member_id)
        views[member_id] = _pokemon_view(static_state, perspective=perspective, active=False)
        active_overrides[member_id] = displayed_view

    return active_overrides


def _named_mapping(values: Iterable[tuple[str, int]]) -> Mapping[ReplayNamedValue, int]:
    return _FrozenMapping({ReplayNamedValue(_enum_name(name)): value for name, value in values})


def project_battle_view(
    state: ReplayBattleState,
    ots: tuple[OTSData, OTSData],
    *,
    perspective: int,
    decision: DecisionView,
    spatial_events: tuple[EventRecord, ...] = (),
    preview: bool = False,
    dex: Mapping[str, Any] | None = None,
) -> ReplayBattleView:
    """Project one immutable public snapshot into the observation-facing contract."""
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")

    active_ids = {
        member_id for side in state.sides for member_id in side.active if member_id is not None
    }
    views = {
        member.member_id: _pokemon_view(
            member,
            perspective=perspective,
            active=member.member_id in active_ids,
        )
        for member in state.members
    }
    active_overrides = _mask_hidden_illusion(
        state,
        ots,
        perspective,
        views,
        default_runtime_resources().dex if dex is None else dex,
    )
    teams = tuple(
        _FrozenMapping(
            {
                f"{member.member_id.roster_index}:{views[member.member_id].species}": views[
                    member.member_id
                ]
                for member in ots[side].members
            }
        )
        for side in (0, 1)
    )
    active = tuple(
        tuple(
            active_overrides.get(member_id, views.get(member_id)) if member_id is not None else None
            for member_id in side.active
        )
        for side in state.sides
    )
    own_active = (active[perspective][0], active[perspective][1])
    opponent = 1 - perspective
    opponent_active = (active[opponent][0], active[opponent][1])
    switches = tuple(
        view
        for member_id, view in views.items()
        if member_id.side.side_index == perspective
        and member_id not in state.sides[perspective].active
        and view.selected_in_teampreview is True
        and not view.fainted
    )
    available_moves = tuple(
        tuple() if view is None else tuple(view.moves.values()) for view in own_active
    )
    identifiers: dict[str, ReplayPokemonView] = {}
    for side_index, side_active in enumerate(active):
        for slot, view in enumerate(side_active):
            if view is not None:
                identifiers[f"p{side_index + 1}{'ab'[slot]}"] = view
                identifiers[f"p{side_index + 1}{'ab'[slot]}: {view.state.nickname}"] = view

    return ReplayBattleView(
        team=teams[perspective],
        opponent_team=teams[opponent],
        active_pokemon=own_active,
        opponent_active_pokemon=opponent_active,
        available_moves=(available_moves[0], available_moves[1]),
        available_switches=(switches, switches),
        can_mega_evolve=(decision.slots[0].can_mega, decision.slots[1].can_mega),
        force_switch=(decision.slots[0].force_switch, decision.slots[1].force_switch),
        trapped=(decision.slots[0].trapped, decision.slots[1].trapped),
        maybe_trapped=(False, False),
        teampreview=preview,
        player_role=f"p{perspective + 1}",
        wait=decision.wait,
        weather=_named_mapping(state.weather),
        fields=_named_mapping(state.fields),
        side_conditions=_named_mapping(state.sides[perspective].conditions),
        opponent_side_conditions=_named_mapping(state.sides[opponent].conditions),
        turn=state.turn,
        used_mega_evolve=state.sides[perspective].used_mega,
        opponent_used_mega_evolve=state.sides[opponent].used_mega,
        decision=decision,
        identifiers=_FrozenMapping(identifiers),
        spatial_events=spatial_events,
    )


def _hp_before(
    state: ReplayBattleState | None,
    event: ResolvedProtocolEvent,
) -> float | None:
    if state is None:
        return None
    reference = next(
        (reference for reference in event.pokemon_refs if reference.argument_index == 0),
        None,
    )
    if reference is None or reference.member_id is None:
        return None
    return state.member(reference.member_id).hp_fraction


def _spatial_records(
    document: ReplayDocument,
    events: tuple[ResolvedProtocolEvent, ...],
    state: Mapping[int, ReplayBattleState],
    *,
    perspective: int,
    starts: tuple[int, ...],
) -> dict[int, tuple[EventRecord, ...]]:
    """Return the events each of this player's decisions observed, keyed by decision start."""
    recorder = SpatialEventRecorder(player_role=f"p{perspective + 1}")
    records: dict[int, tuple[EventRecord, ...]] = {}
    cursor = 0
    for start in starts:
        for line_index in range(cursor, start):
            event = events[line_index]
            before = state.get(line_index - 1)
            recorder.apply_line(
                document.protocol_lines[line_index].parts,
                tokenizer,
                lambda _identifier, before=before, event=event: _hp_before(before, event),
            )
        records[start] = recorder.consume()
        cursor = start
    return records


def _projection_snapshot_lines(
    events: tuple[ResolvedProtocolEvent, ...],
    windows: tuple[DecisionWindow, ...],
) -> frozenset[int]:
    policy_windows = tuple(window for window in windows if window.is_policy_request)
    lines = {
        len(events) - 1,
        *(
            window.start_line_index
            if window.kind is BoundaryKind.TEAM_PREVIEW
            else window.start_line_index - 1
            for window in policy_windows
        ),
        *(
            event.event.line_index - 1
            for event in events
            if event.event.line_index > 0 and event.event.tag in {"-damage", "-heal"}
        ),
    }
    return frozenset(line for line in lines if line >= 0)


def project_replay_perspectives(
    document: ReplayDocument,
    events: Iterable[ResolvedProtocolEvent],
    state: ReconstructedReplayState,
    decisions: tuple[DecisionReconstruction, DecisionReconstruction],
    *,
    dex: Mapping[str, Any],
) -> tuple[ProjectedPerspective, ProjectedPerspective]:
    """Build p1 and p2 views from one shared immutable trace."""
    event_tuple = tuple(events)
    snapshots = state.require_accepted()
    if any(result.diagnostics for result in decisions):
        raise ValueError("Projection requires accepted decision reconstruction")
    snapshots_by_line = {snapshot.line_index: snapshot for snapshot in snapshots}
    required_lines = _projection_snapshot_lines(event_tuple, decisions[0].windows)
    missing_lines = required_lines - snapshots_by_line.keys()
    if missing_lines:
        raise ValueError(
            f"State reconstruction is missing projection lines: {sorted(missing_lines)}"
        )

    mega_rules = _mega_rules(dex)
    window_by_bounds = {
        (window.start_line_index, window.end_line_index): window for window in decisions[0].windows
    }
    # Each player's interval closes only at that player's own decisions; a
    # waiting side keeps accumulating until it next chooses. The final interval
    # runs to the last line.
    final_start = len(event_tuple)
    spatial = tuple(
        _spatial_records(
            document,
            event_tuple,
            snapshots_by_line,
            perspective=perspective,
            starts=(
                *(decision.pre_line_index for decision in reconstruction.decisions),
                final_start,
            ),
        )
        for perspective, reconstruction in enumerate(decisions)
    )

    projected: list[ProjectedPerspective] = []
    for perspective, reconstruction in enumerate(decisions):
        projected_snapshots: list[ProjectedSnapshot] = []
        for decision in reconstruction.decisions:
            key = (decision.pre_line_index, decision.post_line_index)
            window = window_by_bounds.get(key)
            if window is None:
                raise ValueError("Decision record does not match a classified window")
            window_events = event_tuple[window.start_line_index : window.end_line_index]
            preview = decision.decision_type is DecisionType.TEAM_PREVIEW
            if preview:
                snapshot = snapshots_by_line[decision.pre_line_index]
            else:
                if decision.pre_line_index == 0:
                    raise ValueError("Non-preview decision cannot start at line zero")
                snapshot = snapshots_by_line[decision.pre_line_index - 1]
            decision_view = build_decision_view(
                snapshot,
                document.ots[perspective],
                perspective,
                window_events,
                preview=preview,
                dex=dex,
                mega_rules=mega_rules,
            )
            projected_snapshots.append(
                ProjectedSnapshot(
                    decision.decision_index,
                    snapshot.turn,
                    decision.pre_line_index,
                    decision.post_line_index,
                    project_battle_view(
                        snapshot,
                        document.ots,
                        perspective=perspective,
                        decision=decision_view,
                        spatial_events=spatial[perspective][decision.pre_line_index],
                        preview=preview,
                        dex=dex,
                    ),
                    tuple(
                        line.raw
                        for line in document.protocol_lines[
                            decision.pre_line_index : decision.post_line_index
                        ]
                    ),
                )
            )
        projected.append(
            ProjectedPerspective(
                document.metadata.replay_id,
                perspective,
                tuple(projected_snapshots),
                reconstruction.decisions,
                ReplayDiagnostics(
                    counters={},
                    parse_errors=tuple(
                        diagnostic.reason for diagnostic in reconstruction.diagnostics
                    ),
                ),
                project_battle_view(
                    snapshots_by_line[final_start - 1],
                    document.ots,
                    perspective=perspective,
                    decision=GAME_END_DECISION,
                    spatial_events=spatial[perspective][final_start],
                    dex=dex,
                ),
            )
        )
    return projected[0], projected[1]


__all__ = [
    "ProjectedPerspective",
    "ProjectedSnapshot",
    "ReplayBattleView",
    "ReplayNamedValue",
    "ReplayMoveView",
    "ReplayPokemonView",
    "ReplayStatValue",
    "impute_replay_stats",
    "project_battle_view",
    "project_replay_perspectives",
]
