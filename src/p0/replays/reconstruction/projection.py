"""Causal player-relative views and replay stat estimates for reconstruction v2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from p0.battle.events import SpatialSlotRecord, SpatialTurnRecorder
from p0.battle.legality import DecisionView
from p0.model.tokenizer import tokenizer
from p0.replays.identity import ReplayMemberId, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.decisions import (
    DecisionReconstruction,
    _mega_rules,
    build_decision_view,
)
from p0.replays.reconstruction.resolution import ResolvedProtocolEvent
from p0.replays.reconstruction.state import (
    MoveState,
    ReconstructedReplayState,
    ReplayBattleState,
    ReplayPokemonState,
)
from p0.replays.schema import DecisionRecord, DecisionType, OTSData, ReplayDiagnostics
from p0.teams.spread_usage import cosmetic_forme_aliases, load_spread_table_file
from p0.teams.stat_points import BaseStats, calculate_stats

_STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
_IMPUTATION_SOURCE_VERSION = 2
_ENUM_ALIASES = {
    "raindance": "RAINDANCE",
    "sunnyday": "SUNNYDAY",
    "trickroom": "TRICK_ROOM",
    "magicroom": "MAGIC_ROOM",
    "wonderroom": "WONDER_ROOM",
    "toxicspikes": "TOXIC_SPIKES",
}
_DEX_INDEX_CACHE: tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], dict[str, str]] | None = (
    None
)


def _enum_name(value: str) -> str:
    """Return the enum-like spelling expected by the observation tokenizer."""
    normalized = normalize_showdown_id(value)
    return _ENUM_ALIASES.get(normalized, normalized.upper())


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
    spatial_turn: tuple[SpatialSlotRecord, ...] = ()
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
    """One pre-decision causal view and its preceding spatial history."""

    decision_index: int
    turn: int
    pre_line_index: int
    post_line_index: int
    view: ReplayBattleView
    spatial_turn: tuple[SpatialSlotRecord, ...]
    raw_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectedPerspective:
    """All projected decision views for one replay player."""

    game_id: str
    player: int
    snapshots: tuple[ProjectedSnapshot, ...]
    decisions: tuple[DecisionRecord, ...]
    diagnostics: ReplayDiagnostics

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
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    global _DEX_INDEX_CACHE
    if _DEX_INDEX_CACHE is None or _DEX_INDEX_CACHE[0] is not dex:
        move_categories = {
            normalize_showdown_id(str(entry.get("id", entry.get("name", "")))): str(
                entry.get("category", "")
            )
            for entry in dex.get("moves", ())
            if isinstance(entry, Mapping)
        }
        _DEX_INDEX_CACHE = (dex, _species_index(dex), move_categories)
    return _DEX_INDEX_CACHE[1], _DEX_INDEX_CACHE[2]


def impute_replay_stats(
    document: ReplayDocument,
    *,
    dex: Mapping[str, Any],
) -> tuple[ReplayStatValue, ...]:
    """Estimate every OTS member exactly once, retaining explicit unknowns."""
    table = load_spread_table_file()
    species, move_categories = _cached_dex_indexes(dex)
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
        base_stats = dict(member.base_stats)
        base_stats.update(dict(transform.non_hp_base_stats))
    else:
        base_stats = member.base_stats
    if transform is not None:
        species = transform.species
        ability = transform.ability
        types = transform.types
        base_species = transform.species
    elif perspective == member.member_id.side.side_index or not active or member.revealed:
        species = member.current_form
        ability = member.ability.current
        types = member.current_types
        base_species = member.original_species
    else:
        species = member.displayed_species
        ability = member.ability.current
        types = member.current_types
        base_species = member.original_species
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
        tuple(ReplayNamedValue(value) for value in types),
        _FrozenMapping(base_stats),
        move_views,
        effects,
        transform is not None,
        _boosts=_FrozenMapping(member.boosts),
    )


def _named_mapping(values: Iterable[tuple[str, int]]) -> Mapping[ReplayNamedValue, int]:
    return _FrozenMapping({ReplayNamedValue(_enum_name(name)): value for name, value in values})


def project_battle_view(
    state: ReplayBattleState,
    ots: tuple[OTSData, OTSData],
    *,
    perspective: int,
    decision: DecisionView,
    spatial_turn: tuple[SpatialSlotRecord, ...] = (),
    preview: bool = False,
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
        tuple(views.get(member_id) if member_id is not None else None for member_id in side.active)
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
        spatial_turn=spatial_turn,
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
    state: tuple[ReplayBattleState, ...],
    *,
    perspective: int,
    starts: tuple[int, ...],
) -> dict[int, tuple[SpatialSlotRecord, ...]]:
    recorder = SpatialTurnRecorder(player_role=f"p{perspective + 1}")
    records: dict[int, tuple[SpatialSlotRecord, ...]] = {}
    cursor = 0
    for start in starts:
        for line_index in range(cursor, start):
            event = events[line_index]
            if event.event.tag == "turn":
                recorder.reset_turn()
            before = None if line_index == 0 else state[line_index - 1]
            recorder.apply_line(
                document.protocol_lines[line_index].parts,
                tokenizer,
                lambda _identifier, before=before, event=event: _hp_before(before, event),
            )
        records[start] = recorder.to_records()
        cursor = start
    return records


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
    if len(event_tuple) != len(snapshots):
        raise ValueError("Projection requires one state snapshot per resolved event")
    if any(result.diagnostics for result in decisions):
        raise ValueError("Projection requires accepted decision reconstruction")

    mega_rules = _mega_rules(dex)
    window_by_bounds = {
        (window.start_line_index, window.end_line_index): window for window in decisions[0].windows
    }
    starts = tuple(
        dict.fromkeys(
            decision.pre_line_index
            for reconstruction in decisions
            for decision in reconstruction.decisions
        )
    )
    spatial = tuple(
        _spatial_records(
            document,
            event_tuple,
            snapshots,
            perspective=perspective,
            starts=starts,
        )
        for perspective in (0, 1)
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
                snapshot = snapshots[decision.pre_line_index]
            else:
                if decision.pre_line_index == 0:
                    raise ValueError("Non-preview decision cannot start at line zero")
                snapshot = snapshots[decision.pre_line_index - 1]
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
                        spatial_turn=spatial[perspective][decision.pre_line_index],
                        preview=preview,
                    ),
                    spatial[perspective][decision.pre_line_index],
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
