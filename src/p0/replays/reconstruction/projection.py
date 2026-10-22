"""Causal player-relative views and replay stat estimates for reconstruction v2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Any, NamedTuple

from p0.battle.events import EventRecord, SpatialEventRecorder
from p0.battle.legality import GAME_END_DECISION, DecisionView
from p0.model.resources import default_runtime_resources
from p0.model.tokenizer import tokenizer
from p0.replays.identity import ReplayMemberId, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.decisions import (
    DecisionReconstruction,
    DecisionWindow,
)
from p0.replays.reconstruction.resolution import ResolvedProtocolEvent
from p0.replays.reconstruction.state import (
    MoveState,
    ReconstructedReplayState,
    ReplayBattleState,
    ReplayPokemonState,
    dex_indexes,
)
from p0.replays.schema import DecisionRecord, DecisionType, OTSData
from p0.teams.spread_usage import cosmetic_forme_aliases, load_spread_table_file
from p0.teams.stat_points import BaseStats, calculate_stats


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


class ReplayMoveView(NamedTuple):
    """Observation-facing move metadata copied from an immutable move state."""

    id: str
    type: ReplayNamedValue
    category: ReplayNamedValue
    current_pp: int
    max_pp: int


@dataclass(frozen=True, slots=True, eq=False)
class ReplayPokemonView:
    """Causal observation view for one stable replay member."""

    member_id: ReplayMemberId
    state: ReplayPokemonState
    species: str
    base_species: str
    ability: str
    types: tuple[ReplayNamedValue, ...]
    base_stats: Mapping[str, int]
    moves: Mapping[str, ReplayMoveView]
    effects: Mapping[ReplayNamedValue, int]
    revealed: bool
    boosts: Mapping[str, int]

    @property
    def item(self) -> str | None:
        return self.state.item

    @property
    def nature(self) -> str:
        return self.state.nature

    @property
    def type_1(self) -> ReplayNamedValue | None:
        return self.types[0] if self.types else None

    @property
    def type_2(self) -> ReplayNamedValue | None:
        return self.types[1] if len(self.types) > 1 else None

    @property
    def status(self) -> ReplayNamedValue | None:
        return None if self.state.status is None else ReplayNamedValue(self.state.status.upper())

    @property
    def stats(self) -> Mapping[str, int | None]:
        """Replay stats are never exact; the pipeline supplies explicit overrides."""
        return _EMPTY_STATS

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
    def selected_in_teampreview(self) -> bool | None:
        return self.state.selected

    @property
    def status_counter(self) -> int:
        return self.state.status_counter

    @property
    def preparing(self) -> bool:
        return self.state.preparing is not None

    @property
    def last_move(self) -> ReplayMoveView | None:
        return None if self.state.last_move is None else self.moves.get(self.state.last_move)

    @property
    def level(self) -> int:
        return self.state.level

    @property
    def is_transformed(self) -> bool:
        return self.state.transform is not None


@dataclass(frozen=True, slots=True)
class ReplayBattleView:
    """Immutable player-relative battle view consumed by ObservationBuilder."""

    team: Mapping[str, ReplayPokemonView]
    opponent_team: Mapping[str, ReplayPokemonView]
    active_pokemon: tuple[ReplayPokemonView | None, ReplayPokemonView | None]
    opponent_active_pokemon: tuple[ReplayPokemonView | None, ReplayPokemonView | None]
    available_switches: tuple[tuple[ReplayPokemonView, ...], tuple[ReplayPokemonView, ...]]
    can_mega_evolve: tuple[bool, bool]
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
    spatial_events: tuple[EventRecord, ...] = ()
    stat_cache: dict[Any, Any] = field(default_factory=dict, compare=False, repr=False)

    def last_move(self, pokemon: ReplayPokemonView) -> str | None:
        return None if pokemon.last_move is None else pokemon.last_move.id


class ReplayStatValue(NamedTuple):
    """One explicit replay stat estimate keyed by stable roster identity."""

    member_id: ReplayMemberId
    values: tuple[int, int, int, int, int, int] | None
    provenance: str
    confidence: float


class ProjectedSnapshot(NamedTuple):
    """One pre-decision causal view, including the events since the previous decision."""

    decision_index: int
    pre_line_index: int
    post_line_index: int
    view: ReplayBattleView
    raw_lines: tuple[str, ...]


class ProjectedPerspective(NamedTuple):
    """All projected decision views for one replay player."""

    player: int
    snapshots: tuple[ProjectedSnapshot, ...]
    decisions: tuple[DecisionRecord, ...]
    # The board after the last line with the events since this player's last
    # decision; it feeds series memory and is never a labeled decision.
    final_view: ReplayBattleView


@cache
def _species_entries() -> dict[str, Mapping[str, Any]]:
    """Index dex species by id, cosmetic alias, and forme name."""
    dex = default_runtime_resources().dex
    by_id = {entry["id"]: entry for entry in dex["species"]}
    for alias, target in cosmetic_forme_aliases(dex).items():
        if target in by_id:
            by_id.setdefault(alias, by_id[target])
    for entry in dex["species"]:
        fallback = by_id.get(normalize_showdown_id(entry["baseSpecies"]), entry)
        for value in (*entry.get("formeOrder", ()), *entry.get("otherFormes", ())):
            by_id.setdefault(normalize_showdown_id(value), fallback)
    return by_id


def impute_replay_stats(document: ReplayDocument) -> tuple[ReplayStatValue, ...]:
    """Estimate every OTS member exactly once, retaining explicit unknowns."""
    table = load_spread_table_file()
    species = _species_entries()
    _, moves, _ = dex_indexes()
    estimates: list[ReplayStatValue] = []
    for sheet in document.ots:
        for member in sheet.members:
            entry = species.get(normalize_showdown_id(member.species))
            base_stats = entry.get("baseStats") if entry is not None else None
            if not isinstance(base_stats, Mapping):
                estimates.append(ReplayStatValue(member.member_id, None, "UNKNOWN", 0.0))
                continue

            categories = tuple(
                moves[move_id]["category"]
                if (move_id := normalize_showdown_id(move)) in moves
                else ""
                for move in member.moves
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


# Move slots repeat across snapshots; most change only when their PP does.
@lru_cache(maxsize=1 << 14)
def _move_view(move: MoveState) -> ReplayMoveView:
    return ReplayMoveView(
        move.move_id,
        ReplayNamedValue(move.move_type),
        ReplayNamedValue(move.category),
        move.current_pp,
        move.max_pp,
    )


def _pokemon_view(
    member: ReplayPokemonState,
    *,
    perspective: int,
) -> ReplayPokemonView:
    own = perspective == member.member_id.side.side_index
    transform = member.transform
    move_views = _FrozenMapping({move.move_id: _move_view(move) for move in member.moves})
    if transform is not None:
        base_stats = {**dict(member.base_stats), **dict(transform.non_hp_base_stats)}
        species = base_species = transform.species
        ability = transform.ability
    else:
        base_stats = member.base_stats
        species = member.current_form
        base_species = member.original_species
        ability = member.ability.current
    effects = _FrozenMapping(
        {ReplayNamedValue(_enum_name(name)): value for name, value in member.effects}
    )
    return ReplayPokemonView(
        member.member_id,
        member,
        species,
        base_species,
        ability,
        tuple(ReplayNamedValue(value) for value in member.current_types),
        _FrozenMapping(base_stats),
        move_views,
        effects,
        # The owner also knows its Illusion user entered while disguised.
        member.revealed or (own and member.selected is True),
        _FrozenMapping(member.boosts),
    )


def _named_mapping(values: Iterable[tuple[str, int]]) -> Mapping[ReplayNamedValue, int]:
    return _FrozenMapping({ReplayNamedValue(_enum_name(name)): value for name, value in values})


def _cached_pokemon_view(
    member: ReplayPokemonState,
    perspective: int,
    active: bool,
    cache: dict[tuple[ReplayMemberId, bool], ReplayPokemonView],
) -> ReplayPokemonView:
    # A disguise and its benched team entry can share a member id.
    key = (member.member_id, active)
    view = cache.get(key)
    if view is None or view.state is not member:
        view = cache[key] = _pokemon_view(member, perspective=perspective)
    return view


def project_battle_view(
    state: ReplayBattleState,
    ots: tuple[OTSData, OTSData],
    *,
    perspective: int,
    decision: DecisionView,
    view_cache: dict[tuple[ReplayMemberId, bool], ReplayPokemonView],
    spatial_events: tuple[EventRecord, ...] = (),
    preview: bool = False,
) -> ReplayBattleView:
    """
    Project one immutable public snapshot into the observation-facing contract.

    view_cache holds one perspective's earlier member views. A view is reused while
    the member's snapshot is the same object and its active flag is unchanged.
    """
    opponent = 1 - perspective
    public = state.public_sides[opponent]
    active_ids = {
        member_id for side in state.sides for member_id in side.active if member_id is not None
    }
    views = {}
    for member in state.members:
        if public is not None and member.member_id.side.side_index == opponent:
            continue
        views[member.member_id] = _cached_pokemon_view(
            member, perspective, member.member_id in active_ids, view_cache
        )
    public_active = None
    if public is not None:
        public_active = tuple(
            None if member is None else _cached_pokemon_view(member, perspective, True, view_cache)
            for member in public.active
        )
        for member in public.members:
            active_view = next(
                (view for view in public_active if view is not None and view.state is member), None
            )
            views[member.member_id] = active_view or _cached_pokemon_view(
                member, perspective, False, view_cache
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
        public_active
        if side_index == opponent and public_active is not None
        else tuple(views[member_id] if member_id is not None else None for member_id in side.active)
        for side_index, side in enumerate(state.sides)
    )
    own_active = (active[perspective][0], active[perspective][1])
    opponent_active = (active[opponent][0], active[opponent][1])
    switches = tuple(
        view
        for member_id, view in views.items()
        if member_id.side.side_index == perspective
        and member_id not in state.sides[perspective].active
        and view.selected_in_teampreview is True
        and not view.fainted
    )
    return ReplayBattleView(
        team=teams[perspective],
        opponent_team=teams[opponent],
        active_pokemon=own_active,
        opponent_active_pokemon=opponent_active,
        available_switches=(switches, switches),
        can_mega_evolve=(decision.slots[0].can_mega, decision.slots[1].can_mega),
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
        spatial_events=spatial_events,
    )


def _spatial_records(
    document: ReplayDocument,
    hp_before: Mapping[int, float | None],
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
            recorder.apply_line(
                document.protocol_lines[line_index].parts,
                tokenizer,
                lambda _identifier, line_index=line_index: hp_before.get(line_index),
            )
        records[start] = recorder.consume()
        cursor = start
    return records


def _projection_snapshot_lines(
    events: tuple[ResolvedProtocolEvent, ...],
    windows: tuple[DecisionWindow, ...],
) -> frozenset[int]:
    lines = {
        len(events) - 1,
        *(
            window.start_line_index
            if window.decision_type is DecisionType.TEAM_PREVIEW
            else window.start_line_index - 1
            for window in windows
            if window.is_policy_request
        ),
    }
    return frozenset(line for line in lines if line >= 0)


def project_replay_perspectives(
    document: ReplayDocument,
    state: ReconstructedReplayState,
    decisions: tuple[DecisionReconstruction, DecisionReconstruction],
) -> tuple[ProjectedPerspective, ProjectedPerspective]:
    """Build p1 and p2 views from one shared immutable trace."""
    snapshots_by_line = {snapshot.line_index: snapshot for snapshot in state.snapshots}
    # Each player's interval closes only at that player's own decisions; a
    # waiting side keeps accumulating until it next chooses. The final interval
    # runs to the last line.
    final_start = len(document.protocol_lines)
    spatial = tuple(
        _spatial_records(
            document,
            state.hp_before,
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
        view_cache: dict[tuple[ReplayMemberId, bool], ReplayPokemonView] = {}
        for decision, decision_view in zip(
            reconstruction.decisions, reconstruction.views, strict=True
        ):
            preview = decision.decision_type is DecisionType.TEAM_PREVIEW
            if preview:
                snapshot = snapshots_by_line[decision.pre_line_index]
            else:
                snapshot = snapshots_by_line[decision.pre_line_index - 1]
            projected_snapshots.append(
                ProjectedSnapshot(
                    decision.decision_index,
                    decision.pre_line_index,
                    decision.post_line_index,
                    project_battle_view(
                        snapshot,
                        document.ots,
                        perspective=perspective,
                        decision=decision_view,
                        spatial_events=spatial[perspective][decision.pre_line_index],
                        view_cache=view_cache,
                        preview=preview,
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
                perspective,
                tuple(projected_snapshots),
                reconstruction.decisions,
                project_battle_view(
                    snapshots_by_line[final_start - 1],
                    document.ots,
                    perspective=perspective,
                    decision=GAME_END_DECISION,
                    spatial_events=spatial[perspective][final_start],
                    view_cache=view_cache,
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
