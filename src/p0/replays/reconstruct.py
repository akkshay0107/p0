"""Pure player-relative replay reconstruction from protocol lines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from p0.battle.actions import encode_team_pair
from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    BattleEvent,
    EventResolver,
    RawBattleEvent,
    build_raw_event,
    get_hp_fraction,
    parse_events,
)
from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import FixtureBattleView
from p0.replays.evidence import (
    EvidenceRequest,
    ObservedAction,
    extract_action_evidence,
)
from p0.replays.protocol import ReplayDocument
from p0.replays.schema import (
    DecisionRecord,
    DecisionType,
    GameRecord,
    ReplayDiagnostics,
)
from p0.teams.stat_points import (
    BaseStats,
    ImputationInput,
    PrecomputedStats,
    StatPoints,
    calculate_stats,
    impute_candidates,
    select_candidate,
)


def normalize_id(value: str) -> str:
    """Normalize a protocol identifier without importing runtime team adapters."""
    return "".join(character for character in value.casefold() if character.isalnum())


class _ReplayResolver(EventResolver):
    """Resolve known-looking names without coupling replay code to model vocabularies."""

    def id_for(self, table: str, name: str | None) -> int:
        return 0 if not name else 1

    def effect_id_for(self, table: str, name: str | None) -> int:
        return self.id_for(table, name)

    def resolve(self, table: str, name: str | None) -> tuple[int, str]:
        return (0, "known_none") if not name else (1, "known")


@dataclass(frozen=True, slots=True)
class ReplayPokemon:
    species: str
    moves: tuple[str, ...] = ()
    current_hp_fraction: float = 1.0
    fainted: bool = False
    revealed: bool = True
    selected_in_teampreview: bool = False
    ability: str | None = None
    item: str | None = None
    status: Any = None
    boosts: Mapping[str, int] = field(default_factory=dict)
    level: int | None = 50

    @property
    def base_species(self) -> str:
        return self.species

    @property
    def type_1(self) -> Any:
        return None

    @property
    def type_2(self) -> Any:
        return None

    @property
    def base_stats(self) -> Mapping[str, int]:
        return {}

    @property
    def stats(self) -> Mapping[str, int | None]:
        return {}

    @property
    def protect_counter(self) -> int:
        return 0

    @property
    def first_turn(self) -> bool:
        return False

    @property
    def weight(self) -> float:
        return 0.0

    @property
    def effects(self) -> Mapping[Any, int]:
        return {}

    @property
    def status_counter(self) -> int:
        return 0

    @property
    def preparing(self) -> Any:
        return None

    @property
    def last_move(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ReconstructedSnapshot:
    """Pre-decision state and events produced by the following protocol segment."""

    decision_index: int
    turn: int
    pre_line_index: int
    post_line_index: int
    view: FixtureBattleView
    events: tuple[BattleEvent, ...]
    raw_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconstructedPerspective:
    game_id: str
    player: int
    snapshots: tuple[ReconstructedSnapshot, ...]
    decisions: tuple[DecisionRecord, ...]
    diagnostics: ReplayDiagnostics

    def to_game_record(self, *, series_id: str | None = None, game_number: int = 1) -> GameRecord:
        outcome = self._outcome
        return GameRecord(
            game_id=self.game_id,
            series_id=series_id or self.game_id,
            game_number=game_number,
            protocol_lines=tuple(
                snapshot_line for snapshot in self.snapshots for snapshot_line in snapshot.raw_lines
            ),
            ots_payloads=self._ots_payloads,
            winner=outcome.winner,
            end_reason=outcome.end_reason,
            turns=outcome.turns,
            decisions=self.decisions,
            diagnostics=self.diagnostics,
        )

    _outcome: Any = None
    _ots_payloads: tuple[str, str] = ("", "")


@dataclass(frozen=True, slots=True)
class StatPointEstimate:
    """A causal spread estimate attached to an OTS member."""

    side: int
    species: str
    provenance: str
    points: StatPoints
    precomputed: PrecomputedStats | None
    confidence: float


def impute_stat_points(
    document: ReplayDocument,
    *,
    dex: Mapping[str, Any],
    seed: int = 0,
) -> tuple[StatPointEstimate, ...]:
    """Seed a legal public-spread estimate, or return explicit UNKNOWN values."""
    species_entries = dex.get("species", ())
    by_id = {
        str(entry.get("id", entry.get("name", ""))).casefold(): entry
        for entry in species_entries
        if isinstance(entry, Mapping)
    }
    move_entries = dex.get("moves", ())
    move_categories = {
        str(entry.get("id", entry.get("name", ""))).casefold(): str(entry.get("category", ""))
        for entry in move_entries
        if isinstance(entry, Mapping)
    }
    estimates: list[StatPointEstimate] = []
    for side, ots in enumerate(document.ots):
        for index, species in enumerate(ots.revealed_species):
            details = ots.revealed_details.get(species, {})
            entry = by_id.get(normalize_id(species).casefold())
            base_mapping = entry.get("baseStats") if isinstance(entry, Mapping) else None
            if not isinstance(base_mapping, Mapping):
                estimates.append(
                    StatPointEstimate(side, species, "UNKNOWN", StatPoints(), None, 0.0)
                )
                continue
            moves = tuple(str(move) for move in details.get("moves", ()) if isinstance(move, str))
            categories = tuple(
                move_categories.get(normalize_id(move).casefold(), "") for move in moves
            )
            level_value = details.get("level", 50)
            if isinstance(level_value, str):
                level_value = level_value.lstrip("L")
            try:
                level = int(level_value)
            except (TypeError, ValueError):
                estimates.append(
                    StatPointEstimate(side, species, "UNKNOWN", StatPoints(), None, 0.0)
                )
                continue
            value = ImputationInput(
                species=species,
                nature=str(details.get("nature", "serious")),
                item=str(details.get("item", "")),
                ability=str(details.get("ability", "")),
                moves=moves,
                move_categories=categories,
                base_stats=BaseStats.from_mapping(base_mapping),
                level=level,
            )
            candidates = impute_candidates(value)
            candidate = select_candidate(value, seed + side * 1009 + index)
            stats = PrecomputedStats(
                calculate_stats(value.base_stats, candidate.points, value.nature, value.level)
            )
            total_weight = sum(item.weight for item in candidates)
            estimates.append(
                StatPointEstimate(
                    side,
                    species,
                    "IMPUTED",
                    candidate.points,
                    stats,
                    candidate.weight / max(1, total_weight),
                )
            )
    return tuple(estimates)


class _ReplayState:
    def __init__(self, document: ReplayDocument):
        self.turn = 0
        self.used_mega = [False, False]
        self.active: list[list[ReplayPokemon | None]] = [[None, None], [None, None]]
        self.teams: list[list[ReplayPokemon]] = [
            [
                ReplayPokemon(
                    species,
                    tuple(
                        str(move)
                        for move in document.ots[side]
                        .revealed_details.get(species, {})
                        .get("moves", ())
                        if isinstance(move, str)
                    ),
                )
                for species in document.ots[side].revealed_species
            ]
            for side in (0, 1)
        ]
        self.hp: dict[str, float] = {}
        self.fainted: set[str] = set()

    def clone_active(self) -> list[list[ReplayPokemon | None]]:
        return deepcopy(self.active)

    def pokemon_for(self, side: int, species: str) -> ReplayPokemon:
        normalized = normalize_id(species)
        for pokemon in self.teams[side]:
            if normalize_id(pokemon.species) == normalized:
                return pokemon
        pokemon = ReplayPokemon(species)
        if len(self.teams[side]) < 6:
            self.teams[side].append(pokemon)
        return pokemon

    def _replace_active(self, side: int, old: ReplayPokemon, **changes: Any) -> ReplayPokemon:
        updated = replace(old, **changes)
        self.teams[side] = [updated if pokemon is old else pokemon for pokemon in self.teams[side]]
        self.active[side] = [
            updated if pokemon is old else pokemon for pokemon in self.active[side]
        ]
        return updated

    @staticmethod
    def _endpoint(identifier: str) -> tuple[int, int] | None:
        if len(identifier) < 3 or identifier[:2] not in ("p1", "p2"):
            return None
        slot = ord(identifier[2].lower()) - ord("a")
        return (int(identifier[1]) - 1, slot) if slot in (0, 1) else None

    def hp_for(self, identifier: str) -> float | None:
        return self.hp.get(identifier.split(":", 1)[0])

    def apply(self, parts: Sequence[str]) -> None:
        if len(parts) < 2:
            return
        tag = parts[1]
        if tag == "turn" and len(parts) >= 3 and parts[2].isdigit():
            self.turn = int(parts[2])
            return
        if tag in ("switch", "drag") and len(parts) >= 3:
            endpoint = self._endpoint(parts[2])
            if endpoint is None:
                return
            side, slot = endpoint
            species = parts[2].split(":", 1)[1].strip() if ":" in parts[2] else "unknown"
            pokemon = self.pokemon_for(side, species)
            if pokemon.fainted:
                pokemon = self._replace_active(side, pokemon, fainted=False)
            self.active[side][slot] = pokemon
            self.hp[parts[2].split(":", 1)[0]] = 1.0
            return
        if tag == "faint" and len(parts) >= 3:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                side, slot = endpoint
                pokemon = self.active[side][slot]
                if pokemon is not None:
                    self._replace_active(side, pokemon, fainted=True, current_hp_fraction=0.0)
                    self.fainted.add(parts[2].split(":", 1)[0])
                self.active[side][slot] = None
            return
        if tag in ("-damage", "-heal") and len(parts) >= 4:
            identifier = parts[2].split(":", 1)[0]
            fraction = get_hp_fraction(parts[3])
            self.hp[identifier] = fraction
            endpoint = self._endpoint(parts[2])
            current = None if endpoint is None else self.active[endpoint[0]][endpoint[1]]
            if endpoint is not None and current is not None:
                self._replace_active(
                    endpoint[0], current, current_hp_fraction=fraction, fainted=fraction == 0.0
                )
            return
        if tag == "-mega" and len(parts) >= 3:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                self.used_mega[endpoint[0]] = True


def _team_mapping(team: Sequence[ReplayPokemon]) -> dict[str, ReplayPokemon]:
    return {f"{index}:{pokemon.species}": pokemon for index, pokemon in enumerate(team)}


def _target_code(actor: str, target: str | None) -> int | None:
    if not target:
        return None
    actor_endpoint = _ReplayState._endpoint(actor)
    target_endpoint = _ReplayState._endpoint(target)
    if actor_endpoint is None or target_endpoint is None:
        return None
    if actor_endpoint == target_endpoint:
        return -2
    if actor_endpoint[0] == target_endpoint[0]:
        return -1
    return target_endpoint[1]


def _move_slot(state: _ReplayState, side: int, slot: int, move: str) -> int | None:
    active = state.active[side][slot]
    if active is None:
        return None
    normalized = normalize_id(move)
    try:
        return tuple(normalize_id(value) for value in active.moves).index(normalized)
    except ValueError:
        return None


def _observed_actions(
    state: _ReplayState,
    lines: Sequence[Any],
    perspective: int,
    diagnostics: Counter[str],
) -> tuple[ObservedAction | None, ObservedAction | None, tuple[str, ...]]:
    observed: list[ObservedAction | None] = [None, None]
    tags: list[str] = []
    mega_slots: set[tuple[int, int]] = set()
    for line in lines:
        parts = line.parts
        if len(parts) >= 3 and parts[1] == "-mega":
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is not None:
                mega_slots.add(endpoint)
        if len(parts) < 3:
            continue
        if parts[1] == "move":
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is None or endpoint[0] != perspective:
                continue
            slot = endpoint[1]
            generated = any(part.startswith("[from]") for part in parts[5:])
            if generated:
                observed[slot] = ObservedAction(None, exact=False, tag="externally_generated_move")
                diagnostics["externally_generated_move"] += 1
                tags.append("externally_generated_move")
                continue
            if observed[slot] is not None:
                observed[slot] = ObservedAction(None, exact=False, tag="multiple_moves_same_slot")
                diagnostics["externally_generated_move"] += 1
                continue
            move = parts[3] if len(parts) >= 4 else ""
            forced = normalize_id(move) in {"struggle", "recharge"}
            if forced:
                observed[slot] = ObservedAction(
                    47 if (endpoint in mega_slots) else 48, tag="forced_move"
                )
                continue
            target = parts[4] if len(parts) >= 5 else None
            move_slot = _move_slot(state, perspective, slot, move)
            target_code = _target_code(parts[2], target)
            if move_slot is None or target_code is None:
                diagnostics["move_slot_or_target_unknown"] += 1
                observed[slot] = ObservedAction(
                    None, exact=False, tag="move_slot_or_target_unknown"
                )
            else:
                action = 7 + move_slot * 5 + target_code + 2
                if endpoint in mega_slots:
                    action += 20
                observed[slot] = ObservedAction(action, tag="move")
        elif parts[1] == "switch":
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is None or endpoint[0] != perspective:
                continue
            species = parts[2].split(":", 1)[1].strip() if ":" in parts[2] else ""
            try:
                action = 1 + next(
                    index
                    for index, pokemon in enumerate(state.teams[perspective])
                    if normalize_id(pokemon.species) == normalize_id(species)
                )
                observed[endpoint[1]] = ObservedAction(action, tag="switch")
            except StopIteration:
                observed[endpoint[1]] = ObservedAction(None, exact=False, tag="switch_slot_unknown")
                diagnostics["switch_slot_unknown"] += 1
        elif parts[1] == "cant":
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is not None and endpoint[0] == perspective:
                observed[endpoint[1]] = ObservedAction(None, exact=False, tag="cant")
                tags.append("cant")
    return observed[0], observed[1], tuple(dict.fromkeys(tags))


def _preview_actions(
    state: _ReplayState, lines: Sequence[Any], perspective: int
) -> tuple[ObservedAction | None, ObservedAction | None, tuple[str, ...]]:
    leads: list[str] = []
    for line in lines:
        if len(line.parts) < 3 or line.parts[1] != "switch":
            continue
        endpoint = _ReplayState._endpoint(line.parts[2])
        if endpoint is None or endpoint[0] != perspective:
            continue
        species = line.parts[2].split(":", 1)[1].strip() if ":" in line.parts[2] else ""
        if species and species not in leads:
            leads.append(species)
    if len(leads) != 2:
        return None, None, ("preview_leads_unknown",)
    try:
        lead_indices = tuple(
            next(
                index
                for index, pokemon in enumerate(state.teams[perspective])
                if normalize_id(pokemon.species) == normalize_id(species)
            )
            for species in leads
        )
    except StopIteration:
        return None, None, ("preview_roster_unknown",)
    if len(set(lead_indices)) != 2:
        return None, None, ("preview_duplicate_lead",)
    first = encode_team_pair(*sorted(lead_indices), team_size=len(state.teams[perspective]))
    alternatives = tuple(
        encode_team_pair(first_index, second_index, team_size=len(state.teams[perspective]))
        for first_index in range(len(state.teams[perspective]))
        for second_index in range(first_index + 1, len(state.teams[perspective]))
        if first_index not in lead_indices and second_index not in lead_indices
    )
    return (
        ObservedAction(first, tag="preview_leads"),
        ObservedAction(alternatives=alternatives, exact=False, tag="preview_reserves_unknown"),
        ("preview_reserves_unknown",),
    )


def _segments(document: ReplayDocument) -> tuple[tuple[int, int, DecisionType], ...]:
    turns = [
        line.index
        for line in document.protocol_lines
        if len(line.parts) > 1 and line.parts[1] == "turn"
    ]
    preview = [
        line.index
        for line in document.protocol_lines
        if len(line.parts) > 1 and line.parts[1] == "teampreview"
    ]
    segments: list[tuple[int, int, DecisionType]] = []
    if preview and turns and turns[0] > 0:
        segments.append((0, turns[0], DecisionType.TEAM_PREVIEW))
    elif preview and not turns:
        segments.append((0, len(document.protocol_lines), DecisionType.TEAM_PREVIEW))
    for index, start in enumerate(turns):
        end = turns[index + 1] if index + 1 < len(turns) else len(document.protocol_lines)
        boundaries = [start]
        saw_action = False
        for line in document.protocol_lines[start:end]:
            if len(line.parts) > 1 and line.parts[1] in {"move", "cant", "switch", "drag"}:
                if line.parts[1] in {"switch", "drag"} and saw_action:
                    boundaries.append(line.index)
                    saw_action = False
                else:
                    saw_action = True
        boundaries.append(end)
        segments.extend(
            (left, right, DecisionType.TURN)
            for left, right in zip(boundaries, boundaries[1:])
            if left < right
        )
    if not segments and not preview:
        segments.append((0, len(document.protocol_lines), DecisionType.TURN))
    return tuple(segments)


def _view(state: _ReplayState, perspective: int, *, preview: bool) -> FixtureBattleView:
    opponent = 1 - perspective
    own_active = tuple(state.active[perspective])
    opponent_active = tuple(state.active[opponent])
    slots = []
    for pokemon in own_active:
        moves = () if pokemon is None else tuple((tuple((-2, -1, 0, 1)),) for _ in pokemon.moves)
        slots.append(
            SlotDecision(
                switch_slots=tuple(
                    index
                    for index, candidate in enumerate(state.teams[perspective])
                    if candidate is not pokemon
                ),
                move_targets=tuple(targets[0] for targets in moves),
                active=pokemon is not None,
                force_switch=pokemon is None,
                can_mega=not state.used_mega[perspective],
            )
        )
    decision = DecisionView(
        slots=(slots[0], slots[1]),
        team_preview=preview,
        team_size=max(1, len(state.teams[perspective])),
    )
    return FixtureBattleView(
        team=_team_mapping(state.teams[perspective]),
        opponent_team=_team_mapping(state.teams[opponent]),
        active_pokemon=own_active,
        opponent_active_pokemon=opponent_active,
        available_moves=tuple(
            tuple(pokemon.moves) if pokemon is not None else () for pokemon in own_active
        ),
        available_switches=tuple(tuple(state.teams[perspective]) for _ in (0, 1)),
        can_mega_evolve=(not state.used_mega[perspective],) * 2,
        force_switch=(own_active[0] is None, own_active[1] is None),
        trapped=(False, False),
        maybe_trapped=(False, False),
        teampreview=preview,
        player_role=f"p{perspective + 1}",
        wait=False,
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=state.turn,
        used_mega_evolve=state.used_mega[perspective],
        opponent_used_mega_evolve=state.used_mega[opponent],
        decision=decision,
        identifiers={},
    )


def _events_and_update(
    state: _ReplayState, lines: Sequence[Any], diagnostics: Counter[str]
) -> tuple[BattleEvent, ...]:
    raw_events: list[RawBattleEvent] = []
    for line in lines:
        raw_events.append(build_raw_event(line.parts, state.hp_for))
    before = Counter(EVENT_DIAGNOSTICS)
    try:
        events = parse_events(raw_events, _ReplayResolver())
    except (TypeError, ValueError, IndexError):
        diagnostics["parser_errors"] += 1
        diagnostics["parse_error_lines"] += len(lines)
        events = []
    after = Counter(EVENT_DIAGNOSTICS)
    for key, count in after.items():
        diagnostics[key] += max(0, count - before[key])
    for line in lines:
        try:
            state.apply(line.parts)
        except (TypeError, ValueError, IndexError):
            diagnostics["state_update_errors"] += 1
    return tuple(events)


def reconstruct_perspective(
    document: ReplayDocument,
    *,
    perspective: int,
    max_candidates: int = 256,
) -> ReconstructedPerspective:
    """Build pre-decision player-relative views while enforcing causal cutoffs."""
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    state = _ReplayState(document)
    counters: Counter[str] = Counter()
    snapshots: list[ReconstructedSnapshot] = []
    decisions: list[DecisionRecord] = []
    for decision_index, (start, end, decision_type) in enumerate(_segments(document)):
        lines = document.protocol_lines[start:end]
        view = _view(state, perspective, preview=decision_type is DecisionType.TEAM_PREVIEW)
        observed = (
            _preview_actions(state, lines, perspective)
            if decision_type is DecisionType.TEAM_PREVIEW
            else _observed_actions(state, lines, perspective, counters)
        )
        tags = list(observed[2])
        if decision_type is DecisionType.TEAM_PREVIEW:
            tags.append("preview_selection_not_public")
        if any(action is not None and action.tag == "switch" for action in observed[:2]):
            if any(action is not None and action.tag == "move" for action in observed[:2]):
                decision_type = DecisionType.PIVOT_SWITCH
            elif any(
                view.decision.slots[index].force_switch
                for index in (0, 1)
                if observed[index] is not None
            ):
                decision_type = DecisionType.FORCED_SWITCH
        request = EvidenceRequest(
            view=view.decision,
            slots=(observed[0], observed[1]),
            tags=tuple(tags),
            max_candidates=max_candidates,
            unknown=False,
        )
        evidence = extract_action_evidence(request)
        decisions.append(
            DecisionRecord(
                decision_index=decision_index,
                player=perspective,
                decision_type=decision_type,
                pre_line_index=start,
                post_line_index=end,
                evidence=evidence,
            )
        )
        events = _events_and_update(state, lines, counters)
        snapshots.append(
            ReconstructedSnapshot(
                decision_index=decision_index,
                turn=state.turn
                if decision_type is not DecisionType.TURN
                else (lines[0].turn or state.turn),
                pre_line_index=start,
                post_line_index=end,
                view=view,
                events=events,
                raw_lines=tuple(line.raw for line in lines),
            )
        )
    diagnostics = ReplayDiagnostics(dict(counters), ())
    return ReconstructedPerspective(
        game_id=document.metadata.replay_id,
        player=perspective,
        snapshots=tuple(snapshots),
        decisions=tuple(decisions),
        diagnostics=diagnostics,
        _outcome=document.outcome,
        _ots_payloads=(document.ots[0].raw_payload, document.ots[1].raw_payload),
    )


def reconstruct_both(
    document: ReplayDocument,
    *,
    max_candidates: int = 256,
) -> tuple[ReconstructedPerspective, ReconstructedPerspective]:
    return (
        reconstruct_perspective(document, perspective=0, max_candidates=max_candidates),
        reconstruct_perspective(document, perspective=1, max_candidates=max_candidates),
    )


reconstruct_game = reconstruct_perspective


__all__ = [
    "ReplayPokemon",
    "ReconstructedPerspective",
    "ReconstructedSnapshot",
    "reconstruct_both",
    "reconstruct_game",
    "reconstruct_perspective",
]
