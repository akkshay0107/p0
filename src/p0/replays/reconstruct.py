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
    RawBattleEvent,
    build_raw_event,
    get_hp_fraction,
    parse_events,
)
from p0.battle.legality import DecisionView, SlotDecision, legal_actions
from p0.battle.views import FixtureBattleView
from p0.model.tokenizer import tokenizer
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


def _species_base_stats(dex: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Index exact species and documented form aliases by normalized id."""
    index: dict[str, dict[str, int]] = {}
    entries = tuple(entry for entry in dex.get("species", ()) if isinstance(entry, Mapping))
    for entry in entries:
        base_stats = entry.get("baseStats")
        if not isinstance(base_stats, Mapping):
            continue
        stats = {str(key): int(value) for key, value in base_stats.items()}
        for value in (entry.get("id"), entry.get("name")):
            if isinstance(value, str) and value:
                index[normalize_id(value)] = stats
    for entry in entries:
        base_stats = entry.get("baseStats")
        if not isinstance(base_stats, Mapping):
            continue
        stats = {str(key): int(value) for key, value in base_stats.items()}
        aliases = (*entry.get("formeOrder", ()), *entry.get("otherFormes", ()))
        for value in aliases:
            if isinstance(value, str) and value:
                index.setdefault(normalize_id(value), stats)
    return index


def _species_identity_index(dex: Mapping[str, Any]) -> dict[str, str]:
    """Map battle forms to their species-clause/base-species identity."""
    aliases: dict[str, str] = {}
    for entry in dex.get("species", ()):
        if not isinstance(entry, Mapping):
            continue
        base = entry.get("baseSpecies", entry.get("name", entry.get("id", "")))
        if not isinstance(base, str) or not base:
            continue
        identity = normalize_id(base)
        values = (
            entry.get("id"),
            entry.get("name"),
            entry.get("baseSpecies"),
            *entry.get("formeOrder", ()),
            *entry.get("otherFormes", ()),
        )
        for value in values:
            if isinstance(value, str) and value:
                aliases.setdefault(normalize_id(value), identity)
    return aliases


_BASE_STATS_CACHE: dict[int, dict[str, dict[str, int]]] = {}


def _get_base_stats_index(dex: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    dex_id = id(dex)
    if dex_id not in _BASE_STATS_CACHE:
        _BASE_STATS_CACHE[dex_id] = _species_base_stats(dex)
    return _BASE_STATS_CACHE[dex_id]


def _make_replay_pokemon(
    species: str,
    base_stats_index: Mapping[str, Mapping[str, int]],
    moves: Mapping[str, ReplayMove] | None = None,
    ability: str | None = None,
    item: str | None = None,
    nature: str | None = None,
) -> ReplayPokemon:
    normalized = normalize_id(species)
    base_stats = base_stats_index.get(normalized, {})
    return ReplayPokemon(
        species=species,
        moves=moves if moves is not None else {},
        ability=ability,
        item=item,
        nature=nature,
        base_stats_data=base_stats,
    )


def _zero_boosts() -> dict[str, int]:
    return {stat: 0 for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")}


@dataclass(frozen=True, slots=True)
class ReplayName:
    """Enum-like placeholder used when OTS omits move mechanics."""

    name: str


@dataclass(frozen=True, slots=True)
class ReplayMove:
    """Minimal move view required by the offline observation builder."""

    id: str
    type: ReplayName = ReplayName("unknown")
    category: ReplayName = ReplayName("unknown")
    current_pp: int | None = None
    max_pp: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ReplayPokemon:
    species: str
    moves: Mapping[str, ReplayMove] = field(default_factory=dict)
    current_hp_fraction: float = 1.0
    fainted: bool = False
    revealed: bool = True
    selected_in_teampreview: bool = False
    ability: str | None = None
    item: str | None = None
    nature: str | None = None
    base_stats_data: Mapping[str, int] = field(default_factory=dict)
    status: Any = None
    boosts: Mapping[str, int] = field(default_factory=_zero_boosts)
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
        return self.base_stats_data

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
        return 0.0

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
        normalize_id(str(entry.get("id", entry.get("name", "")))): entry
        for entry in species_entries
        if isinstance(entry, Mapping)
    }
    for entry in species_entries:
        if not isinstance(entry, Mapping):
            continue
        base = entry.get("baseSpecies", entry.get("name", entry.get("id", "")))
        base_entry = by_id.get(normalize_id(str(base))) if base else None
        fallback = base_entry if isinstance(base_entry, Mapping) else entry
        for value in (*entry.get("formeOrder", ()), *entry.get("otherFormes", ())):
            if isinstance(value, str) and value:
                by_id.setdefault(normalize_id(value), fallback)
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


def _replay_moves(names: tuple[str, ...]) -> dict[str, ReplayMove]:
    return {name: ReplayMove(name) for name in names}


class _ReplayState:
    def __init__(self, document: ReplayDocument, dex: Mapping[str, Any]):
        self.turn = 0
        self.used_mega = [False, False]
        self.active: list[list[ReplayPokemon | None]] = [[None, None], [None, None]]
        self.base_stats_index = _get_base_stats_index(dex)
        self.species_identity_index = _species_identity_index(dex)
        self.teams: list[list[ReplayPokemon]] = []
        for side in (0, 1):
            side_teams = []
            for species in document.ots[side].revealed_species:
                details = document.ots[side].revealed_details.get(species, {})
                moves_tuple = tuple(
                    str(move) for move in details.get("moves", ()) if isinstance(move, str)
                )
                ability = str(details.get("ability", "")) or None
                item = str(details.get("item", "")) or None
                nature = str(details.get("nature", "")) or None
                pokemon = _make_replay_pokemon(
                    species=species,
                    base_stats_index=self.base_stats_index,
                    moves=_replay_moves(moves_tuple),
                    ability=ability,
                    item=item,
                    nature=nature,
                )
                side_teams.append(pokemon)
            self.teams.append(side_teams)
        self.hp: dict[str, float] = {}
        self.fainted: set[str] = set()
        # A switch can initially identify an Illusion user as another roster
        # member. Keep the pre-effect object so a later ``replace`` line can
        # restore the disguised member before transferring runtime state to
        # the revealed Pokemon.
        self._illusion_baselines: dict[tuple[int, int], ReplayPokemon] = {}
        # Events carry Showdown identifiers while the observation builder
        # indexes concrete Pokemon objects. Keep this mapping across request
        # boundaries so replay event grounding matches the live adapter.
        self.identifiers: dict[str, ReplayPokemon] = {}

    def clone_active(self) -> list[list[ReplayPokemon | None]]:
        return deepcopy(self.active)

    def team_pokemon(self, side: int, species: str) -> ReplayPokemon | None:
        normalized = normalize_id(species)
        for pokemon in self.teams[side]:
            if normalize_id(pokemon.species) == normalized:
                return pokemon
        identity = self.species_identity_index.get(normalized, normalized)
        for pokemon in self.teams[side]:
            pokemon_identity = self.species_identity_index.get(
                normalize_id(pokemon.species), normalize_id(pokemon.species)
            )
            if pokemon_identity == identity:
                return pokemon
        return None

    def pokemon_for(self, side: int, species: str) -> ReplayPokemon:
        existing = self.team_pokemon(side, species)
        if existing is not None:
            return existing
        pokemon = _make_replay_pokemon(species, self.base_stats_index)
        if len(self.teams[side]) < 6:
            self.teams[side].append(pokemon)
        return pokemon

    def _replace_active(self, side: int, old: ReplayPokemon, **changes: Any) -> ReplayPokemon:
        updated = replace(old, **changes)
        self._replace_references(side, old, updated)
        return updated

    def _replace_references(self, side: int, old: ReplayPokemon, updated: ReplayPokemon) -> None:
        self.teams[side] = [updated if pokemon is old else pokemon for pokemon in self.teams[side]]
        self.active[side] = [
            updated if pokemon is old else pokemon for pokemon in self.active[side]
        ]
        for identifier, pokemon in tuple(self.identifiers.items()):
            if pokemon is old:
                self.identifiers[identifier] = updated

    def _canonicalize_active_aliases(self, side: int) -> None:
        """Attach duplicate display aliases to their revealed roster members."""
        for slot, current in enumerate(self.active[side]):
            if current is None or any(pokemon is current for pokemon in self.teams[side]):
                continue
            canonical = self.team_pokemon(side, current.species)
            if canonical is None or any(pokemon is canonical for pokemon in self.active[side]):
                continue
            updated = replace(
                canonical,
                current_hp_fraction=current.current_hp_fraction,
                fainted=current.fainted,
                selected_in_teampreview=True,
                status=current.status,
                boosts=current.boosts,
            )
            self._replace_references(side, canonical, updated)
            self.active[side][slot] = updated
            for identifier, pokemon in tuple(self.identifiers.items()):
                if pokemon is current:
                    self.identifiers[identifier] = updated
            self._illusion_baselines.pop((side, slot), None)

    def _promote_active_illusion_if_duplicate(
        self, side: int, slot: int, displayed_species: str
    ) -> None:
        """Separate a hidden Illusion before a same-species switch enters."""
        current = self.active[side][slot]
        canonical = self.team_pokemon(side, displayed_species)
        if current is None or canonical is None or current is not canonical:
            return
        candidates = [
            pokemon
            for pokemon in self.teams[side]
            if normalize_id(pokemon.ability or "") == "illusion"
            and not pokemon.fainted
            and all(active is not pokemon for active in self.active[side])
        ]
        if len(candidates) != 1:
            return
        alias = replace(
            current,
            fainted=False,
            selected_in_teampreview=True,
        )
        self.active[side][slot] = alias
        self._illusion_baselines[(side, slot)] = alias
        for identifier, pokemon in tuple(self.identifiers.items()):
            if pokemon is current:
                self.identifiers[identifier] = alias

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
            species = _switch_species(parts) or "unknown"
            pokemon = self.pokemon_for(side, species)
            for active_slot, active in enumerate(self.active[side]):
                if active is pokemon:
                    self._promote_active_illusion_if_duplicate(side, active_slot, species)
            hp_fraction = get_hp_fraction(parts[4]) if len(parts) >= 5 else 1.0
            if any(active is pokemon for active in self.active[side]):
                # A duplicate displayed species is the characteristic replay
                # shape of an Illusion masking a roster member that is already
                # active. Keep the alias out of the team list so effects on it
                # cannot mutate the real member in the other active slot.
                pokemon = replace(
                    pokemon,
                    fainted=False,
                    current_hp_fraction=hp_fraction,
                    selected_in_teampreview=True,
                )
            elif (
                pokemon.fainted
                or pokemon.current_hp_fraction != hp_fraction
                or not pokemon.selected_in_teampreview
            ):
                pokemon = self._replace_active(
                    side,
                    pokemon,
                    fainted=False,
                    current_hp_fraction=hp_fraction,
                    selected_in_teampreview=True,
                )
            self.active[side][slot] = pokemon
            self._illusion_baselines[(side, slot)] = pokemon
            endpoint_id = parts[2].split(":", 1)[0]
            self.hp[endpoint_id] = hp_fraction
            self.identifiers[parts[2]] = pokemon
            self.identifiers[endpoint_id] = pokemon
            return
        if tag == "replace" and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is None:
                return
            side, slot = endpoint
            current = self.active[side][slot]
            if current is None:
                return

            # ``replace`` is an Illusion identity transition. Damage and
            # faint lines preceding it were keyed to the displayed alias, so
            # move those runtime fields to the actual roster member while
            # restoring the alias member's pre-Illusion state.
            actual_species = parts[3].split(",", 1)[0].strip()
            actual = self.pokemon_for(side, actual_species)
            baseline = self._illusion_baselines.pop((side, slot), None)
            if baseline is not None and current is not baseline:
                self._replace_references(side, current, baseline)

            hp_fraction = (
                get_hp_fraction(parts[4])
                if len(parts) >= 5
                else current.current_hp_fraction
            )
            actual = self._replace_active(
                side,
                actual,
                current_hp_fraction=hp_fraction,
                fainted=current.fainted or hp_fraction == 0.0,
                selected_in_teampreview=True,
                status=current.status,
                boosts=current.boosts,
            )
            self.active[side][slot] = actual
            endpoint_id = parts[2].split(":", 1)[0]
            self.hp[endpoint_id] = hp_fraction
            self.identifiers[parts[2]] = actual
            self.identifiers[endpoint_id] = actual
            self._canonicalize_active_aliases(side)
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
        if tag in ("-status", "-curestatus") and len(parts) >= 3:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    status = (
                        None if tag == "-curestatus" else (parts[3] if len(parts) >= 4 else None)
                    )
                    self._replace_active(endpoint[0], current, status=status)
            return
        if tag in ("-boost", "-unboost", "-setboost") and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    boosts = dict(current.boosts)
                    stat = parts[3]
                    if tag == "-setboost":
                        boosts[stat] = int(parts[4]) if len(parts) >= 5 else 0
                    else:
                        delta = int(parts[4]) if len(parts) >= 5 else 0
                        boosts[stat] = boosts.get(stat, 0) + (delta if tag == "-boost" else -delta)
                    self._replace_active(endpoint[0], current, boosts=boosts)
            return
        if tag in ("-clearboost", "-clearnegativeboost", "-clearallboost"):
            targets: list[tuple[int, ReplayPokemon]] = []
            if tag == "-clearallboost":
                targets = [
                    (side, pokemon)
                    for side in (0, 1)
                    for pokemon in self.active[side]
                    if pokemon is not None
                ]
            elif len(parts) >= 3:
                endpoint = self._endpoint(parts[2])
                if endpoint is not None:
                    current = self.active[endpoint[0]][endpoint[1]]
                    if current is not None:
                        targets.append((endpoint[0], current))
            for side, current in targets:
                self._replace_active(side, current, boosts=_zero_boosts())
            return
        if tag in ("detailschange", "-formechange") and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    self._replace_active(endpoint[0], current, species=parts[3].split(",", 1)[0])
            return
        if tag == "-ability" and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    self._replace_active(endpoint[0], current, ability=parts[3])
            return
        if (tag == "-enditem" and len(parts) >= 3) or (tag == "-item" and len(parts) >= 4):
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    self._replace_active(
                        endpoint[0],
                        current,
                        item=None if tag == "-enditem" else parts[3],
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


def _move_slot(
    state: _ReplayState,
    side: int,
    slot: int,
    move: str,
    *,
    species: str | None = None,
) -> int | None:
    active = state.active[side][slot]
    if species is not None:
        active = state.team_pokemon(side, species)
    if active is None:
        return None
    normalized = normalize_id(move)
    try:
        return tuple(normalize_id(value) for value in active.moves).index(normalized)
    except ValueError:
        return None


def _switch_species(parts: Sequence[str]) -> str:
    """Return the species field from a Showdown switch command.

    Showdown encodes the display nickname in the endpoint field and the actual
    species (including forms) in the following field. Reconstruction needs the
    latter to map the event back to the OTS roster.
    """
    if len(parts) < 4:
        return ""
    return parts[3].split(",", 1)[0].strip()


def _is_choice_switch(parts: Sequence[str]) -> bool:
    """Whether a switch line represents a submitted order.

    Showdown also emits ``switch`` for automatic pivots such as Parting Shot
    and U-turn. Those lines update state and produce events, but they are not
    decisions visible to the environment.
    """
    return (
        len(parts) >= 3
        and parts[1] == "switch"
        and not any(part.startswith("[from]") for part in parts[4:])
    )


def _is_choice_move(parts: Sequence[str]) -> bool:
    """Whether a move line represents a submitted order rather than a proc."""
    return (
        len(parts) >= 4
        and parts[1] == "move"
        and not any(part.startswith("[from]") for part in parts[5:])
    )


def _animation_targets(lines: Sequence[Any]) -> dict[tuple[int, int, str], str]:
    """Collect targets from animation lines paired with targetless move lines.

    Showdown deliberately clears the target field on a ``move`` line when it
    appends ``[still]``.  Some moves, notably Electro Shot in rain, then emit a
    separate ``-anim`` line containing the target.  Keep the first target for
    each actor/move pair; later animation lines can describe additional hits
    rather than a new choice.
    """
    targets: dict[tuple[int, int, str], str] = {}
    for line in lines:
        parts = line.parts
        if len(parts) < 5 or parts[1] != "-anim":
            continue
        endpoint = _ReplayState._endpoint(parts[2])
        target = parts[4]
        if endpoint is None or not target or _ReplayState._endpoint(target) is None:
            continue
        key = (endpoint[0], endpoint[1], normalize_id(parts[3]))
        targets.setdefault(key, target)
    return targets


def _illusion_species_by_line(lines: Sequence[Any]) -> dict[tuple[int, int, int], str]:
    """Resolve action labels emitted while an Illusion is still disguised.

    The replay can split the alias ``switch`` and its later ``replace`` across
    request/turn chunks. The future ``replace`` is valid supervision for the
    observed action, even though it must not be applied to the pre-decision
    state until its protocol line is reached.
    """
    pending: dict[tuple[int, int], int] = {}
    resolved: dict[tuple[int, int, int], str] = {}
    for line in lines:
        parts = line.parts
        if len(parts) < 3:
            continue
        endpoint = _ReplayState._endpoint(parts[2])
        if endpoint is None:
            continue
        if parts[1] in ("switch", "drag"):
            pending[endpoint] = line.index
        elif parts[1] == "replace" and len(parts) >= 4:
            start = pending.pop(endpoint, None)
            if start is None:
                continue
            species = parts[3].split(",", 1)[0].strip()
            for index in range(start, line.index):
                resolved[(index, endpoint[0], endpoint[1])] = species
    return resolved


def _infer_illusion_active_species(
    state: _ReplayState,
    lines: Sequence[Any],
    perspective: int,
    illusion_species_by_line: Mapping[tuple[int, int, int], str],
) -> dict[tuple[int, int], str]:
    """Recover an earlier undisclosed Illusion before a displayed switch.

    A same-species switch can mean either that an earlier Illusion is leaving
    and the real roster member is entering, or that a new Illusion is entering
    now. The latter has a future ``replace`` entry for the current switch and
    must not change this segment's pre-decision view.
    """
    inferred: dict[tuple[int, int], str] = {}
    for line in lines:
        parts = line.parts
        if not _is_choice_switch(parts):
            continue
        endpoint = _ReplayState._endpoint(parts[2])
        if endpoint is None or endpoint[0] != perspective:
            continue
        if (line.index, endpoint[0], endpoint[1]) in illusion_species_by_line:
            continue
        current = state.active[endpoint[0]][endpoint[1]]
        displayed_species = _switch_species(parts)
        if current is None or not displayed_species:
            continue
        current_identity = state.species_identity_index.get(
            normalize_id(current.species), normalize_id(current.species)
        )
        displayed_identity = state.species_identity_index.get(
            normalize_id(displayed_species), normalize_id(displayed_species)
        )
        if current_identity != displayed_identity:
            continue
        candidates = [
            pokemon
            for pokemon in state.teams[perspective]
            if normalize_id(pokemon.ability or "") == "illusion"
            and not pokemon.fainted
            and all(active is not pokemon for active in state.active[perspective])
        ]
        if len(candidates) == 1:
            inferred[endpoint] = candidates[0].species
    return inferred


def _observed_actions(
    state: _ReplayState,
    lines: Sequence[Any],
    perspective: int,
    diagnostics: Counter[str],
    illusion_species_by_line: Mapping[tuple[int, int, int], str] | None = None,
) -> tuple[ObservedAction | None, ObservedAction | None, tuple[str, ...]]:
    observed: list[ObservedAction | None] = [None, None]
    tags: list[str] = []
    mega_slots: set[tuple[int, int]] = set()
    animation_targets = _animation_targets(lines)
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
            if not _is_choice_move(parts):
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
            target_tag = "move"
            if not target:
                target = animation_targets.get(
                    (endpoint[0], endpoint[1], normalize_id(move))
                )
                if target:
                    target_tag = "move_anim_target"
            effective_species = illusion_species_by_line.get(
                (line.index, endpoint[0], endpoint[1])
            ) if illusion_species_by_line is not None else None
            move_slot = _move_slot(
                state,
                perspective,
                slot,
                move,
                species=effective_species,
            )
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
                observed[slot] = ObservedAction(action, tag=target_tag)
        elif _is_choice_switch(parts):
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is None or endpoint[0] != perspective:
                continue
            species = illusion_species_by_line.get(
                (line.index, endpoint[0], endpoint[1]), _switch_species(parts)
            ) if illusion_species_by_line is not None else _switch_species(parts)
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
        species = _switch_species(line.parts)
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
            # ``cant`` is an outcome of a submitted move (flinch, paralysis,
            # sleep, Armor Tail, ...), and ``drag`` is an automatic battle
            # effect. Neither creates a new request in poke-env. A switch
            # after an action is a forced replacement request; switches
            # before the first action belong to the current request (for
            # example a voluntary pivot at the start of a turn).
            if _is_choice_move(line.parts) or _is_choice_switch(line.parts):
                if _is_choice_switch(line.parts) and saw_action:
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


def _view(
    state: _ReplayState,
    perspective: int,
    *,
    preview: bool,
    effective_species: Mapping[tuple[int, int], str] | None = None,
    forced_move_slots: Sequence[int] = (),
) -> FixtureBattleView:
    opponent = 1 - perspective
    teams = [list(state.teams[side]) for side in (0, 1)]
    active = [list(state.active[side]) for side in (0, 1)]
    for (side, slot), species in (effective_species or {}).items():
        current = active[side][slot]
        actual = state.team_pokemon(side, species)
        if current is None or actual is None or actual is current:
            continue
        effective = replace(
            actual,
            current_hp_fraction=current.current_hp_fraction,
            fainted=current.fainted,
            selected_in_teampreview=True,
            status=current.status,
            boosts=current.boosts,
        )
        actual_index = next(index for index, pokemon in enumerate(teams[side]) if pokemon is actual)
        teams[side][actual_index] = effective
        active[side][slot] = effective
    own_active = tuple(active[perspective])
    opponent_active = tuple(active[opponent])
    slots = []
    known_switches = tuple(
        tuple(
            candidate
            for candidate in teams[perspective]
            if candidate.selected_in_teampreview
            and not candidate.fainted
            and candidate not in own_active
        )
        for _ in (0, 1)
    )
    for slot_index, pokemon in enumerate(own_active):
        moves = () if pokemon is None else tuple((tuple((-2, -1, 0, 1)),) for _ in pokemon.moves)
        switch_slots = tuple(
            index
            for index, candidate in enumerate(teams[perspective])
            if not candidate.fainted and candidate not in own_active
        )
        slots.append(
            SlotDecision(
                switch_slots=switch_slots,
                move_targets=tuple(targets[0] for targets in moves),
                active=pokemon is not None,
                # A public replay does not reveal reserve selections until a
                # reserve actually enters. Keep all non-fainted OTS members
                # as conservative switch candidates, but only assert the
                # forced-switch phase when a known selected reserve exists.
                force_switch=pokemon is None and bool(known_switches[slot_index]),
                can_mega=not state.used_mega[perspective],
                forced_move=slot_index in forced_move_slots,
            )
        )
    decision = DecisionView(
        slots=(slots[0], slots[1]),
        team_preview=preview,
        team_size=max(1, len(teams[perspective])),
    )
    return FixtureBattleView(
        team=_team_mapping(teams[perspective]),
        opponent_team=_team_mapping(teams[opponent]),
        active_pokemon=own_active,
        opponent_active_pokemon=opponent_active,
        available_moves=tuple(
            tuple(pokemon.moves) if pokemon is not None else () for pokemon in own_active
        ),
        available_switches=known_switches,
        can_mega_evolve=(not state.used_mega[perspective],) * 2,
        force_switch=tuple(slot.force_switch for slot in slots),
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
        identifiers=dict(state.identifiers),
    )


def _events_and_update(
    state: _ReplayState, lines: Sequence[Any], diagnostics: Counter[str]
) -> tuple[BattleEvent, ...]:
    raw_events: list[RawBattleEvent] = []
    for line in lines:
        raw_events.append(build_raw_event(line.parts, state.hp_for))
        # The live capture snapshots HP immediately before each parsed
        # protocol line. Apply each line after building its raw event so two
        # consecutive damage/heal messages get the same pre-HP baselines as
        # poke-env, rather than all seeing the state from the start of the
        # request.
        try:
            state.apply(line.parts)
        except (TypeError, ValueError, IndexError):
            diagnostics["state_update_errors"] += 1
    before = Counter(EVENT_DIAGNOSTICS)
    try:
        events = parse_events(raw_events, tokenizer)
    except (TypeError, ValueError, IndexError):
        diagnostics["parser_errors"] += 1
        diagnostics["parse_error_lines"] += len(lines)
        events = []
    after = Counter(EVENT_DIAGNOSTICS)
    for key, count in after.items():
        diagnostics[key] += max(0, count - before[key])
    return tuple(events)


def reconstruct_perspective(
    document: ReplayDocument,
    *,
    perspective: int,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
) -> ReconstructedPerspective:
    """Build pre-decision player-relative views while enforcing causal cutoffs."""
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    if dex is None:
        from p0.model.resources import default_runtime_resources

        dex = default_runtime_resources().dex
    state = _ReplayState(document, dex)
    counters: Counter[str] = Counter()
    snapshots: list[ReconstructedSnapshot] = []
    decisions: list[DecisionRecord] = []
    pending_events: tuple[BattleEvent, ...] = ()
    illusion_species_by_line = _illusion_species_by_line(document.protocol_lines)
    for decision_index, (start, end, decision_type) in enumerate(_segments(document)):
        lines = document.protocol_lines[start:end]
        effective_species = {
            (side, slot): species
            for (line_index, side, slot), species in illusion_species_by_line.items()
            if line_index == start - 1 and side == perspective
        }
        effective_species.update(
            _infer_illusion_active_species(
                state,
                lines,
                perspective,
                illusion_species_by_line,
            )
        )
        forced_move_slots = {
            endpoint[1]
            for line in lines
            if _is_choice_move(line.parts)
            and len(line.parts) >= 4
            and (endpoint := _ReplayState._endpoint(line.parts[2])) is not None
            and endpoint[0] == perspective
            and normalize_id(line.parts[3]) in {"struggle", "recharge"}
        }
        view = _view(
            state,
            perspective,
            preview=decision_type is DecisionType.TEAM_PREVIEW,
            effective_species=effective_species,
            forced_move_slots=forced_move_slots,
        )
        # The live environment presents the state after the preceding
        # request's protocol messages and exposes exactly those messages as
        # this request's event window. Do not attach the current action's
        # consequences to the action that caused them.
        view.events = list(pending_events)
        observed = (
            _preview_actions(state, lines, perspective)
            if decision_type is DecisionType.TEAM_PREVIEW
            else _observed_actions(
                state,
                lines,
                perspective,
                counters,
                illusion_species_by_line,
            )
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
        for slot, action in enumerate(observed[:2]):
            if action is not None and action.action is not None:
                if action.action not in legal_actions(view.decision, slot):
                    counters["observed_illegal_action"] += 1
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
                events=pending_events,
                raw_lines=tuple(line.raw for line in lines),
            )
        )
        pending_events = events
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
    dex: Mapping[str, Any] | None = None,
) -> tuple[ReconstructedPerspective, ReconstructedPerspective]:
    return (
        reconstruct_perspective(document, perspective=0, max_candidates=max_candidates, dex=dex),
        reconstruct_perspective(document, perspective=1, max_candidates=max_candidates, dex=dex),
    )


reconstruct_game = reconstruct_perspective


__all__ = [
    "ReplayMove",
    "ReplayName",
    "ReplayPokemon",
    "ReconstructedPerspective",
    "ReconstructedSnapshot",
    "reconstruct_both",
    "reconstruct_game",
    "reconstruct_perspective",
]
