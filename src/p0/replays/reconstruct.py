"""Pure player-relative replay reconstruction from protocol lines."""

from __future__ import annotations

import hashlib
import re
import typing
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import orjson
from poke_env.battle import Move
from poke_env.data import GenData

from p0.battle.actions import (
    FORCED_ACTION,
    MEGA_FORCED_ACTION,
    MOVE_END,
    MOVE_START,
    PASS_ACTION,
    SWITCH_START,
    TARGET_COUNT,
    encode_team_pair,
)
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
    StatPoints,
    calculate_stats,
    impute_candidates,
    select_candidate,
)

_ENDPOINT_ACTION_STATE_TAGS = frozenset(
    {
        "-ability",
        "-activate",
        "-boost",
        "-clearallboost",
        "-clearboost",
        "-clearnegativeboost",
        "-clearpositiveboost",
        "-copyboost",
        "-curestatus",
        "-end",
        "-enditem",
        "-formechange",
        "-invertboost",
        "-item",
        "-singlemove",
        "-singleturn",
        "-start",
        "-status",
        "-swapboost",
        "-terastallize",
        "-transform",
        "-unboost",
        "detailschange",
    }
)
_GLOBAL_ACTION_STATE_TAGS = frozenset(
    {"-fieldend", "-fieldstart", "-sideend", "-sidestart", "-weather"}
)
_TURN_COUNTER_EFFECTS = frozenset(
    {
        "BIDE",
        "BIND",
        "CLAMP",
        "DISABLE",
        "DOOMDESIRE",
        "DYNAMAX",
        "EMBARGO",
        "ENCORE",
        "FIRESPIN",
        "FUTURESIGHT",
        "GRAVITY",
        "GMAXCENTIFERNO",
        "GMAXSANDBLAST",
        "HEALBLOCK",
        "INFESTATION",
        "MAGMASTORM",
        "MAGNETRISE",
        "SANDTOMB",
        "SKYDROP",
        "SLOWSTART",
        "SNAPTRAP",
        "TAUNT",
        "TELEKINESIS",
        "THROATCHOP",
        "THUNDERCAGE",
        "UPROAR",
        "WHIRLPOOL",
        "WRAP",
    }
)
_END_ON_MOVE_EFFECTS = frozenset(
    {"GLAIVERUSH", "DANCER", "GRUDGE", "DESTINYBOND", "RAGE", "INSTRUCT", "FOCUSPUNCH"}
)
_END_ON_TURN_EFFECTS = frozenset(
    {
        "AFTERYOU",
        "BANEFULBUNKER",
        "BEAKBLAST",
        "BURNINGBULWARK",
        "CRAFTYSHIELD",
        "FEINT",
        "FLINCH",
        "FOCUSPUNCH",
        "FOLLOWME",
        "INSTRUCT",
        "KINGSSHIELD",
        "CUSTAPBERRY",
        "MINDREADER",
        "MAGICCOAT",
        "OBSTRUCT",
        "PROTECT",
        "QUASH",
        "QUICKCLAW",
        "QUICKDRAW",
        "QUICKGUARD",
        "RAGEPOWDER",
        "ROOST",
        "SPIKYSHIELD",
        "SPOTLIGHT",
        "WIDEGUARD",
    }
)


def normalize_id(value: str) -> str:
    """Normalize a protocol identifier without importing runtime team adapters."""
    return "".join(character for character in value.casefold() if character.isalnum())


def _species_base_stats(dex: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Index exact species and documented form aliases by normalized id.

    Args:
        dex: The pokedex data mapping containing species and baseStats information.

    Returns:
        A dictionary mapping normalized species and alias IDs to their base stats.
    """
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


def _species_data_index(dex: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index exact species and form aliases by normalized identifier."""
    index: dict[str, Mapping[str, Any]] = {}
    entries = tuple(entry for entry in dex.get("species", ()) if isinstance(entry, Mapping))
    by_name = {
        normalize_id(str(entry.get("id", entry.get("name", "")))): entry for entry in entries
    }
    for entry in entries:
        for value in (entry.get("id"), entry.get("name")):
            if isinstance(value, str) and value:
                index[normalize_id(value)] = entry
        base = entry.get("baseSpecies")
        base_entry = by_name.get(normalize_id(str(base))) if base else None
        fallback = base_entry if isinstance(base_entry, Mapping) else entry
        for value in (*entry.get("formeOrder", ()), *entry.get("otherFormes", ())):
            if isinstance(value, str) and value:
                index.setdefault(normalize_id(value), fallback)
    return index


def _move_data_index(dex: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        normalize_id(str(entry.get("id", entry.get("name", "")))): entry
        for entry in dex.get("moves", ())
        if isinstance(entry, Mapping)
    }


def _enum_name(value: str) -> str:
    """Return a poke-env-compatible enum member name for a protocol value."""
    normalized = normalize_id(value)
    aliases = {
        "sunnyday": "SUNNYDAY",
        "raindance": "RAINDANCE",
        "trickroom": "TRICK_ROOM",
        "magicroom": "MAGIC_ROOM",
        "wonderroom": "WONDER_ROOM",
        "toxicspikes": "TOXIC_SPIKES",
    }
    return aliases.get(normalized, normalized.upper())


_BASE_STATS_CACHE: dict[str, dict[str, dict[str, int]]] = {}


def _get_base_stats_index(dex: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Compute the species base-stats index, caching by a content fingerprint.

    The cache is keyed by a SHA-256 of the canonical JSON of the dex's
    ``species`` list, avoiding the stale-hit and memory-leak problems of an
    ``id(dex)``-keyed cache when callers pass freshly-loaded mappings.
    """
    species = dex.get("species", ())
    fingerprint = hashlib.sha256(orjson.dumps(species, option=orjson.OPT_SORT_KEYS)).hexdigest()
    if fingerprint not in _BASE_STATS_CACHE:
        _BASE_STATS_CACHE[fingerprint] = _species_base_stats(dex)
    return _BASE_STATS_CACHE[fingerprint]


def _make_replay_pokemon(
    species: str,
    species_data_index: Mapping[str, Mapping[str, Any]],
    move_data_index: Mapping[str, Mapping[str, Any]],
    moves: Mapping[str, ReplayMove] | None = None,
    ability: str | None = None,
    item: str | None = None,
    nature: str | None = None,
) -> ReplayPokemon:
    normalized = normalize_id(species)
    species_data = species_data_index.get(normalized, {})
    base_stats = species_data.get("baseStats", {})
    types = tuple(str(value) for value in species_data.get("types", ()) if isinstance(value, str))
    return ReplayPokemon(
        species=species,
        moves=moves if moves is not None else {},
        ability=ability,
        base_ability_data=ability,
        forme_ability_data=None,
        item=item,
        nature=nature,
        base_stats_data=base_stats if isinstance(base_stats, Mapping) else {},
        type_1_data=ReplayName(_enum_name(types[0])) if types else None,
        type_2_data=ReplayName(_enum_name(types[1])) if len(types) > 1 else None,
        weight_data=float(species_data.get("weightkg", 0.0)),
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
    target: str = "normal"
    increments_protect_counter: bool = False
    current_pp: int | None = None
    max_pp: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ReplayPokemon:
    species: str
    moves: Mapping[str, ReplayMove] = field(default_factory=dict)
    current_hp_fraction: float = 1.0
    fainted: bool = False
    revealed: bool = True
    selected_in_teampreview: bool | None = None
    ability: str | None = None
    base_ability_data: str | None = None
    forme_ability_data: str | None = None
    item: str | None = None
    nature: str | None = None
    base_stats_data: Mapping[str, int] = field(default_factory=dict)
    type_1_data: ReplayName | None = None
    type_2_data: ReplayName | None = None
    weight_data: float = 0.0
    status: Any = None
    boosts: Mapping[str, int] = field(default_factory=_zero_boosts)
    level: int | None = 50
    effects_data: Mapping[ReplayName, int] = field(default_factory=dict)
    presence_effects: frozenset[ReplayName] = frozenset()
    turn_effects: frozenset[ReplayName] = frozenset()
    active_turns: int = 0
    protect_counter_data: int = 0
    status_counter_data: int = 0
    preparing_data: bool = False
    last_move_data: str | None = None

    @property
    def hp_provenance(self) -> int:
        # Replay HP may be quantized to the uploader's public percentage.
        from p0.model.structured_observation import Provenance

        return int(Provenance.OBSERVED)

    @property
    def base_species(self) -> str:
        return self.species

    @property
    def type_1(self) -> Any:
        return self.type_1_data

    @property
    def type_2(self) -> Any:
        return self.type_2_data

    @property
    def base_stats(self) -> Mapping[str, int]:
        return self.base_stats_data

    @property
    def stats(self) -> Mapping[str, int | None]:
        return {}

    @property
    def protect_counter(self) -> int:
        return self.protect_counter_data

    @property
    def first_turn(self) -> bool:
        return self.active_turns == 1

    @property
    def weight(self) -> float:
        return self.weight_data

    @property
    def effects(self) -> Mapping[Any, int]:
        return self.effects_data

    @property
    def status_counter(self) -> int:
        return self.status_counter_data

    @property
    def preparing(self) -> Any:
        return self.preparing_data

    @property
    def last_move(self) -> ReplayMove | None:
        if self.last_move_data is None:
            return None
        return self.moves.get(self.last_move_data)


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
    action_view: DecisionView | None = None


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
    precomputed: tuple[int, int, int, int, int, int] | None
    confidence: float


@dataclass(frozen=True, slots=True)
class PerspectiveKnowledge:
    """Private facts recoverable for one player from the complete replay."""

    player: int
    selected_species: frozenset[str]
    selection_size: int
    selection_complete: bool

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("PerspectiveKnowledge.player must be 0 or 1")
        if self.selection_size < 0:
            raise ValueError("PerspectiveKnowledge.selection_size must be non-negative")
        if len(self.selected_species) > self.selection_size:
            raise ValueError("Recovered selections exceed the declared team size")
        if self.selection_complete != (len(self.selected_species) == self.selection_size):
            raise ValueError("selection_complete does not match the recovered selection count")

    def selected_state(self, side: int, species: str) -> bool | None:
        """Return own oracle selection, leaving opponent and ambiguous members unknown."""
        if side != self.player:
            return None
        if normalize_id(species) in self.selected_species:
            return True
        return False if self.selection_complete else None


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
            value: dict[str, typing.Any] = dict(
                nature=str(details.get("nature", "serious")),
                item=str(details.get("item", "")),
                ability=str(details.get("ability", "")),
                moves=moves,
                move_categories=categories,
                base_stats=BaseStats.from_mapping(base_mapping),
            )
            candidates = impute_candidates(**value)
            candidate = select_candidate(seed=seed + side * 1009 + index, **value)
            stats = tuple[int, int, int, int, int, int](
                calculate_stats(value["base_stats"], candidate.points, value["nature"], level)
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


def _format_generation(format_id: str) -> int:
    match = re.match(r"gen(\d+)", normalize_id(format_id))
    return int(match.group(1)) if match is not None else 9


def _live_max_pp(move_id: str, generation: int, data: Mapping[str, Any]) -> int | None:
    """Use the same move PP contract as the live poke-env observation path."""
    try:
        return int(Move(move_id, generation).max_pp)
    except (KeyError, ValueError):
        pp = int(data.get("pp", 0))
        value = pp if data.get("noPPBoosts") else pp * 8 // 5
        return value or None


def _replay_moves(
    names: tuple[str, ...],
    move_data_index: Mapping[str, Mapping[str, Any]],
    generation: int,
) -> dict[str, ReplayMove]:
    moves: dict[str, ReplayMove] = {}
    for name in names:
        move_id = normalize_id(name)
        data = move_data_index.get(move_id, {})
        max_pp = _live_max_pp(move_id, generation, data)
        moves[move_id] = ReplayMove(
            id=move_id,
            type=ReplayName(_enum_name(str(data.get("type", "unknown")))),
            category=ReplayName(_enum_name(str(data.get("category", "unknown")))),
            target=str(data.get("target", "normal")),
            increments_protect_counter=move_id
            in {
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
            },
            current_pp=max_pp,
            max_pp=max_pp,
        )
    return moves


class _ReplayState:
    def __init__(
        self,
        document: ReplayDocument,
        dex: Mapping[str, Any],
        knowledge: PerspectiveKnowledge,
    ):
        self.turn = 0
        self.used_mega = [False, False]
        self.active: list[list[ReplayPokemon | None]] = [[None, None], [None, None]]
        self.base_stats_index = _get_base_stats_index(dex)
        self.species_identity_index = _species_identity_index(dex)
        self.species_data_index = _species_data_index(dex)
        self.move_data_index = _move_data_index(dex)
        self.generation = _format_generation(document.metadata.format_id)
        self.mega_items = frozenset(
            normalize_id(str(entry.get("requiredItem", "")))
            for entry in dex.get("transformations", ())
            if isinstance(entry, Mapping) and entry.get("isMega") and entry.get("requiredItem")
        )
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
                pokemon = _make_replay_pokemon(
                    species=species,
                    species_data_index=self.species_data_index,
                    move_data_index=self.move_data_index,
                    moves=_replay_moves(
                        moves_tuple,
                        self.move_data_index,
                        self.generation,
                    ),
                    ability=ability,
                    item=item,
                    # poke-env's OTS parser does not expose nature on its
                    # Pokemon objects. Matching that runtime contract avoids a
                    # replay-only feature unavailable during live inference.
                    nature=None,
                )
                pokemon = replace(
                    pokemon,
                    current_hp_fraction=0.0,
                    revealed=False,
                    selected_in_teampreview=knowledge.selected_state(side, species),
                )
                side_teams.append(pokemon)
            self.teams.append(side_teams)
        self.hp: dict[str, float] = {}
        self.fainted: set[str] = set()
        self.weather: dict[ReplayName, int] = {}
        self.fields: dict[ReplayName, int] = {}
        self.side_conditions: list[dict[ReplayName, int]] = [{}, {}]
        # A switch can initially identify an Illusion user as another roster
        # member. Keep the pre-effect object so a later ``replace`` line can
        # restore the disguised member before transferring runtime state to
        # the revealed Pokemon.
        self._illusion_baselines: dict[tuple[int, int], ReplayPokemon] = {}
        # Events carry Showdown identifiers while the observation builder
        # indexes concrete Pokemon objects. Keep this mapping across request
        # boundaries so replay event grounding matches the live adapter.
        self.identifiers: dict[str, ReplayPokemon] = {}

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
        pokemon = _make_replay_pokemon(species, self.species_data_index, self.move_data_index)
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

    def _active_for(self, identifier: str) -> tuple[int, ReplayPokemon] | None:
        endpoint = self._endpoint(identifier)
        if endpoint is None:
            return None
        pokemon = self.active[endpoint[0]][endpoint[1]]
        if pokemon is None:
            return None
        return endpoint[0], pokemon

    def _end_turn(self, turn: int) -> None:
        self.turn = turn
        for side in (0, 1):
            for pokemon in tuple(self.active[side]):
                if pokemon is None:
                    continue
                effects = {
                    effect: (
                        counter + 1
                        if effect not in pokemon.presence_effects
                        and effect.name in _TURN_COUNTER_EFFECTS
                        else counter
                    )
                    for effect, counter in pokemon.effects.items()
                    if (
                        effect not in pokemon.turn_effects
                        and effect.name not in _END_ON_TURN_EFFECTS
                    )
                }
                status_counter = pokemon.status_counter
                if getattr(pokemon.status, "name", "") == "TOX":
                    status_counter += 1
                self._replace_active(
                    side,
                    pokemon,
                    active_turns=pokemon.active_turns + 1,
                    effects_data=effects,
                    turn_effects=frozenset(),
                    status_counter_data=status_counter,
                )

    def _clear_switch_state(self, side: int, pokemon: ReplayPokemon) -> None:
        status_counter = pokemon.status_counter
        if getattr(pokemon.status, "name", "") == "TOX":
            status_counter = 0
        self._replace_active(
            side,
            pokemon,
            ability=pokemon.forme_ability_data or pokemon.base_ability_data,
            boosts=_zero_boosts(),
            effects_data={},
            presence_effects=frozenset(),
            turn_effects=frozenset(),
            active_turns=0,
            protect_counter_data=0,
            status_counter_data=status_counter,
            preparing_data=False,
            last_move_data=None,
        )

    def _apply_move(self, parts: Sequence[str]) -> None:
        if len(parts) < 4:
            return
        located = self._active_for(parts[2])
        if located is None:
            return
        side, pokemon = located
        move_id = normalize_id(parts[3])
        move = pokemon.moves.get(move_id)
        failed = any(part in {"[miss]", "[still]", "[notarget]"} for part in parts[4:])
        moves = dict(pokemon.moves)
        if move is not None and move.current_pp is not None and _is_choice_move(parts):
            moves[move_id] = replace(move, current_pp=max(0, move.current_pp - 1))
        status_counter = pokemon.status_counter
        if getattr(pokemon.status, "name", "") == "SLP":
            status_counter += 1
        effects = dict(pokemon.effects)
        presence_effects = set(pokemon.presence_effects)
        effects = {
            effect: counter
            for effect, counter in effects.items()
            if effect.name not in _END_ON_MOVE_EFFECTS
            and not (
                effect.name == "FLASHFIRE"
                and move is not None
                and getattr(move.type, "name", "") == "FIRE"
            )
        }
        presence_effects.intersection_update(effects)
        if move_id == "minimize":
            effect = ReplayName("MINIMIZE")
            effects.setdefault(effect, 0)
            presence_effects.add(effect)
        self._replace_active(
            side,
            pokemon,
            moves=moves,
            protect_counter_data=(
                pokemon.protect_counter + 1
                if move is not None and move.increments_protect_counter and not failed
                else 0
            ),
            status_counter_data=status_counter,
            effects_data=effects,
            presence_effects=frozenset(presence_effects),
            preparing_data=False,
            last_move_data=move_id,
        )

    @staticmethod
    def _effect(value: str) -> ReplayName:
        _, separator, remainder = value.partition(":")
        name = remainder.strip() if separator else value
        return ReplayName(_enum_name(name))

    def _species_changes(self, species: str) -> dict[str, Any]:
        data = self.species_data_index.get(normalize_id(species), {})
        base_stats = data.get("baseStats", {})
        types = tuple(str(value) for value in data.get("types", ()) if isinstance(value, str))
        changes: dict[str, Any] = {
            "species": species,
            "base_stats_data": base_stats if isinstance(base_stats, Mapping) else {},
            "type_1_data": ReplayName(_enum_name(types[0])) if types else None,
            "type_2_data": ReplayName(_enum_name(types[1])) if len(types) > 1 else None,
            "weight_data": float(data.get("weightkg", 0.0)),
        }
        # Replay form names are interpreted by the same pinned Showdown dex as
        # the live adapter. The project dex also contains custom form entries,
        # but some of their ability overrides differ from poke-env's runtime
        # data (for example Scolipede-Mega and Raichu-Mega-Y).
        live_data = GenData.from_gen(9).pokedex.get(normalize_id(species), {})
        abilities = live_data.get("abilities") or data.get("abilities", {})
        if isinstance(abilities, Mapping) and abilities.get("0"):
            changes["ability"] = str(abilities["0"])
            changes["forme_ability_data"] = str(abilities["0"])
        return changes

    def apply(self, parts: Sequence[str]) -> None:
        """Advance the state machine by applying one parsed protocol line.

        Dispatches on ``parts[1]`` (the Showdown tag) to update active
        pokemon, HP, boosts, status, weather, fields, and Illusion aliases.
        Each handler mutates ``self.teams``, ``self.active``, ``self.hp``,
        and ``self.identifiers`` in place so the next ``_view`` call observes
        the post-line state.

        Arguments:
            parts: The split protocol line (``|tag|...``).

        Returns:
            None; mutates the replay state in place.
        """
        if len(parts) < 2:
            return
        tag = parts[1]
        if tag == "turn" and len(parts) >= 3 and parts[2].isdigit():
            self._end_turn(int(parts[2]))
            return
        if tag in ("switch", "drag") and len(parts) >= 3:
            endpoint = self._endpoint(parts[2])
            if endpoint is None:
                return
            side, slot = endpoint
            endpoint_id = parts[2].split(":", 1)[0]
            outgoing = self.active[side][slot]
            if outgoing is None:
                outgoing = self.identifiers.get(endpoint_id)
            if outgoing is not None:
                self._clear_switch_state(side, outgoing)
            species = _switch_species(parts) or "unknown"
            pokemon = self.pokemon_for(side, species)
            for active_slot, active in enumerate(self.active[side]):
                if active is pokemon:
                    self._promote_active_illusion_if_duplicate(side, active_slot, species)
            hp_fraction = get_hp_fraction(parts[4]) if len(parts) >= 5 else 1.0
            hp_status = parts[4].split() if len(parts) >= 5 else ()
            switch_status = (
                ReplayName(_enum_name(hp_status[-1]))
                if hp_status and hp_status[-1] in {"brn", "frz", "par", "psn", "slp", "tox"}
                else pokemon.status
            )
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
                    revealed=True,
                    status=switch_status,
                    active_turns=0,
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
                    revealed=True,
                    status=switch_status,
                    active_turns=0,
                )
            self.active[side][slot] = pokemon
            self._illusion_baselines[(side, slot)] = pokemon
            self.hp[endpoint_id] = hp_fraction
            self.identifiers[parts[2]] = pokemon
            self.identifiers[endpoint_id] = pokemon
            return
        if tag == "move":
            self._apply_move(parts)
            return
        if tag == "cant" and len(parts) >= 3:
            located = self._active_for(parts[2])
            if located is not None:
                side, pokemon = located
                status_counter = pokemon.status_counter
                if getattr(pokemon.status, "name", "") == "SLP":
                    status_counter += 1
                self._replace_active(
                    side,
                    pokemon,
                    protect_counter_data=0,
                    status_counter_data=status_counter,
                )
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
                get_hp_fraction(parts[4]) if len(parts) >= 5 else current.current_hp_fraction
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
                    self._replace_active(
                        side,
                        pokemon,
                        fainted=True,
                        current_hp_fraction=0.0,
                        status=ReplayName("FNT"),
                        ability=pokemon.forme_ability_data or pokemon.base_ability_data,
                        effects_data={},
                        presence_effects=frozenset(),
                        turn_effects=frozenset(),
                    )
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
                    status = None
                    if tag == "-status" and len(parts) >= 4:
                        status = ReplayName(_enum_name(parts[3]))
                    self._replace_active(endpoint[0], current, status=status, status_counter_data=0)
            return
        if tag in ("-boost", "-unboost", "-setboost") and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    boosts = dict(current.boosts)
                    stat = parts[3]
                    if tag == "-setboost":
                        value = int(parts[4]) if len(parts) >= 5 else 0
                    else:
                        delta = int(parts[4]) if len(parts) >= 5 else 0
                        value = boosts.get(stat, 0) + (delta if tag == "-boost" else -delta)
                    boosts[stat] = max(-6, min(6, value))
                    self._replace_active(endpoint[0], current, boosts=boosts)
            return
        if tag in {
            "-clearboost",
            "-clearnegativeboost",
            "-clearpositiveboost",
            "-clearallboost",
        }:
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
                boosts = dict(current.boosts)
                if tag in {"-clearboost", "-clearallboost"}:
                    boosts = _zero_boosts()
                elif tag == "-clearnegativeboost":
                    boosts = {stat: max(0, value) for stat, value in boosts.items()}
                else:
                    boosts = {stat: min(0, value) for stat, value in boosts.items()}
                self._replace_active(side, current, boosts=boosts)
            return
        if tag == "-copyboost" and len(parts) >= 4:
            source = self._active_for(parts[2])
            target = self._active_for(parts[3])
            if source is not None and target is not None:
                _, source_pokemon = source
                target_side, target_pokemon = target
                self._replace_active(target_side, target_pokemon, boosts=source_pokemon.boosts)
            return
        if tag == "-invertboost" and len(parts) >= 3:
            located = self._active_for(parts[2])
            if located is not None:
                side, pokemon = located
                self._replace_active(
                    side,
                    pokemon,
                    boosts={stat: -value for stat, value in pokemon.boosts.items()},
                )
            return
        if tag == "-swapboost" and len(parts) >= 4:
            source = self._active_for(parts[2])
            target = self._active_for(parts[3])
            if source is not None and target is not None:
                source_side, source_pokemon = source
                target_side, target_pokemon = target
                stats = (
                    tuple(_zero_boosts())
                    if len(parts) < 5
                    else tuple(
                        stat
                        for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
                        if stat in parts[4] or "[from]" in parts[4]
                    )
                )
                if not stats:
                    stats = tuple(_zero_boosts())
                source_boosts = dict(source_pokemon.boosts)
                target_boosts = dict(target_pokemon.boosts)
                self._replace_active(
                    source_side,
                    source_pokemon,
                    boosts={
                        **source_boosts,
                        **{stat: target_boosts[stat] for stat in stats},
                    },
                )
                self._replace_active(
                    target_side,
                    target_pokemon,
                    boosts={
                        **target_boosts,
                        **{stat: source_boosts[stat] for stat in stats},
                    },
                )
            return
        if tag in ("detailschange", "-formechange") and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    species = parts[3].split(",", 1)[0]
                    self._replace_active(endpoint[0], current, **self._species_changes(species))
            return
        if tag == "-ability" and len(parts) >= 4:
            endpoint = self._endpoint(parts[2])
            if endpoint is not None:
                current = self.active[endpoint[0]][endpoint[1]]
                if current is not None:
                    base_ability = current.base_ability_data or parts[3]
                    self._replace_active(
                        endpoint[0],
                        current,
                        ability=parts[3],
                        base_ability_data=base_ability,
                        forme_ability_data=current.forme_ability_data,
                    )
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
            return
        if tag == "-weather" and len(parts) >= 3:
            self.weather = (
                {} if normalize_id(parts[2]) == "none" else {self._effect(parts[2]): self.turn}
            )
            return
        if tag in {"-fieldstart", "-fieldend"} and len(parts) >= 3:
            effect = self._effect(parts[2])
            if tag == "-fieldstart":
                self.fields[effect] = self.turn
            else:
                self.fields.pop(effect, None)
            return
        if tag in {"-sidestart", "-sideend"} and len(parts) >= 4:
            side_prefix = parts[2][:2]
            if side_prefix not in {"p1", "p2"}:
                return
            conditions = self.side_conditions[int(side_prefix[1]) - 1]
            effect = self._effect(parts[3])
            if tag == "-sideend":
                conditions.pop(effect, None)
            elif effect.name in {"SPIKES", "TOXIC_SPIKES"}:
                conditions[effect] = conditions.get(effect, 0) + 1
            else:
                conditions.setdefault(effect, self.turn)
            return
        if tag in {"-activate", "-start", "-end", "-singleturn", "-singlemove"} and len(parts) >= 4:
            located = self._active_for(parts[2])
            if located is None:
                return
            side, pokemon = located
            effect = self._effect(parts[3])
            effects = dict(pokemon.effects)
            presence_effects = set(pokemon.presence_effects)
            turn_effects = set(pokemon.turn_effects)
            if tag == "-end":
                effects.pop(effect, None)
                presence_effects.discard(effect)
                turn_effects.discard(effect)
            else:
                effects.setdefault(effect, 0)
                if tag == "-activate":
                    if effect.name in _TURN_COUNTER_EFFECTS:
                        presence_effects.discard(effect)
                    else:
                        presence_effects.add(effect)
                elif tag == "-start":
                    presence_effects.discard(effect)
                if tag in {"-singleturn", "-singlemove"} and effect.name in _END_ON_TURN_EFFECTS:
                    turn_effects.add(effect)
            self._replace_active(
                side,
                pokemon,
                effects_data=effects,
                presence_effects=frozenset(presence_effects),
                turn_effects=frozenset(turn_effects),
            )
            return
        if tag == "-prepare" and len(parts) >= 4:
            located = self._active_for(parts[2])
            if located is not None:
                side, pokemon = located
                self._replace_active(side, pokemon, preparing_data=True)


def _team_mapping(team: Sequence[ReplayPokemon]) -> dict[str, ReplayPokemon]:
    return {f"{index}:{pokemon.species}": pokemon for index, pokemon in enumerate(team)}


def _target_code(actor: str, target: str | None) -> int | None:
    if not target:
        return None
    actor_endpoint = _ReplayState._endpoint(actor)
    target_endpoint = _ReplayState._endpoint(target)
    if actor_endpoint is None or target_endpoint is None:
        return None
    if actor_endpoint[0] == target_endpoint[0]:
        return -(target_endpoint[1] + 1)
    return target_endpoint[1] + 1


def _observed_move_target(
    state: _ReplayState,
    side: int,
    slot: int,
    move_slot: int | None,
    actor: str,
    target: str | None,
    *,
    species: str | None = None,
) -> int | None:
    active = state.active[side][slot]
    if species is not None:
        active = state.team_pokemon(side, species)
    if active is None or move_slot is None:
        return None
    moves = tuple(active.moves.values())
    if move_slot >= len(moves):
        return None
    move_target = moves[move_slot].target
    if move_target == "self":
        return 0
    if move_target in {
        "all",
        "allAdjacent",
        "allAdjacentFoes",
        "allies",
        "allySide",
        "allyTeam",
        "foeSide",
        "randomNormal",
        "scripted",
    }:
        return 0
    return _target_code(actor, target)


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


def _is_pivot_switch(parts: Sequence[str]) -> bool:
    return any(
        normalize_id(part.removeprefix("[from] "))
        in {"uturn", "flipturn", "voltswitch", "batonpass", "partingshot"}
        for part in parts[4:]
    )


def _is_choice_switch(parts: Sequence[str]) -> bool:
    """Whether a switch line represents a submitted order.

    A pivot switch emitted after U-turn/Flip Turn is the outcome of the first
    request and the replacement request is represented by that switch line.
    Other ``[from]`` switches are automatic state updates.
    """
    return (
        len(parts) >= 3
        and parts[1] == "switch"
        and (not any(part.startswith("[from]") for part in parts[4:]) or _is_pivot_switch(parts))
    )


def _is_choice_move(parts: Sequence[str]) -> bool:
    """Whether a move line represents a submitted order rather than a proc."""
    return (
        len(parts) >= 4
        and parts[1] == "move"
        and not any(part.startswith("[from]") for part in parts[5:])
    )


def _perspective_has_choice(lines: Sequence[Any], perspective: int) -> bool:
    """Whether *perspective* submitted a move or switch in *lines*."""
    for line in lines:
        parts = line.parts
        if not (_is_choice_move(parts) or _is_choice_switch(parts)):
            continue
        endpoint = _ReplayState._endpoint(parts[2])
        if endpoint is not None and endpoint[0] == perspective:
            return True
    return False


def _is_terminal_segment(lines: Sequence[Any]) -> bool:
    """Whether *lines* contain a game-ending ``win`` or ``tie`` marker."""
    for line in lines:
        parts = line.parts
        if len(parts) >= 2 and parts[1] in ("win", "tie"):
            return True
    return False


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


def _perspective_knowledge(
    document: ReplayDocument,
    perspective: int,
    dex: Mapping[str, Any],
) -> PerspectiveKnowledge:
    """Recover private team selection without backfilling opponent knowledge."""
    player_id = f"p{perspective + 1}"
    roster = document.ots[perspective].revealed_species
    roster_by_identity: dict[str, str] = {}
    species_identities = _species_identity_index(dex)
    for species in roster:
        normalized = normalize_id(species)
        identity = species_identities.get(normalized, normalized)
        roster_by_identity.setdefault(identity, normalized)

    selection_size = min(4, len(roster))
    for line in document.protocol_lines:
        if (
            len(line.parts) >= 4
            and line.parts[1] == "teamsize"
            and line.parts[2] == player_id
            and line.parts[3].isdigit()
        ):
            selection_size = int(line.parts[3])

    illusion_species = _illusion_species_by_line(document.protocol_lines)
    selected: set[str] = set()
    for line in document.protocol_lines:
        parts = line.parts
        if len(parts) < 4 or parts[1] not in {"switch", "drag"}:
            continue
        endpoint = _ReplayState._endpoint(parts[2])
        if endpoint is None or endpoint[0] != perspective:
            continue
        species = illusion_species.get(
            (line.index, perspective, endpoint[1]), _switch_species(parts)
        )
        normalized = normalize_id(species)
        identity = species_identities.get(normalized, normalized)
        roster_species = roster_by_identity.get(identity)
        if roster_species is not None:
            selected.add(roster_species)

    return PerspectiveKnowledge(
        player=perspective,
        selected_species=frozenset(selected),
        selection_size=selection_size,
        selection_complete=len(selected) == selection_size,
    )


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
    """Extract the observed slot-0 and slot-1 actions for one decision segment.

    Walks the segment's protocol lines and maps each submitted ``move`` or
    ``switch`` order belonging to *perspective* into an ``ObservedAction``
    using the 49-action codec. Target ambiguity, forced-move (Struggle /
    Recharge), and Illusion-disguised switches are resolved conservatively.

    Arguments:
        state: The replay state machine (pre-segment view).
        lines: The protocol lines that compose this decision segment.
        perspective: The player index (0 or 1) to extract actions for.
        diagnostics: A counter to record extraction diagnostics.
        illusion_species_by_line: Optional Illusion species map for
            resolving displayed aliases to actual roster members.

    Returns:
        A ``(slot0, slot1, tags)`` triple where each slot is an
        ``ObservedAction`` or ``None`` when no action was observed.
    """
    observed: list[ObservedAction | None] = [None, None]
    tags: list[str] = []
    mega_slots: set[tuple[int, int]] = set()
    submission_invalidated: set[tuple[int, int]] = set()
    globally_invalidated = False
    animation_targets = _animation_targets(lines)
    for line in lines:
        parts = line.parts
        tag = parts[1] if len(parts) >= 2 else ""
        if len(parts) >= 3 and parts[1] == "-mega":
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is not None:
                mega_slots.add(endpoint)
        if len(parts) >= 3 and tag in _ENDPOINT_ACTION_STATE_TAGS:
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is not None:
                submission_invalidated.add(endpoint)
        elif tag in _GLOBAL_ACTION_STATE_TAGS:
            globally_invalidated = True
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
            if endpoint in submission_invalidated or globally_invalidated:
                # The request was submitted before this state transition. A
                # replay move records the engine's resulting execution, which
                # need not be the command the player originally selected.
                observed[slot] = ObservedAction(None, exact=False, tag="submission_state_changed")
                diagnostics["submission_state_changed"] += 1
                tags.append("submission_state_changed")
                continue
            move = parts[3] if len(parts) >= 4 else ""
            forced = normalize_id(move) in {"struggle", "recharge"}
            if forced:
                observed[slot] = ObservedAction(
                    MEGA_FORCED_ACTION if (endpoint in mega_slots) else FORCED_ACTION,
                    tag="forced_move",
                )
                continue
            target = parts[4] if len(parts) >= 5 else None
            target_tag = "move"
            if not target:
                target = animation_targets.get((endpoint[0], endpoint[1], normalize_id(move)))
                if target:
                    target_tag = "move_anim_target"
            effective_species = (
                illusion_species_by_line.get((line.index, endpoint[0], endpoint[1]))
                if illusion_species_by_line is not None
                else None
            )
            move_slot = _move_slot(
                state,
                perspective,
                slot,
                move,
                species=effective_species,
            )
            target_code = _observed_move_target(
                state,
                perspective,
                slot,
                move_slot,
                parts[2],
                target,
                species=effective_species,
            )
            if move_slot is None or target_code is None:
                diagnostics["move_slot_or_target_unknown"] += 1
                observed[slot] = ObservedAction(
                    None, exact=False, tag="move_slot_or_target_unknown"
                )
            else:
                action = MOVE_START + move_slot * TARGET_COUNT + target_code + 2
                if endpoint in mega_slots:
                    action += MOVE_END - MOVE_START
                if target_code == 0:
                    observed[slot] = ObservedAction(action, tag=target_tag)
                else:
                    mega_offset = MOVE_END - MOVE_START if endpoint in mega_slots else 0
                    alternatives = tuple(
                        MOVE_START + move_slot * TARGET_COUNT + target + 2 + mega_offset
                        for target in (-2, -1, 1, 2)
                    )
                    if target_tag != "move":
                        tags.append(target_tag)
                    observed[slot] = ObservedAction(
                        action,
                        alternatives=alternatives,
                        exact=False,
                        tag="execution_target",
                    )
        elif _is_choice_switch(parts):
            endpoint = _ReplayState._endpoint(parts[2])
            if endpoint is None or endpoint[0] != perspective:
                continue
            species = (
                illusion_species_by_line.get(
                    (line.index, endpoint[0], endpoint[1]), _switch_species(parts)
                )
                if illusion_species_by_line is not None
                else _switch_species(parts)
            )
            try:
                action = SWITCH_START + next(
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
    except (StopIteration, RuntimeError):
        return None, None, ("preview_roster_unknown",)
    if len(set(lead_indices)) != 2:
        return None, None, ("preview_duplicate_lead",)
    team_size = len(state.teams[perspective])
    first = encode_team_pair(lead_indices[0], lead_indices[1], team_size=team_size)
    alternatives = tuple(
        encode_team_pair(first_index, second_index, team_size=team_size)
        for first_index in range(team_size)
        for second_index in range(team_size)
        if first_index != second_index
        and first_index not in lead_indices
        and second_index not in lead_indices
    )
    return (
        ObservedAction(first, tag="preview_leads"),
        ObservedAction(alternatives=alternatives, exact=False, tag="preview_reserves_unknown"),
        ("preview_reserves_unknown",),
    )


def _segments(document: ReplayDocument) -> tuple[tuple[int, int, DecisionType], ...]:
    """Partition the protocol lines into contiguous decision segments.

    Args:
        document: The complete parsed replay document.

    Returns:
        A tuple of (start_index, end_index, decision_type) representing the protocol line
        boundaries for each decision request.
    """
    turns: list[int] = []
    preview: list[int] = []

    for line in document.protocol_lines:
        if len(line.parts) > 1:
            if line.parts[1] == "turn":
                turns.append(line.index)
            elif line.parts[1] == "teampreview":
                preview.append(line.index)

    segments: list[tuple[int, int, DecisionType]] = []

    if preview and turns and turns[0] > 0:
        segments.append((0, turns[0], DecisionType.TEAM_PREVIEW))
    elif preview and not turns:
        segments.append((0, len(document.protocol_lines), DecisionType.TEAM_PREVIEW))

    for index, start in enumerate(turns):
        end = turns[index + 1] if index + 1 < len(turns) else len(document.protocol_lines)
        boundaries = [start]
        saw_move = False

        for line in document.protocol_lines[start:end]:
            # cant is an outcome of a submitted move (flinch, paralysis,
            # sleep, Armor Tail, ...), and drag is an automatic battle
            # effect. Neither creates a new request in poke-env. A switch
            # after an action is a forced replacement request; switches
            # before the first action belong to the current request (for
            # example a voluntary pivot at the start of a turn).
            if _is_choice_move(line.parts):
                saw_move = True
            elif _is_choice_switch(line.parts) and saw_move:
                boundaries.append(line.index)
                saw_move = False

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
    wait: bool = False,
) -> FixtureBattleView:
    """Build a player-relative pre-decision view from the current state.

    Constructs the ``FixtureBattleView`` and ``DecisionView`` that the live
    environment would present at this request boundary. The view reflects
    the state *before* the segment's protocol lines are applied, with
    Illusion aliases and forced-move slots overlaid where applicable.

    Arguments:
        state: The replay state machine (pre-segment view).
        perspective: The player index (0 or 1) to build the view for.
        preview: Whether this is a team-preview decision.
        effective_species: Optional mapping of ``(side, slot)`` to the actual
            Illusion species to display instead of the disguised alias.
        forced_move_slots: Slot indices that are locked into a forced move
            (Struggle / Recharge) during this segment.
        wait: When ``True``, both slots are forced to ``PASS_ACTION`` — used
            for ``FORCED_PASS`` decisions where only the opponent acts.

    Returns:
        A ``FixtureBattleView`` ready for evidence extraction.
    """
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
    teams[perspective] = [
        replace(pokemon, current_hp_fraction=1.0)
        if not pokemon.revealed and pokemon.current_hp_fraction == 0.0
        else pokemon
        for pokemon in teams[perspective]
    ]
    own_active = tuple(active[perspective])
    opponent_active = tuple(active[opponent])
    can_mega_evolve = tuple(
        bool(
            pokemon is not None
            and not wait
            and not state.used_mega[perspective]
            and normalize_id(pokemon.item or "") in state.mega_items
            and "mega" not in normalize_id(pokemon.species)
        )
        for pokemon in own_active
    )
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
        moves = () if pokemon is None else tuple((tuple((-2, -1, 0, 1, 2)),) for _ in pokemon.moves)
        switch_slots = tuple(
            index
            for index, candidate in enumerate(teams[perspective])
            if candidate.selected_in_teampreview is not False
            and not candidate.fainted
            and candidate not in own_active
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
                can_mega=can_mega_evolve[slot_index],
                mega_known=False,
                forced_move=slot_index in forced_move_slots,
                # Replays contain executed outcomes, not the authoritative
                # request mask sent to this player. The action superset stays
                # available for evidence extraction, but its provenance is
                # explicitly unknown to the observation encoder.
                legality_known=False,
            )
        )
    decision = DecisionView(
        slots=(slots[0], slots[1]),
        team_preview=preview,
        team_size=max(1, len(teams[perspective])),
        wait=wait,
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
        can_mega_evolve=can_mega_evolve,
        force_switch=tuple(slot.force_switch for slot in slots),
        trapped=(False, False),
        maybe_trapped=(False, False),
        teampreview=preview,
        player_role=f"p{perspective + 1}",
        wait=False,
        weather=dict(state.weather),
        fields=dict(state.fields),
        side_conditions=dict(state.side_conditions[perspective]),
        opponent_side_conditions=dict(state.side_conditions[opponent]),
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
    """Build pre-decision player-relative views while enforcing causal cutoffs.

    Arguments:
        document: The complete parsed replay document.
        perspective: The index of the player to reconstruct the view for (0 or 1).
        max_candidates: The maximum number of joint action candidates to evaluate.
        dex: Optional pokedex data mapping. If omitted, uses the default runtime resources.

    Returns:
        A ReconstructedPerspective containing the sequential snapshots, decisions,
        and diagnostics built from the observed replay.
    """
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")

    if dex is None:
        from p0.model.resources import default_runtime_resources

        dex = default_runtime_resources().dex

    knowledge = _perspective_knowledge(document, perspective, dex)
    state = _ReplayState(document, dex, knowledge)
    counters: Counter[str] = Counter()
    if not knowledge.selection_complete:
        counters["own_selection_incomplete"] += 1
    snapshots: list[ReconstructedSnapshot] = []
    decisions: list[DecisionRecord] = []
    pending_events: tuple[BattleEvent, ...] = ()
    illusion_species_by_line = _illusion_species_by_line(document.protocol_lines)

    for decision_index, (start, end, decision_type) in enumerate(_segments(document)):
        lines = document.protocol_lines[start:end]
        state_update_lines = lines
        if lines and len(lines[0].parts) >= 2 and lines[0].parts[1] == "turn":
            state.apply(lines[0].parts)
            state_update_lines = lines[1:]
        pivot_segment = bool(state_update_lines and _is_pivot_switch(state_update_lines[0].parts))
        decision_lines = state_update_lines[:1] if pivot_segment else lines

        effective_species = {}
        for slot in (0, 1):
            species = illusion_species_by_line.get((start - 1, perspective, slot))
            if species is not None:
                effective_species[(perspective, slot)] = species

        effective_species.update(
            _infer_illusion_active_species(
                state,
                lines,
                perspective,
                illusion_species_by_line,
            )
        )
        forced_move_slots = tuple(
            endpoint[1]
            for line in decision_lines
            if _is_choice_move(line.parts)
            and len(line.parts) >= 4
            and (endpoint := _ReplayState._endpoint(line.parts[2])) is not None
            and endpoint[0] == perspective
            and normalize_id(line.parts[3]) in {"struggle", "recharge"}
        )
        # Detect a one-sided waiting segment: only the opponent has an
        # asynchronous replacement or pivot choice, so this perspective is
        # forced to pass. Boundary rule 1: a terminal KO that ends the game
        # has no later request, so it must not produce a FORCED_PASS record.
        # Boundary rule 2: a simultaneous replacement where this perspective
        # must also choose is handled by the ``not _perspective_has_choice``
        # guard below — it keeps the normal FORCED_SWITCH / TURN path.
        is_waiting = (
            decision_type is not DecisionType.TEAM_PREVIEW
            and not (
                decision_lines
                and len(decision_lines[0].parts) >= 2
                and decision_lines[0].parts[1] == "turn"
            )
            and not _perspective_has_choice(decision_lines, perspective)
            and _perspective_has_choice(decision_lines, 1 - perspective)
            and not _is_terminal_segment(decision_lines)
        )
        view = _view(
            state,
            perspective,
            preview=decision_type is DecisionType.TEAM_PREVIEW,
            effective_species=effective_species,
            forced_move_slots=forced_move_slots,
            wait=is_waiting,
        )
        # The live environment presents the state after the preceding
        # request's protocol messages and exposes exactly those messages as
        # this request's event window. Do not attach the current action's
        # consequences to the action that caused them.
        view.events = list(pending_events)
        observed_actions = (
            _preview_actions(state, lines, perspective)
            if decision_type is DecisionType.TEAM_PREVIEW
            else _observed_actions(
                state,
                decision_lines,
                perspective,
                counters,
                illusion_species_by_line,
            )
        )
        observed = list(observed_actions[:2])
        tags = list(observed_actions[2])
        if decision_type is DecisionType.TEAM_PREVIEW:
            tags.append("preview_selection_not_public")
        if any(action is not None and action.tag == "switch" for action in observed[:2]):
            if any(
                action is not None
                and any(candidate >= MOVE_START for candidate in action.candidates)
                for action in observed[:2]
            ):
                decision_type = DecisionType.PIVOT_SWITCH
            elif any(
                view.decision.slots[index].force_switch
                for index in (0, 1)
                if observed[index] is not None
            ):
                decision_type = DecisionType.FORCED_SWITCH
        elif is_waiting:
            decision_type = DecisionType.FORCED_PASS
            tags.append("forced_pass")
        for slot, action in enumerate(observed[:2]):
            if action is not None and action.action is not None:
                if action.action not in legal_actions(view.decision, slot):
                    counters["observed_illegal_action"] += 1
        evidence_view = view
        if pivot_segment:
            evidence_slots = list(view.decision.slots)
            for slot, action in enumerate(observed[:2]):
                if action is not None:
                    continue
                observed[slot] = ObservedAction(PASS_ACTION, tag="implicit_pass")
                evidence_slots[slot] = replace(
                    evidence_slots[slot],
                    active=False,
                    switch_slots=(),
                    move_targets=(),
                    force_switch=False,
                    forced_move=False,
                )
            evidence_view = replace(
                view,
                decision=replace(view.decision, slots=tuple(evidence_slots)),
            )
        request = EvidenceRequest(
            view=evidence_view.decision,
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
        events = _events_and_update(state, state_update_lines, counters)
        snapshots.append(
            ReconstructedSnapshot(
                decision_index=decision_index,
                turn=view.turn,
                pre_line_index=start,
                post_line_index=end,
                view=view,
                events=pending_events,
                raw_lines=tuple(line.raw for line in (decision_lines if pivot_segment else lines)),
                # Pivot replacement requests include an implicit pass for the
                # slot whose order was already consumed. Keep that adjusted
                # legality view separate from the observation view, whose
                # public state must remain identical to the live request.
                action_view=evidence_view.decision,
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
