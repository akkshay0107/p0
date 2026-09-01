"""Detached prior-game local-summary storage for differentiable training replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import MAX_PRIOR_GAMES

SeriesHistoryKey = str | SeriesPerspectiveKey
SeriesHistorySnapshot = tuple[Tensor, ...]
SeriesStateSnapshot = tuple[
    tuple[tuple[int, Tensor], ...],
    int | None,
    tuple[Tensor, ...],
    bool,
]


@dataclass(slots=True)
class _SeriesState:
    completed_games: list[tuple[int, Tensor]] = field(default_factory=list)
    active_game_number: int | None = None
    active_fragments: list[Tensor] = field(default_factory=list)
    ended: bool = False


class SeriesHistoryStore:
    """Retain detached full-game summaries and their active fragments."""

    def __init__(self, d_model: int, max_games: int = MAX_PRIOR_GAMES) -> None:
        if d_model <= 0 or not 0 < max_games <= MAX_PRIOR_GAMES:
            raise ValueError(
                f"Series history width must be positive and max_games must be in [1, {MAX_PRIOR_GAMES}]"
            )
        self.d_model = d_model
        self.max_games = max_games
        self._states: dict[SeriesHistoryKey, _SeriesState] = {}

    @property
    def has_partial_games(self) -> bool:
        """Return whether any perspective currently has an unfinished game."""
        return any(state.active_game_number is not None for state in self._states.values())

    def next_game_number(self, key: SeriesHistoryKey) -> int:
        """Return the game number expected for the next fragment under key."""
        self._validate_key(key)
        state = self._states.get(key)
        if state is None:
            return 1
        if state.active_game_number is not None:
            return state.active_game_number
        if state.completed_games:
            return state.completed_games[-1][0] + 1
        return 1

    def snapshot(self, key: SeriesHistoryKey) -> SeriesHistorySnapshot:
        """Return prior completed games in chronological order as immutable references."""
        self._validate_key(key)
        state = self._states.get(key)
        if state is None or state.ended:
            return ()
        return tuple(value for _, value in state.completed_games[-self.max_games :])

    def planning_state(self, key: SeriesHistoryKey) -> SeriesStateSnapshot:
        """Return immutable state views for within-batch chronological planning."""
        self._validate_key(key)
        state = self._states.get(key)
        if state is None:
            return (), None, (), False
        return (
            tuple(state.completed_games),
            state.active_game_number,
            tuple(state.active_fragments),
            state.ended,
        )

    def append(
        self,
        key: SeriesHistoryKey,
        game_number: int,
        values: Tensor,
        *,
        is_game_end: bool,
        is_series_end: bool,
    ) -> None:
        """Append one detached local-summary fragment to a perspective-series."""
        self._validate_key(key)
        if type(game_number) is not int or not 1 <= game_number <= 3:
            raise ValueError("game_number must be an integer in [1, 3]")
        if (
            not isinstance(values, Tensor)
            or values.dim() != 2
            or values.size(1) != self.d_model
            or values.size(0) == 0
        ):
            raise ValueError(
                f"values must have shape (decisions, {self.d_model}) with decisions > 0"
            )
        if type(is_game_end) is not bool or type(is_series_end) is not bool:
            raise ValueError("game boundary flags must be booleans")
        if is_series_end and not is_game_end:
            raise ValueError("A series can end only at a game boundary")

        state = self._states.setdefault(key, _SeriesState())
        if state.ended:
            raise ValueError("A perspective-series cannot receive data after it ended")
        if state.active_game_number is None:
            expected_game_number = state.completed_games[-1][0] + 1 if state.completed_games else 1
            if game_number != expected_game_number:
                raise ValueError("Game numbers must be consecutive within a perspective-series")
            state.active_game_number = game_number
        elif game_number != state.active_game_number:
            raise ValueError("A perspective-game changed before its previous game ended")

        detached = values.detach().to(device="cpu", dtype=torch.float32).clone()
        state.active_fragments.append(detached)
        if is_game_end:
            completed = torch.cat(state.active_fragments, dim=0)
            state.completed_games.append((game_number, completed))
            state.completed_games = state.completed_games[-self.max_games :]
            state.active_game_number = None
            state.active_fragments.clear()

        if is_series_end:
            state.completed_games.clear()
            state.ended = True

    def drop(self, key: SeriesHistoryKey) -> None:
        """Remove one perspective-series while snapshots remain valid by reference."""
        self._validate_key(key)
        self._states.pop(key, None)

    def clear(self) -> None:
        """Remove all live series histories."""
        self._states.clear()

    def discard_active_games(self) -> None:
        """Discard unfinished games while retaining completed prior-game histories."""
        for state in self._states.values():
            state.active_game_number = None
            state.active_fragments.clear()

    def training_state(self) -> dict[str, dict[str, object]]:
        """Capture string-keyed state for a safe checkpoint boundary."""
        state: dict[str, dict[str, object]] = {}
        for key, value in self._states.items():
            if not isinstance(key, str):
                raise ValueError("Only string-keyed series history can be checkpointed")
            state[key] = {
                "completed_games": tuple(
                    (number, tensor.clone()) for number, tensor in value.completed_games
                ),
                "active_game_number": value.active_game_number,
                "active_fragments": tuple(tensor.clone() for tensor in value.active_fragments),
                "ended": value.ended,
            }
        return state

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore state captured by training_state."""
        restored: dict[SeriesHistoryKey, _SeriesState] = {}
        for key, raw_value in state.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Series history checkpoint keys must be non-empty strings")
            if not isinstance(raw_value, Mapping):
                raise ValueError("Series history checkpoint entries must be mappings")

            raw_completed = raw_value.get("completed_games")
            raw_active_number = raw_value.get("active_game_number")
            raw_fragments = raw_value.get("active_fragments")
            raw_ended = raw_value.get("ended")
            if not isinstance(raw_completed, Sequence) or not isinstance(raw_fragments, Sequence):
                raise ValueError("Series history checkpoint state is malformed")
            if raw_active_number is not None and (
                type(raw_active_number) is not int or not 1 <= raw_active_number <= 3
            ):
                raise ValueError("Series history active game number is invalid")
            if type(raw_ended) is not bool:
                raise ValueError("Series history ended flag is invalid")

            completed_games: list[tuple[int, Tensor]] = []
            for item in raw_completed:
                if (
                    not isinstance(item, Sequence)
                    or len(item) != 2
                    or type(item[0]) is not int
                    or not 1 <= item[0] <= 3
                    or not isinstance(item[1], Tensor)
                ):
                    raise ValueError("Series history completed game is malformed")
                number, tensor = item
                expected_number = completed_games[-1][0] + 1 if completed_games else number
                if completed_games and number != expected_number:
                    raise ValueError("Series history completed games are not consecutive")
                self._validate_stored_tensor(tensor)
                completed_games.append((number, tensor.detach().to("cpu", torch.float32).clone()))
            if len(completed_games) > self.max_games:
                raise ValueError("Series history checkpoint contains too many completed games")

            active_fragments: list[Tensor] = []
            for tensor in raw_fragments:
                if not isinstance(tensor, Tensor):
                    raise ValueError("Series history active fragment is not a tensor")
                self._validate_stored_tensor(tensor)
                active_fragments.append(tensor.detach().to("cpu", torch.float32).clone())

            if raw_active_number is not None:
                expected_number = completed_games[-1][0] + 1 if completed_games else 1
                if raw_active_number != expected_number or not active_fragments:
                    raise ValueError("Series history active game chronology is invalid")
            elif active_fragments:
                raise ValueError("Series history has active fragments without an active game")

            restored[key] = _SeriesState(
                completed_games=completed_games,
                active_game_number=raw_active_number,
                active_fragments=active_fragments,
                ended=raw_ended,
            )
        self._states = restored

    def _validate_stored_tensor(self, tensor: Tensor) -> None:
        if tensor.dim() != 2 or tensor.size(1) != self.d_model or tensor.size(0) == 0:
            raise ValueError("Series history tensor shape does not match the policy")

    @staticmethod
    def _validate_key(key: SeriesHistoryKey) -> None:
        if isinstance(key, str) and key:
            return
        if isinstance(key, SeriesPerspectiveKey):
            return
        raise ValueError("Series history keys must be non-empty strings or perspective keys")
