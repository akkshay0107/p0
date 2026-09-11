"""Storage for previous games in a Best-of-3 series during training."""

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

_SPK_PREFIX = "__spk__:"


def _norm_key(key: SeriesHistoryKey) -> str:
    if isinstance(key, SeriesPerspectiveKey):
        return f"{_SPK_PREFIX}{key.series_id}:{key.canonical_player}"
    if isinstance(key, str) and key:
        return key
    raise ValueError("Series history keys must be non-empty strings or perspective keys")


@dataclass(slots=True)
class _SeriesState:
    completed_games: list[tuple[int, Tensor]] = field(default_factory=list)
    active_game_number: int | None = None
    active_fragments: list[Tensor] = field(default_factory=list)
    ended: bool = False


class SeriesHistoryStore:
    """Stores decision summaries and active turns for prior games in a series."""

    def __init__(self, d_model: int, max_games: int = MAX_PRIOR_GAMES) -> None:
        if d_model <= 0 or not 0 < max_games <= MAX_PRIOR_GAMES:
            raise ValueError(
                f"Series history width must be positive and max_games must be in [1, {MAX_PRIOR_GAMES}]"
            )
        self.d_model = d_model
        self.max_games = max_games
        self._states: dict[str, _SeriesState] = {}

    @property
    def has_partial_games(self) -> bool:
        """Return whether any series currently has an unfinished game."""
        return any(s.active_game_number is not None for s in self._states.values())

    def next_game_number(self, key: SeriesHistoryKey) -> int:
        """Return the game number expected for the next fragment under key."""
        state = self._states.get(_norm_key(key))
        if state is None:
            return 1
        if state.active_game_number is not None:
            return state.active_game_number
        return (state.completed_games[-1][0] + 1) if state.completed_games else 1

    def snapshot(self, key: SeriesHistoryKey) -> SeriesHistorySnapshot:
        """Return prior completed games in chronological order as immutable references."""
        state = self._states.get(_norm_key(key))
        if state is None or state.ended:
            return ()
        return tuple(val for _, val in state.completed_games[-self.max_games :])

    def planning_state(self, key: SeriesHistoryKey) -> SeriesStateSnapshot:
        """Return state views for prior-game simulation."""
        state = self._states.get(_norm_key(key))
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
        """Append a decision summary fragment to a series history."""
        norm_key = _norm_key(key)
        if type(game_number) is not int or not 1 <= game_number <= 3:
            raise ValueError("game_number must be an integer in [1, 3]")
        self._validate_tensor(values)
        if type(is_game_end) is not bool or type(is_series_end) is not bool:
            raise ValueError("game boundary flags must be booleans")
        if is_series_end and not is_game_end:
            raise ValueError("A series can end only at a game boundary")

        state = self._states.setdefault(norm_key, _SeriesState())
        if state.ended:
            raise ValueError("A perspective-series cannot receive data after it ended")

        if state.active_game_number is None:
            expected = state.completed_games[-1][0] + 1 if state.completed_games else 1
            if game_number != expected:
                raise ValueError("Game numbers must be consecutive within a perspective-series")
            state.active_game_number = game_number
        elif game_number != state.active_game_number:
            raise ValueError("A perspective-game changed before its previous game ended")

        detached = (
            values.detach().clone()
            if (values.device.type == "cpu" and values.dtype == torch.float32)
            else values.detach().to(device="cpu", dtype=torch.float32).clone()
        )
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
        """Remove one series while existing snapshots remain valid by reference."""
        self._states.pop(_norm_key(key), None)

    def clear(self) -> None:
        """Remove all series histories."""
        self._states.clear()

    def discard_active_games(self) -> None:
        """Discard unfinished games while retaining completed prior-game histories."""
        for state in self._states.values():
            state.active_game_number = None
            state.active_fragments.clear()

    def training_state(self) -> dict[str, dict[str, object]]:
        """Capture series history for checkpointing."""
        return {
            key: {
                "completed_games": tuple((n, t.clone()) for n, t in s.completed_games),
                "active_game_number": s.active_game_number,
                "active_fragments": tuple(t.clone() for t in s.active_fragments),
                "ended": s.ended,
            }
            for key, s in self._states.items()
        }

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore series history captured by training_state."""
        restored: dict[str, _SeriesState] = {}
        for raw_key, raw_val in state.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError("Series history checkpoint keys must be non-empty strings")
            if not isinstance(raw_val, Mapping):
                raise ValueError("Series history checkpoint entries must be mappings")

            raw_completed = raw_val.get("completed_games")
            raw_active_number = raw_val.get("active_game_number")
            raw_fragments = raw_val.get("active_fragments")
            raw_ended = raw_val.get("ended")

            if not isinstance(raw_completed, Sequence) or not isinstance(raw_fragments, Sequence):
                raise ValueError("Series history checkpoint state is malformed")
            if raw_active_number is not None and (
                type(raw_active_number) is not int or not 1 <= raw_active_number <= 3
            ):
                raise ValueError("Series history active game number is invalid")
            if type(raw_ended) is not bool:
                raise ValueError("Series history ended flag is invalid")

            completed: list[tuple[int, Tensor]] = []
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
                if completed and number != completed[-1][0] + 1:
                    raise ValueError("Series history completed games are not consecutive")
                self._validate_tensor(tensor)
                completed.append((number, tensor.detach().to("cpu", torch.float32).clone()))

            if len(completed) > self.max_games:
                raise ValueError("Series history checkpoint contains too many completed games")

            fragments: list[Tensor] = []
            for tensor in raw_fragments:
                if not isinstance(tensor, Tensor):
                    raise ValueError("Series history active fragment is not a tensor")
                self._validate_tensor(tensor)
                fragments.append(tensor.detach().to("cpu", torch.float32).clone())

            if raw_active_number is not None:
                expected = completed[-1][0] + 1 if completed else 1
                if raw_active_number != expected or not fragments:
                    raise ValueError("Series history active game chronology is invalid")
            elif fragments:
                raise ValueError("Series history has active fragments without an active game")

            restored[raw_key] = _SeriesState(
                completed_games=completed,
                active_game_number=raw_active_number,
                active_fragments=fragments,
                ended=raw_ended,
            )
        self._states = restored

    def _validate_tensor(self, tensor: Tensor) -> None:
        if (
            not isinstance(tensor, Tensor)
            or tensor.dim() != 2
            or tensor.size(1) != self.d_model
            or tensor.size(0) == 0
        ):
            raise ValueError(
                f"values must have shape (decisions, {self.d_model}) with decisions > 0"
            )
