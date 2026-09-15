"""Storage for previous games in a Best-of-3 series during training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

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
_EMPTY_STATE: SeriesStateSnapshot = (), None, (), False


def _norm_key(key: SeriesHistoryKey) -> str:
    if isinstance(key, SeriesPerspectiveKey):
        return f"{_SPK_PREFIX}{key.series_id}:{key.canonical_player}"
    if isinstance(key, str) and key:
        return key
    raise ValueError("Series history keys must be non-empty strings or perspective keys")


def advance_series_state(
    state: SeriesStateSnapshot,
    game_number: int,
    values: Tensor,
    *,
    is_game_end: bool,
    is_series_end: bool,
    max_games: int = MAX_PRIOR_GAMES,
) -> SeriesStateSnapshot:
    """Advance series history state for a game step or boundary."""
    completed_games, active_game_number, active_fragments, ended = state
    if ended:
        raise ValueError("A perspective-series cannot receive data after it ended")
    if is_series_end and not is_game_end:
        raise ValueError("A series can end only at a game boundary")

    if active_game_number is None:
        expected = completed_games[-1][0] + 1 if completed_games else 1
        if game_number != expected:
            raise ValueError("Game numbers must be consecutive within a perspective-series")
        active_game_number = game_number
    elif game_number != active_game_number:
        raise ValueError("A perspective-game changed before its previous game ended")

    if is_series_end:
        return (), None, (), True
    if not is_game_end:
        return completed_games, active_game_number, (*active_fragments, values), False

    completed_game = torch.cat((*active_fragments, values), dim=0) if active_fragments else values
    completed_games = (*completed_games, (game_number, completed_game))[-max_games:]
    return completed_games, None, (), False


class SeriesHistoryStore:
    """Stores decision summaries and active turns for prior games in a series."""

    def __init__(self, d_model: int, max_games: int = MAX_PRIOR_GAMES) -> None:
        if d_model <= 0 or not 0 < max_games <= MAX_PRIOR_GAMES:
            raise ValueError(
                f"Series history width must be positive and max_games must be in [1, {MAX_PRIOR_GAMES}]"
            )
        self.d_model = d_model
        self.max_games = max_games
        self._states: dict[str, SeriesStateSnapshot] = {}

    @property
    def has_partial_games(self) -> bool:
        """Return whether any series currently has an unfinished game."""
        return any(state[1] is not None for state in self._states.values())

    def next_game_number(self, key: SeriesHistoryKey) -> int:
        """Return the game number expected for the next fragment under key."""
        state = self._states.get(_norm_key(key))
        if state is None:
            return 1
        completed, active_game, _, _ = state
        if active_game is not None:
            return active_game
        return (completed[-1][0] + 1) if completed else 1

    def snapshot(self, key: SeriesHistoryKey) -> SeriesHistorySnapshot:
        """Return prior completed games in chronological order as immutable references."""
        state = self._states.get(_norm_key(key))
        if state is None or state[3]:
            return ()
        return tuple(values for _, values in state[0][-self.max_games :])

    def planning_state(self, key: SeriesHistoryKey) -> SeriesStateSnapshot:
        """Return state views for prior-game simulation."""
        return self._states.get(_norm_key(key), _EMPTY_STATE)

    def snapshot_keys(
        self,
        keys: Iterable[SeriesHistoryKey],
    ) -> dict[str, SeriesStateSnapshot | None]:
        """Snapshot history states for selected series keys for rollback."""
        snapshots: dict[str, SeriesStateSnapshot | None] = {}
        for key in keys:
            normalized = _norm_key(key)
            if normalized in snapshots:
                continue
            snapshots[normalized] = self._states.get(normalized)
        return snapshots

    def restore_keys(self, snapshots: Mapping[str, SeriesStateSnapshot | None]) -> None:
        """Restore series history states from a snapshot."""
        for key, snapshot in snapshots.items():
            if snapshot is None:
                self._states.pop(key, None)
            else:
                self._states[key] = snapshot

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

        retained = values
        if not is_series_end:
            retained = (
                values.detach().clone()
                if (values.device.type == "cpu" and values.dtype == torch.float32)
                else values.detach().to(device="cpu", dtype=torch.float32).clone()
            )
        self._states[norm_key] = advance_series_state(
            self._states.get(norm_key, _EMPTY_STATE),
            game_number,
            retained,
            is_game_end=is_game_end,
            is_series_end=is_series_end,
            max_games=self.max_games,
        )

    def drop(self, key: SeriesHistoryKey) -> None:
        """Remove one series while existing snapshots remain valid by reference."""
        self._states.pop(_norm_key(key), None)

    def clear(self) -> None:
        """Remove all series histories."""
        self._states.clear()

    def discard_active_games(self) -> None:
        """Discard unfinished games while retaining completed prior-game histories."""
        for key, (completed, _, _, ended) in self._states.items():
            self._states[key] = completed, None, (), ended

    def training_state(self) -> dict[str, dict[str, object]]:
        """Capture series history for checkpointing."""
        return {
            key: {
                "completed_games": tuple((number, values.clone()) for number, values in completed),
                "active_game_number": active_game,
                "active_fragments": tuple(values.clone() for values in fragments),
                "ended": ended,
            }
            for key, (completed, active_game, fragments, ended) in self._states.items()
        }

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore series history captured by training_state."""
        restored: dict[str, SeriesStateSnapshot] = {}
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

            restored[raw_key] = (
                tuple(completed),
                raw_active_number,
                tuple(fragments),
                raw_ended,
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
