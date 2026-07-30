"""Ordered raw-token history and Bo3 context preparation for BC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import (
    MAX_PRIOR_GAMES,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.training._bc_batch import BCGameWindow

SeriesResampler = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class _BCHistoryUpdate:
    series_key: SeriesPerspectiveKey
    game_number: int
    tokens: Tensor
    is_game_end: bool
    is_series_end: bool


@dataclass(frozen=True, slots=True)
class _PreparedSeriesContext:
    tokens: Tensor
    mask: Tensor
    updates: tuple[_BCHistoryUpdate, ...]


@dataclass(slots=True)
class _SeriesHistoryState:
    completed_games: list[tuple[int, Tensor]] = field(default_factory=list)
    active_game_number: int | None = None
    active_fragments: list[Tensor] = field(default_factory=list)
    ended: bool = False

    def planning_copy(self, reference: Tensor) -> _SeriesHistoryState:
        return _SeriesHistoryState(
            completed_games=list(self.completed_games),
            active_game_number=self.active_game_number,
            active_fragments=[
                fragment.to(device=reference.device, dtype=reference.dtype)
                for fragment in self.active_fragments
            ],
            ended=self.ended,
        )

    def append(
        self,
        game_number: int,
        tokens: Tensor,
        *,
        is_game_end: bool,
        is_series_end: bool,
    ) -> None:
        if self.ended:
            raise ValueError("A BC batch contains data after a perspective-series ended")
        if is_series_end and not is_game_end:
            raise ValueError("A BC series can end only at a game boundary")

        if self.active_game_number is None:
            if self.completed_games and game_number <= self.completed_games[-1][0]:
                raise ValueError("BC game numbers must increase within a perspective-series")
            self.active_game_number = game_number
        elif game_number != self.active_game_number:
            raise ValueError("A BC perspective-game changed before its previous game ended")

        self.active_fragments.append(tokens)
        if is_game_end:
            completed_tokens = torch.cat(self.active_fragments)
            self.completed_games.append((game_number, completed_tokens))
            self.active_game_number = None
            self.active_fragments = []

        if is_series_end:
            self.completed_games = []
            self.ended = True


class _BCSeriesHistory:
    """Retain detached tokens across updates and prepare ordered series context."""

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self._states: dict[SeriesPerspectiveKey, _SeriesHistoryState] = {}

    @property
    def has_partial_games(self) -> bool:
        return any(state.active_game_number is not None for state in self._states.values())

    def prepare(
        self,
        windows: tuple[BCGameWindow, ...],
        target_tokens: Tensor,
        resample_game: SeriesResampler,
    ) -> _PreparedSeriesContext:
        working_states: dict[SeriesPerspectiveKey, _SeriesHistoryState] = {}
        histories: list[Tensor] = []
        history_rows: dict[int, int] = {}
        window_history_rows: list[tuple[int, ...]] = []
        updates: list[_BCHistoryUpdate] = []

        for window in windows:
            state = working_states.get(window.series_key)
            if state is None:
                persistent = self._states.get(window.series_key, _SeriesHistoryState())
                state = persistent.planning_copy(target_tokens)
                working_states[window.series_key] = state

            prior_rows: list[int] = []
            for _, prior_tokens in state.completed_games[-MAX_PRIOR_GAMES:]:
                identity = id(prior_tokens)
                row = history_rows.get(identity)
                if row is None:
                    row = len(histories)
                    history_rows[identity] = row
                    histories.append(
                        prior_tokens.to(
                            device=target_tokens.device,
                            dtype=target_tokens.dtype,
                        )
                    )
                prior_rows.append(row)
            window_history_rows.append(tuple(prior_rows))

            token_chunk = target_tokens[window.batch_start : window.batch_stop]
            state.append(
                window.game_number,
                token_chunk,
                is_game_end=window.is_game_end,
                is_series_end=window.is_series_end,
            )
            updates.append(
                _BCHistoryUpdate(
                    series_key=window.series_key,
                    game_number=window.game_number,
                    tokens=token_chunk,
                    is_game_end=window.is_game_end,
                    is_series_end=window.is_series_end,
                )
            )

        summaries = _resample_histories(histories, target_tokens, resample_game)
        series_tokens, series_mask = _pack_series_context(
            windows,
            window_history_rows,
            summaries,
            target_tokens,
        )
        return _PreparedSeriesContext(
            tokens=series_tokens,
            mask=series_mask,
            updates=tuple(updates),
        )

    def apply(self, updates: tuple[_BCHistoryUpdate, ...]) -> None:
        for update in updates:
            tokens = update.tokens.detach().to(device="cpu", dtype=torch.float32)
            if tokens.dim() != 2 or tokens.shape[1] != self.d_model:
                raise ValueError("BC game history tokens do not match the policy width")
            state = self._states.setdefault(update.series_key, _SeriesHistoryState())
            state.append(
                update.game_number,
                tokens,
                is_game_end=update.is_game_end,
                is_series_end=update.is_series_end,
            )

    def clear(self) -> None:
        self._states.clear()


def _resample_histories(
    histories: list[Tensor],
    reference: Tensor,
    resample_game: SeriesResampler,
) -> Tensor:
    if not histories:
        return reference.new_empty((0, SERIES_TOKENS_PER_GAME, reference.size(-1)))

    lengths = torch.tensor(
        [history.size(0) for history in histories],
        device=reference.device,
    )
    padded = pad_sequence(histories, batch_first=True)
    positions = torch.arange(padded.size(1), device=reference.device)
    history_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return resample_game(padded, history_mask)


def _pack_series_context(
    windows: tuple[BCGameWindow, ...],
    window_history_rows: list[tuple[int, ...]],
    summaries: Tensor,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    window_tokens: list[Tensor] = []
    window_masks: list[Tensor] = []
    window_lengths: list[int] = []
    for window, history_rows in zip(windows, window_history_rows, strict=True):
        used_slots = len(history_rows) * SERIES_TOKENS_PER_GAME
        if history_rows:
            selected = summaries[list(history_rows)].flatten(0, 1)
            padding = reference.new_zeros((SERIES_SLOTS - used_slots, reference.size(-1)))
            tokens = torch.cat((selected, padding))
        else:
            tokens = reference.new_zeros((SERIES_SLOTS, reference.size(-1)))
        mask = torch.arange(SERIES_SLOTS, device=reference.device) < used_slots
        window_tokens.append(tokens)
        window_masks.append(mask)
        window_lengths.append(window.batch_stop - window.batch_start)

    repeats = torch.tensor(window_lengths, device=reference.device)
    return (
        torch.repeat_interleave(torch.stack(window_tokens), repeats, dim=0),
        torch.repeat_interleave(torch.stack(window_masks), repeats, dim=0),
    )
