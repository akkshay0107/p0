"""Pure planning helpers for BC prior-game context."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

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
from p0.training.series_history import SeriesHistoryStore, SeriesStateSnapshot

SeriesResampler = Callable[[Tensor, Tensor], Tensor]


class _BCHistoryUpdate(NamedTuple):
    series_key: SeriesPerspectiveKey
    game_number: int
    tokens: Tensor
    is_game_end: bool
    is_series_end: bool


def _advance_state(
    state: SeriesStateSnapshot,
    game_number: int,
    tokens: Tensor,
    *,
    is_game_end: bool,
    is_series_end: bool,
) -> SeriesStateSnapshot:
    completed_games, active_game_number, active_fragments, ended = state
    if ended:
        raise ValueError("A BC perspective-series contains data after it ended")
    if is_series_end and not is_game_end:
        raise ValueError("A BC series can end only at a game boundary")

    completed = list(completed_games)
    fragments = list(active_fragments)
    if active_game_number is None:
        expected_game_number = completed[-1][0] + 1 if completed else 1
        if game_number != expected_game_number:
            raise ValueError("BC game numbers must be consecutive within a perspective-series")
        active_game_number = game_number
    elif game_number != active_game_number:
        raise ValueError("A BC perspective-game changed before its previous game ended")

    fragments.append(tokens)
    if is_game_end:
        completed.append((game_number, torch.cat(fragments, dim=0)))
        completed = completed[-MAX_PRIOR_GAMES:]
        active_game_number = None
        fragments = []

    if is_series_end:
        completed = []
        ended = True

    return tuple(completed), active_game_number, tuple(fragments), ended


def prepare_series_context(
    store: SeriesHistoryStore,
    windows: tuple[BCGameWindow, ...],
    target_tokens: Tensor,
    resample_game: SeriesResampler,
) -> tuple[Tensor, Tensor, tuple[_BCHistoryUpdate, ...]]:
    """Plan BC history updates and resample only detached completed games."""
    working_states: dict[SeriesPerspectiveKey, SeriesStateSnapshot] = {}
    histories: list[Tensor] = []
    history_rows: dict[int, int] = {}
    window_history_rows: list[tuple[int, ...]] = []
    updates: list[_BCHistoryUpdate] = []

    for window in windows:
        state = working_states.get(window.series_key)
        if state is None:
            state = store.planning_state(window.series_key)
            working_states[window.series_key] = state

        prior_rows: list[int] = []
        for _, prior_tokens in state[0][-MAX_PRIOR_GAMES:]:
            identity = id(prior_tokens)
            row = history_rows.get(identity)
            if row is None:
                row = len(histories)
                history_rows[identity] = row
                histories.append(
                    prior_tokens.detach().to(
                        device=target_tokens.device,
                        dtype=target_tokens.dtype,
                    )
                )
            prior_rows.append(row)
        window_history_rows.append(tuple(prior_rows))

        token_chunk = target_tokens[window.batch_start : window.batch_stop]
        working_states[window.series_key] = _advance_state(
            state,
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
    return series_tokens, series_mask, tuple(updates)


def commit_history_updates(
    store: SeriesHistoryStore,
    updates: tuple[_BCHistoryUpdate, ...],
) -> None:
    """Commit planned BC fragments only after their graph has been backpropagated."""
    for update in updates:
        store.append(
            update.series_key,
            update.game_number,
            update.tokens,
            is_game_end=update.is_game_end,
            is_series_end=update.is_series_end,
        )


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
        dtype=torch.long,
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
    if summaries.numel():
        row_indices = torch.full(
            (len(windows), MAX_PRIOR_GAMES),
            -1,
            dtype=torch.long,
            device=reference.device,
        )
        for index, rows in enumerate(window_history_rows):
            if rows:
                row_indices[index, : len(rows)] = torch.tensor(
                    rows,
                    dtype=torch.long,
                    device=reference.device,
                )
        valid = row_indices >= 0
        selected = summaries[row_indices.clamp_min(0)]
        selected = selected.masked_fill(~valid[:, :, None, None], 0.0)
        window_tokens = selected.flatten(1, 2)
    else:
        window_tokens = reference.new_zeros((len(windows), SERIES_SLOTS, reference.size(-1)))

    used_games = torch.tensor(
        [len(rows) for rows in window_history_rows],
        dtype=torch.long,
        device=reference.device,
    )
    series_mask = (
        torch.arange(SERIES_SLOTS, device=reference.device).unsqueeze(0)
        < used_games.unsqueeze(1) * SERIES_TOKENS_PER_GAME
    )
    window_lengths = torch.tensor(
        [window.batch_stop - window.batch_start for window in windows],
        dtype=torch.long,
        device=reference.device,
    )
    return (
        torch.repeat_interleave(window_tokens, window_lengths, dim=0),
        torch.repeat_interleave(series_mask, window_lengths, dim=0),
    )
