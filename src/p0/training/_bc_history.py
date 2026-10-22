"""History helpers for behavior cloning prior-game context."""

from __future__ import annotations

import torch
from torch import Tensor

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import MAX_PRIOR_GAMES
from p0.training._bc_batch import BCGameWindow
from p0.training.series_history import (
    SeriesHistoryStore,
    SeriesResampler,
    SeriesStateSnapshot,
    advance_series_state,
    pack_series_context,
)

# Game number and detached CPU fragments of each unfinished game, kept for one
# training update or evaluation pass; completed games move to the store.
ActiveGames = dict[SeriesPerspectiveKey, tuple[int, tuple[Tensor, ...]]]


def window_history_tokens(
    windows: tuple[BCGameWindow, ...],
    target_tokens: Tensor,
    context_tokens: Tensor,
) -> tuple[Tensor, ...]:
    """Return the local summaries each window adds to its game history, final board included."""
    return tuple(
        torch.cat(
            (
                target_tokens[window.batch_start : window.batch_stop],
                context_tokens[window.final_index : window.final_index + 1],
            )
        )
        if window.is_game_end
        else target_tokens[window.batch_start : window.batch_stop]
        for window in windows
    )


def prepare_series_context(
    store: SeriesHistoryStore,
    active: ActiveGames,
    windows: tuple[BCGameWindow, ...],
    window_tokens: tuple[Tensor, ...],
    target_tokens: Tensor,
    resample_game: SeriesResampler,
) -> tuple[Tensor, Tensor]:
    """Prepare prior-game series tokens and mask for the target rows of a batch of windows."""
    working_states: dict[SeriesPerspectiveKey, SeriesStateSnapshot] = {}
    # Stored games are on the CPU; each is moved to the target device once.
    moved: dict[int, Tensor] = {}
    prior_games: list[list[Tensor]] = []

    for window, chunk in zip(windows, window_tokens, strict=True):
        state = working_states.get(window.series_key)
        if state is None:
            games, ended = store.planning_state(window.series_key)
            game_number, fragments = active.get(window.series_key, (None, ()))
            state = (
                games,
                game_number,
                tuple(
                    fragment.to(device=target_tokens.device, dtype=target_tokens.dtype)
                    for fragment in fragments
                ),
                ended,
            )
            working_states[window.series_key] = state

        window_games: list[Tensor] = []
        for _, prior_tokens in state[0][-MAX_PRIOR_GAMES:]:
            identity = id(prior_tokens)
            if identity not in moved:
                moved[identity] = prior_tokens.to(
                    device=target_tokens.device, dtype=target_tokens.dtype
                )
            window_games.append(moved[identity])
        prior_games.append(window_games)

        working_states[window.series_key] = advance_series_state(
            state,
            window.game_number,
            chunk,
            is_game_end=window.is_game_end,
            is_series_end=window.is_series_end,
            max_games=store.max_games,
        )

    window_lengths = torch.tensor(
        [window.batch_stop - window.batch_start for window in windows],
        dtype=torch.long,
        device=target_tokens.device,
    )
    return pack_series_context(prior_games, window_lengths, resample_game, target_tokens)


def commit_history_updates(
    store: SeriesHistoryStore,
    active: ActiveGames,
    windows: tuple[BCGameWindow, ...],
    window_tokens: tuple[Tensor, ...],
) -> None:
    """Keep unfinished window tokens in active, and append completed games to the store."""
    for window, chunk in zip(windows, window_tokens, strict=True):
        _, fragments = active.pop(window.series_key, (window.game_number, ()))
        retained = chunk.detach().to(device="cpu", dtype=torch.float32, copy=True)
        if window.is_game_end:
            store.append(
                window.series_key,
                window.game_number,
                torch.cat((*fragments, retained)) if fragments else retained,
                is_series_end=window.is_series_end,
            )
        else:
            active[window.series_key] = window.game_number, (*fragments, retained)
