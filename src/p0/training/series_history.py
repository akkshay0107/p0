"""Storage for previous games in a Best-of-3 series during training."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import MAX_PRIOR_GAMES, SERIES_SLOTS, SERIES_TOKENS_PER_GAME

SeriesHistoryKey = str | SeriesPerspectiveKey
SeriesHistorySnapshot = tuple[Tensor, ...]
SeriesResampler = Callable[[Tensor, Tensor], Tensor]
CompletedGames = tuple[tuple[int, Tensor], ...]
SeriesStateSnapshot = tuple[CompletedGames, int | None, tuple[Tensor, ...], bool]

_SPK_PREFIX = "__spk__:"


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


def pad_game_histories(histories: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Pad game histories of shape (decisions, d_model) and mark their real rows."""
    padded = pad_sequence(list(histories), batch_first=True)
    lengths = torch.tensor([history.size(0) for history in histories], device=padded.device)
    positions = torch.arange(padded.size(1), device=padded.device)
    return padded, positions.unsqueeze(0) < lengths.unsqueeze(1)


def pack_series_context(
    prior_games: Sequence[Sequence[Tensor]],
    group_lengths: Tensor,
    resample_game: SeriesResampler,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Summarize prior games and repeat each group's series tokens over its rows.

    A group is one BC window or one PPO episode. A game tensor that several
    groups share is summarized once. The game tensors must share one device.

    Arguments:
      prior_games: for each group, its prior-game histories, oldest first.
      group_lengths: row count of each group, on the reference device.
      resample_game: compresses padded histories and their mask into series tokens.
      reference: tensor with one row per output row; gives device, dtype and width.

    Returns:
      Series tokens of shape (rows, SERIES_SLOTS, d_model) and their boolean mask.
    """
    device = reference.device
    rows = reference.size(0)
    histories: list[Tensor] = []
    history_rows: dict[int, int] = {}
    group_rows = [[-1] * MAX_PRIOR_GAMES for _ in prior_games]
    for slots, games in zip(group_rows, prior_games, strict=True):
        for slot, history in enumerate(games):
            row = history_rows.setdefault(id(history), len(histories))
            if row == len(histories):
                histories.append(history)
            slots[slot] = row

    if not histories:
        return (
            reference.new_zeros((rows, SERIES_SLOTS, reference.size(-1))),
            torch.zeros((rows, SERIES_SLOTS), dtype=torch.bool, device=device),
        )

    padded, history_mask = pad_game_histories(histories)
    summaries = resample_game(
        padded.to(device=device, dtype=reference.dtype), history_mask.to(device)
    )
    row_indices = torch.tensor(group_rows, dtype=torch.long, device=device)
    valid = row_indices >= 0
    selected = summaries[row_indices.clamp_min(0)].masked_fill(~valid[:, :, None, None], 0.0)
    series_mask = torch.arange(SERIES_SLOTS, device=device).unsqueeze(0) < (
        valid.sum(dim=1, keepdim=True) * SERIES_TOKENS_PER_GAME
    )
    return (
        torch.repeat_interleave(selected.flatten(1, 2), group_lengths, dim=0, output_size=rows),
        torch.repeat_interleave(series_mask, group_lengths, dim=0, output_size=rows),
    )


class SeriesHistoryStore:
    """Stores decision summaries of the completed prior games in a series."""

    def __init__(self, d_model: int, max_games: int = MAX_PRIOR_GAMES) -> None:
        if d_model <= 0 or not 0 < max_games <= MAX_PRIOR_GAMES:
            raise ValueError(
                f"Series history width must be positive and max_games must be in [1, {MAX_PRIOR_GAMES}]"
            )
        self.d_model = d_model
        self.max_games = max_games
        # Completed games and whether the series has ended, per series key.
        self._states: dict[str, tuple[CompletedGames, bool]] = {}

    def next_game_number(self, key: SeriesHistoryKey) -> int:
        """Return the game number expected for the next completed game under key."""
        completed, _ = self.planning_state(key)
        return (completed[-1][0] + 1) if completed else 1

    def snapshot(self, key: SeriesHistoryKey) -> SeriesHistorySnapshot:
        """Return prior completed games in chronological order as immutable references."""
        completed, ended = self.planning_state(key)
        if ended:
            return ()
        return tuple(values for _, values in completed[-self.max_games :])

    def planning_state(self, key: SeriesHistoryKey) -> tuple[CompletedGames, bool]:
        """Return the completed games and the series-ended flag for prior-game simulation."""
        return self._states.get(_norm_key(key), ((), False))

    def append(
        self,
        key: SeriesHistoryKey,
        game_number: int,
        values: Tensor,
        *,
        is_series_end: bool,
    ) -> None:
        """Append the decision summaries of one completed game to a series history."""
        norm_key = _norm_key(key)
        if type(game_number) is not int or not 1 <= game_number <= 3:
            raise ValueError("game_number must be an integer in [1, 3]")
        self._validate_tensor(values)
        if type(is_series_end) is not bool:
            raise ValueError("is_series_end must be a boolean")

        retained = values
        if not is_series_end:
            retained = (
                values.detach().clone()
                if (values.device.type == "cpu" and values.dtype == torch.float32)
                else values.detach().to(device="cpu", dtype=torch.float32).clone()
            )
        completed, ended = self.planning_state(key)
        completed, _, _, ended = advance_series_state(
            (completed, None, (), ended),
            game_number,
            retained,
            is_game_end=True,
            is_series_end=is_series_end,
            max_games=self.max_games,
        )
        self._states[norm_key] = completed, ended

    def drop(self, key: SeriesHistoryKey) -> None:
        """Remove one series while existing snapshots remain valid by reference."""
        self._states.pop(_norm_key(key), None)

    def clear(self) -> None:
        """Remove all series histories."""
        self._states.clear()

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
