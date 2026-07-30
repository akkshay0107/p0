"""In-memory state manager for detached Bo3 series tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME

SeriesStoreKey = str | SeriesPerspectiveKey


class SeriesTokenStore:
    """Manages the lifecycle of continuous series tokens for Bo3 matches.

    Acts as a pure state manager without neural network dependencies. Tokens
    should be computed by the caller (e.g. via PolicyNet.encode_series) and
    committed here.
    """

    def __init__(self, d_model: int, max_games: int = 2) -> None:
        self.d_model = d_model
        self.max_games = max_games
        self._store: dict[SeriesStoreKey, list[Tensor]] = {}

    def get_tokens(
        self,
        keys: Sequence[SeriesStoreKey],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Return batched series tokens and masks for the requested keys."""
        batch_size = len(keys)
        tokens = torch.zeros((batch_size, SERIES_SLOTS, self.d_model), device=device)
        mask = torch.zeros((batch_size, SERIES_SLOTS), dtype=torch.bool, device=device)

        for i, key in enumerate(keys):
            if key in self._store:
                game_tokens = self._store[key]
                for j, t in enumerate(game_tokens):
                    start = j * SERIES_TOKENS_PER_GAME
                    end = start + SERIES_TOKENS_PER_GAME
                    if end <= SERIES_SLOTS:
                        tokens[i, start:end] = t.to(device)
                        mask[i, start:end] = True

        return tokens, mask

    def append(self, key: SeriesStoreKey, new_game_tokens: Tensor) -> None:
        """Append one completed game's tokens to a series state."""
        expected_shape = (SERIES_TOKENS_PER_GAME, self.d_model)
        if new_game_tokens.shape != expected_shape:
            raise ValueError(
                f"Expected new_game_tokens to have shape {expected_shape}, got {tuple(new_game_tokens.shape)}"
            )

        new_game_tokens = new_game_tokens.detach().to(device="cpu", dtype=torch.float32)

        history = self._store.pop(key, [])
        history.append(new_game_tokens)

        if len(history) > self.max_games:
            history = history[-self.max_games :]

        self._store[key] = history

    def drop(self, key: SeriesStoreKey) -> None:
        """Explicitly clear the tokens for a series state."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear the entire store."""
        self._store.clear()
