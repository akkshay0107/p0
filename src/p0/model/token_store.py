"""
In-memory state manager for Bo3 series tokens.
Meant to be a simple backend that can be upgraded to a centralized store later.
THe player wrapper over the RL model currently holds an instance of this for
managing series tokens.
"""

from __future__ import annotations

import torch
from torch import Tensor

from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME


class SeriesTokenStore:
    """Manages the lifecycle of continuous series tokens for Bo3 matches.

    Acts as a pure state manager without neural network dependencies. Tokens
    should be computed by the caller (e.g. via PolicyNet.encode_series) and
    committed here.
    """

    def __init__(self, d_model: int, max_games: int = 2) -> None:
        self.d_model = d_model
        self.max_games = max_games
        # mapping from link_id -> list of (SERIES_TOKENS_PER_GAME, d_model) tensors
        self._store: dict[str, list[Tensor]] = {}

    def get_tokens(self, link_ids: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        """Returns batched (B, SERIES_SLOTS, d_model) tokens and mask for the requested links."""
        batch_size = len(link_ids)
        tokens = torch.zeros((batch_size, SERIES_SLOTS, self.d_model), device=device)
        mask = torch.zeros((batch_size, SERIES_SLOTS), dtype=torch.bool, device=device)

        for i, link_id in enumerate(link_ids):
            if link_id in self._store:
                game_tokens = self._store[link_id]
                for j, t in enumerate(game_tokens):
                    start = j * SERIES_TOKENS_PER_GAME
                    end = start + SERIES_TOKENS_PER_GAME
                    if end <= SERIES_SLOTS:
                        tokens[i, start:end] = t.to(device)
                        mask[i, start:end] = True

        return tokens, mask

    def append(self, link_id: str, new_game_tokens: Tensor) -> None:
        """Appends new_game_tokens to the series context for the given link_id."""
        expected_shape = (SERIES_TOKENS_PER_GAME, self.d_model)
        if new_game_tokens.shape != expected_shape:
            raise ValueError(
                f"Expected new_game_tokens to have shape {expected_shape}, got {tuple(new_game_tokens.shape)}"
            )

        # Detach and move to CPU to avoid pinning GPU memory indefinitely
        new_game_tokens = new_game_tokens.detach().to(device="cpu", dtype=torch.float32)

        history = self._store.pop(link_id, [])
        history.append(new_game_tokens)

        if len(history) > self.max_games:
            history = history[-self.max_games :]

        self._store[link_id] = history

    def drop(self, link_id: str) -> None:
        """Explicitly clear the tokens for a given link_id."""
        self._store.pop(link_id, None)

    def clear(self) -> None:
        """Clear the entire store."""
        self._store.clear()
