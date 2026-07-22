"""Series-context resampler for Best-of-3 games.

Compresses un-truncated turn histories [z_1, ..., z_T] from prior completed games
into a fixed number of continuous latent summary tokens (4 per game, 8 total for
up to 2 prior games) using a Perceiver Cross-Attention bottleneck with normalized
chronological progress embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from p0.model.architecture_contract import (
    MAX_PRIOR_GAMES,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.model.swiglu_encoder import SwiGLUTransformerEncoder


class DynamicSeriesResampler(nn.Module):
    """Perceiver Cross-Attention Resampler for dynamic Best-of-3 game summaries.

    Compresses an un-truncated turn history sequence [z_1, ..., z_T] of length T
    into K=4 continuous latent summary tokens per game. For up to MAX_PRIOR_GAMES (2)
    prior games, packs the result into SERIES_SLOTS (8) tokens for the memory channel.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_summary_tokens: int = SERIES_TOKENS_PER_GAME,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_summary_tokens = num_summary_tokens

        self.summary_queries = nn.Parameter(torch.empty(1, num_summary_tokens, d_model))
        self.empty_game_context = nn.Parameter(torch.empty(1, num_summary_tokens, d_model))

        self.pos_emb = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        gain = self.d_model**-0.5
        init.normal_(self.summary_queries, std=gain)
        init.normal_(self.empty_game_context, std=gain)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def resample_single_game(
        self,
        history: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Resample a single game's dynamic turn history (B, T, d_model) into (B, 4, d_model)."""
        if history.dim() != 3 or history.size(2) != self.d_model:
            raise ValueError(
                f"Expected history tensor of shape (B, T, {self.d_model}); got {tuple(history.shape)}"
            )
        batch_size, seq_len, _ = history.shape
        device = history.device

        if seq_len == 0:
            return self.empty_game_context.expand(batch_size, -1, -1).to(device)

        positions = torch.linspace(0.0, 1.0, steps=seq_len, device=device).view(1, seq_len, 1)
        pos_vectors = self.pos_emb(positions)
        keys_values = self.norm_k(history + pos_vectors)

        queries = self.norm_q(self.summary_queries.expand(batch_size, -1, -1))

        key_padding_mask = ~history_mask if history_mask is not None else None
        compressed, _ = self.cross_attn(
            query=queries,
            key=keys_values,
            value=keys_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        compressed = queries + compressed

        return self.self_attn(compressed)

    def forward(
        self,
        prior_game_histories: Sequence[Any] | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode prior completed games into series_tokens (B, 8, d_model) and series_mask (B, 8)."""
        if isinstance(prior_game_histories, Tensor):
            if (
                prior_game_histories.dim() == 4
                and prior_game_histories.size(2) == self.num_summary_tokens
            ):
                B, G, K, D = prior_game_histories.shape
                tokens = prior_game_histories.reshape(B, G * K, D)
                mask = torch.ones((B, G * K), dtype=torch.bool, device=tokens.device)
                return tokens, mask
            if (
                prior_game_histories.dim() == 3
                and prior_game_histories.size(1) == SERIES_SLOTS
                and prior_game_histories.size(2) == self.d_model
            ):
                mask = torch.ones(
                    (prior_game_histories.size(0), SERIES_SLOTS),
                    dtype=torch.bool,
                    device=prior_game_histories.device,
                )
                return prior_game_histories, mask
            raise ValueError(
                f"Unexpected tensor shape {tuple(prior_game_histories.shape)} for series context"
            )

        device = self.summary_queries.device
        if not prior_game_histories:
            tokens = torch.zeros((1, SERIES_SLOTS, self.d_model), device=device)
            mask = torch.zeros((1, SERIES_SLOTS), dtype=torch.bool, device=device)
            return tokens, mask

        first_item = prior_game_histories[0]
        if isinstance(first_item, (list, tuple)):
            batch_size = len(prior_game_histories)
            series_tokens = torch.zeros((batch_size, SERIES_SLOTS, self.d_model), device=device)
            series_mask = torch.zeros((batch_size, SERIES_SLOTS), dtype=torch.bool, device=device)
            for episode_idx, ep_histories in enumerate(prior_game_histories):
                if not ep_histories:
                    continue
                for slot_idx, game_history in enumerate(ep_histories[:MAX_PRIOR_GAMES]):
                    if game_history is None:
                        continue
                    if game_history.dim() == 2:
                        game_history = game_history.unsqueeze(0)
                    if game_history.size(1) == 0:
                        continue
                    resampled = self.resample_single_game(game_history)
                    start = slot_idx * self.num_summary_tokens
                    end = start + self.num_summary_tokens
                    series_tokens[episode_idx, start:end] = resampled[0]
                    series_mask[episode_idx, start:end] = True
            return series_tokens, series_mask

        first_tensor = next((h for h in prior_game_histories if isinstance(h, Tensor)), None)
        if first_tensor is None:
            tokens = torch.zeros((1, SERIES_SLOTS, self.d_model), device=device)
            mask = torch.zeros((1, SERIES_SLOTS), dtype=torch.bool, device=device)
            return tokens, mask

        if first_tensor.dim() == 2:
            first_tensor = first_tensor.unsqueeze(0)

        batch_size = first_tensor.size(0)
        device = first_tensor.device

        series_tokens = torch.zeros((batch_size, SERIES_SLOTS, self.d_model), device=device)
        series_mask = torch.zeros((batch_size, SERIES_SLOTS), dtype=torch.bool, device=device)

        for slot_idx, game_history in enumerate(prior_game_histories[:MAX_PRIOR_GAMES]):
            if not isinstance(game_history, Tensor):
                continue
            if game_history.dim() == 2:
                game_history = game_history.unsqueeze(0)
            if game_history.size(1) == 0:
                continue
            resampled = self.resample_single_game(game_history)
            start = slot_idx * self.num_summary_tokens
            end = start + self.num_summary_tokens
            series_tokens[:, start:end] = resampled
            series_mask[:, start:end] = True

        return series_tokens, series_mask
