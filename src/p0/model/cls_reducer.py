"""Stateless memory reducer for the fixed memory-channel layout."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from p0.model.architecture_contract import (
    CURRENT_REDUCER_TOKEN_COUNT,
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
)
from p0.model.structured_observation import POKEMON_TOKENS
from p0.model.swiglu_encoder import SwiGLUTransformerEncoder


class ReducerOutput(NamedTuple):
    """The outputs needed by actor, critic, and runtime orchestration."""

    cls: Tensor
    pokemon: Tensor
    local_history_token: Tensor


def pack_history_tokens(history_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pack chronological history into the fixed 48-slot reducer input.

    ``history_tokens`` is ordered oldest to newest. The returned age identity
    is zero for the newest valid token and increases toward the oldest token.
    """
    if history_tokens.dim() != 3 or history_tokens.size(1) > HISTORY_WINDOW:
        raise ValueError(
            f"history_tokens must have shape (B, N <= {HISTORY_WINDOW}, D); "
            f"got {tuple(history_tokens.shape)}"
        )
    batch, count, width = history_tokens.shape
    packed = history_tokens.new_zeros((batch, HISTORY_WINDOW, width))
    mask = torch.zeros((batch, HISTORY_WINDOW), dtype=torch.bool, device=history_tokens.device)
    ages = torch.zeros((batch, HISTORY_WINDOW), dtype=torch.long, device=history_tokens.device)
    if count:
        packed[:, -count:] = history_tokens
        mask[:, -count:] = True
        ages[:, -count:] = torch.arange(count - 1, -1, -1, device=history_tokens.device)
    return packed, mask, ages


class MemoryReducer(nn.Module):
    """Reduce current observations with explicit series and battle history."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayer: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.local_summary_query = nn.Parameter(torch.empty(1, 1, d_model))
        self.local_summary_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.series_slot_emb = nn.Embedding(SERIES_SLOTS, d_model)
        self.history_age_emb = nn.Embedding(HISTORY_WINDOW, d_model)
        self.segment_emb = nn.Embedding(3, d_model)
        self.current_position_emb = nn.Embedding(CURRENT_REDUCER_TOKEN_COUNT, d_model)
        self.encoder = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=nlayer,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        gain = self.d_model**-0.5
        init.normal_(self.local_summary_query, std=gain)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, std=gain)

    def _validate_inputs(
        self,
        current_tokens: Tensor,
        series_tokens: Tensor,
        series_mask: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
        history_age_ids: Tensor,
    ) -> None:
        if current_tokens.dim() != 3 or current_tokens.shape[1:] != (
            CURRENT_TOKEN_COUNT,
            self.d_model,
        ):
            raise ValueError(
                f"Expected current tokens (B, {CURRENT_TOKEN_COUNT}, {self.d_model}); "
                f"got {tuple(current_tokens.shape)}"
            )
        batch = current_tokens.size(0)
        expected = (batch, SERIES_SLOTS, self.d_model)
        if series_tokens.shape != expected or series_mask.shape != (batch, SERIES_SLOTS):
            raise ValueError("series tokens or mask do not match the series slot contract")
        expected_history = (batch, HISTORY_WINDOW, self.d_model)
        if history_tokens.shape != expected_history:
            raise ValueError("history tokens do not match the fixed 48-slot contract")
        if history_mask.shape != (batch, HISTORY_WINDOW) or history_age_ids.shape != (
            batch,
            HISTORY_WINDOW,
        ):
            raise ValueError("history mask and age ids must have shape (B, 48)")
        if history_age_ids.numel() and (
            history_age_ids.min() < 0 or history_age_ids.max() >= HISTORY_WINDOW
        ):
            raise ValueError("history age ids must be in [0, 48)")

    def forward(
        self,
        current_tokens: Tensor,
        series_tokens: Tensor,
        series_mask: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
        history_age_ids: Tensor,
    ) -> ReducerOutput:
        """Run full attention over the fixed memory window."""
        self._validate_inputs(
            current_tokens,
            series_tokens,
            series_mask,
            history_tokens,
            history_mask,
            history_age_ids,
        )
        device = current_tokens.device
        batch = current_tokens.size(0)

        local_summary = self.local_summary(current_tokens)

        series_slots = torch.arange(SERIES_SLOTS, device=device)
        series = (
            series_tokens + self.series_slot_emb(series_slots)[None] + self.segment_emb.weight[0]
        )
        history = (
            history_tokens + self.history_age_emb(history_age_ids) + self.segment_emb.weight[1]
        )
        current = torch.cat([local_summary[:, None], current_tokens], dim=1)
        current = current + self.current_position_emb.weight[None] + self.segment_emb.weight[2]

        sequence = torch.cat([series, history, current], dim=1)
        padding = torch.cat(
            [
                ~series_mask.bool(),
                ~history_mask.bool(),
                torch.zeros(batch, CURRENT_REDUCER_TOKEN_COUNT, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        if sequence.size(1) != REDUCER_MAX_LENGTH:
            raise RuntimeError(f"Reducer layout drifted to {sequence.size(1)} tokens")
        encoded = self.encoder(sequence, src_key_padding_mask=padding)

        current_start = SERIES_SLOTS + HISTORY_WINDOW
        return ReducerOutput(
            cls=encoded[:, current_start],
            pokemon=encoded[:, current_start + 1 : current_start + 1 + len(POKEMON_TOKENS)],
            local_history_token=local_summary,
        )

    def local_summary(self, current_tokens: Tensor) -> Tensor:
        """Summarize current tokens before any memory interaction."""
        if current_tokens.dim() != 3 or current_tokens.shape[1:] != (
            CURRENT_TOKEN_COUNT,
            self.d_model,
        ):
            raise ValueError("current tokens do not match the fixed 24-token contract")
        batch = current_tokens.size(0)
        query = self.local_summary_query.expand(batch, -1, -1)
        summary, _ = self.local_summary_attn(
            query, current_tokens, current_tokens, need_weights=False
        )
        return summary[:, 0]
