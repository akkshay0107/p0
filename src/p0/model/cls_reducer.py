"""Stateless memory reducer for the fixed memory-channel layout."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from p0.model.architecture_contract import (
    CURRENT_REDUCER_TOKEN_COUNT,
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
)
from p0.model.structured_observation import POKEMON_TOKENS
from p0.model.swiglu_encoder import AttentionPool, SwiGLUTransformerEncoder, initialize_module


class ReducerOutput(NamedTuple):
    """
    Outputs from memory reduction.

    cls is the readout after attending to memory. local_history_token is the
    pre-memory summary saved for subsequent turns so history stays flat.
    """

    cls: Tensor
    pokemon: Tensor
    local_history_token: Tensor


def pack_history_tokens(history_tokens: Tensor) -> tuple[Tensor, Tensor]:
    """
    Pack chronological history into the fixed 48-slot reducer input.

    history_tokens is ordered oldest to newest and right-aligned in the fixed
    history range, whose absolute positions encode chronology.
    """
    if history_tokens.dim() != 3 or history_tokens.size(1) > HISTORY_WINDOW:
        raise ValueError(
            f"history_tokens must have shape (B, N <= {HISTORY_WINDOW}, D); "
            f"got {tuple(history_tokens.shape)}"
        )
    batch, count, width = history_tokens.shape
    packed = history_tokens.new_zeros((batch, HISTORY_WINDOW, width))
    mask = history_tokens.new_zeros((batch, HISTORY_WINDOW), dtype=torch.bool)
    if count:
        packed[:, -count:] = history_tokens
        mask[:, -count:] = True
    return packed, mask


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
        self.local_summary_pool = AttentionPool(d_model, nhead)
        # Series, history, and current tokens occupy fixed, non-overlapping ranges.
        # One absolute table therefore encodes both position and segment identity.
        self.memory_position_emb = nn.Embedding(REDUCER_MAX_LENGTH, d_model)
        self.encoder = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=nlayer,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        initialize_module(self)
        self.local_summary_pool.reset_parameters()
        self.encoder.reset_parameters()

    def _validate_inputs(
        self,
        local_summary: Tensor,
        current_tokens: Tensor,
        series_tokens: Tensor,
        series_mask: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
    ) -> None:
        batch = current_tokens.size(0)
        if current_tokens.dim() != 3 or current_tokens.shape[1:] != (
            CURRENT_TOKEN_COUNT,
            self.d_model,
        ):
            raise ValueError(
                f"Expected current tokens (B, {CURRENT_TOKEN_COUNT}, {self.d_model}); "
                f"got {tuple(current_tokens.shape)}"
            )
        if (
            local_summary.shape != (batch, self.d_model)
            or local_summary.dtype != current_tokens.dtype
            or local_summary.device != current_tokens.device
        ):
            raise ValueError(
                f"Expected local summary ({batch}, {self.d_model}) matching "
                f"{current_tokens.dtype} on {current_tokens.device}; got "
                f"{tuple(local_summary.shape)} of {local_summary.dtype} on {local_summary.device}"
            )
        if series_tokens.shape != (batch, SERIES_SLOTS, self.d_model) or series_mask.shape != (
            batch,
            SERIES_SLOTS,
        ):
            raise ValueError("series tokens or mask do not match the series slot contract")
        if history_tokens.shape != (
            batch,
            HISTORY_WINDOW,
            self.d_model,
        ) or history_mask.shape != (batch, HISTORY_WINDOW):
            raise ValueError("history tokens or mask do not match the fixed 48-slot contract")

    def reduce(
        self,
        local_summary: Tensor,
        current_tokens: Tensor,
        series_tokens: Tensor,
        series_mask: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
    ) -> ReducerOutput:
        """
        Reduce the memory window from an already-computed local summary.

        Behavior cloning passes cached local summaries to avoid recomputing attention.
        The summary must match current_tokens, not a prior readout or another row.
        """
        self._validate_inputs(
            local_summary,
            current_tokens,
            series_tokens,
            series_mask,
            history_tokens,
            history_mask,
        )
        batch = current_tokens.size(0)
        device = current_tokens.device

        # Position 0 is seeded with the current-turn-only summary. The
        # transformer output at this position becomes the readout after it has
        # attended to history, series context, and all current tokens.
        sequence = torch.cat(
            [series_tokens, history_tokens, local_summary[:, None], current_tokens], dim=1
        )
        sequence = sequence + self.memory_position_emb.weight
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
        # Only the readout and Pokemon outputs are used; other rows still supply keys/values.
        current_start = SERIES_SLOTS + HISTORY_WINDOW
        encoded = self.encoder(
            sequence,
            src_key_padding_mask=padding,
            output_slice=slice(current_start, current_start + 1 + len(POKEMON_TOKENS)),
        )
        return ReducerOutput(
            # cls is the post-memory readout for this decision.
            cls=encoded[:, 0],
            pokemon=encoded[:, 1 : 1 + len(POKEMON_TOKENS)],
            # Store the pre-memory summary so each history entry remains a
            # local snapshot rather than recursively containing prior memory.
            local_history_token=local_summary,
        )

    def local_summary(self, current_tokens: Tensor) -> Tensor:
        """Summarize current tokens before attending to history."""
        if current_tokens.dim() != 3 or current_tokens.shape[1:] != (
            CURRENT_TOKEN_COUNT,
            self.d_model,
        ):
            raise ValueError(
                f"current tokens do not match the fixed {CURRENT_TOKEN_COUNT}-token contract"
            )
        return self.local_summary_pool(current_tokens)
