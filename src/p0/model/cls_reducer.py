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
    The outputs needed by actor, critic, and runtime orchestration.

    cls is the memory-aware readout used for the current policy/value
    decision. local_history_token is deliberately the pre-memory summary
    of the current observation; callers store that token for a later turn.
    Keeping these representations separate prevents history from becoming a
    recursively nested copy of the entire reducer context.
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
    mask = torch.zeros((batch, HISTORY_WINDOW), dtype=torch.bool, device=history_tokens.device)
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
        current_tokens: Tensor,
        series_tokens: Tensor,
        series_mask: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
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
        if history_mask.shape != (batch, HISTORY_WINDOW):
            raise ValueError("history mask must have shape (B, 48)")

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

        Behaviour cloning builds the per-decision local summaries to fill its
        history window, so it passes the target rows straight back in rather
        than paying for the same attention twice. The supplied summary must
        describe the same current observation as current_tokens; it must
        not be a previous cls output or a summary from another row.
        """
        self._validate_inputs(
            current_tokens,
            series_tokens,
            series_mask,
            history_tokens,
            history_mask,
        )
        device = current_tokens.device
        batch = current_tokens.size(0)
        if (
            local_summary.shape != (batch, self.d_model)
            or local_summary.dtype != current_tokens.dtype
            or local_summary.device != current_tokens.device
        ):
            raise ValueError(
                f"Expected a local summary of shape ({batch}, {self.d_model}) matching "
                f"{current_tokens.dtype} on {current_tokens.device}; got "
                f"{tuple(local_summary.shape)} of {local_summary.dtype} on {local_summary.device}"
            )

        # Position 0 is seeded with the current-turn-only summary. The
        # transformer output at this position becomes the readout after it has
        # attended to history, series context, and all current tokens.
        current = torch.cat([local_summary[:, None], current_tokens], dim=1)
        sequence = torch.cat([series_tokens, history_tokens, current], dim=1)
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
        encoded = self.encoder(sequence, src_key_padding_mask=padding)

        current_start = SERIES_SLOTS + HISTORY_WINDOW
        return ReducerOutput(
            # cls is the post-memory readout for this decision.
            cls=encoded[:, current_start],
            pokemon=encoded[:, current_start + 1 : current_start + 1 + len(POKEMON_TOKENS)],
            # Store the pre-memory summary so each history entry remains a
            # local snapshot rather than recursively containing prior memory.
            local_history_token=local_summary,
        )

    def local_summary(self, current_tokens: Tensor) -> Tensor:
        """
        Summarize current tokens before any memory interaction.

        This method intentionally cannot see series/history tokens or future
        outcomes. Its output is both the initial reducer readout and the
        snapshot that runtime stores for the next decision; training may keep
        the graph when it reuses local summaries for differentiable history.
        """
        if current_tokens.dim() != 3 or current_tokens.shape[1:] != (
            CURRENT_TOKEN_COUNT,
            self.d_model,
        ):
            raise ValueError(
                f"current tokens do not match the fixed {CURRENT_TOKEN_COUNT}-token contract"
            )
        return self.local_summary_pool(current_tokens)
