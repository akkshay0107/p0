"""Tensor-only prior-game series resampling for Best-of-three context."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.swiglu_encoder import MODEL_INIT_STD, SwiGLUTransformerEncoder, initialize_module


class DynamicSeriesResampler(nn.Module):
    """Compress masked game histories into four trainable series tokens."""

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
        self.position_proj = nn.Linear(1, d_model)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)
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
        initialize_module(self)
        nn.init.normal_(self.summary_queries, std=MODEL_INIT_STD)
        self.self_attn.reset_parameters()

    def forward(self, history: Tensor, history_mask: Tensor) -> Tensor:
        """Compress (games, time, d_model) histories into (games, 4, d_model)."""
        if history.dim() != 3 or history.size(-1) != self.d_model:
            raise ValueError(
                f"Expected history shape (games, time, {self.d_model}); got {tuple(history.shape)}"
            )
        games, time, _ = history.shape
        if time <= 0:
            raise ValueError("history must contain at least one padded time position")
        if (
            history_mask.shape != (games, time)
            or history_mask.dtype is not torch.bool
            or history_mask.device != history.device
        ):
            raise ValueError(
                "history_mask must be a boolean (games, time) tensor on history.device"
            )

        lengths = history_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Each resampled game must contain at least one history token")

        position_ids = torch.arange(time, device=history.device)
        denominators = (lengths - 1).clamp_min(1).unsqueeze(1)
        positions = (position_ids.unsqueeze(0) / denominators).unsqueeze(-1)
        positions = positions * history_mask.unsqueeze(-1)
        keys_values = self.norm_k(history + self.position_proj(positions))
        queries = self.norm_q(self.summary_queries.expand(games, -1, -1))

        compressed, _ = self.cross_attn(
            query=queries,
            key=keys_values,
            value=keys_values,
            key_padding_mask=~history_mask,
            need_weights=False,
        )
        return self.self_attn(queries + compressed)
