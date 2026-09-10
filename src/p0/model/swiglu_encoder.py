"""Small, conventional pre-RMSNorm Transformer encoder."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_INIT_STD = 0.02


@torch.no_grad()
def initialize_module(module: nn.Module, *, std: float = MODEL_INIT_STD) -> None:
    """Apply the same small-normal initialization to learned projections and embeddings."""
    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            nn.init.normal_(child.in_proj_weight, std=std)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)
        elif isinstance(child, nn.Linear):
            nn.init.normal_(child.weight, std=std)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, std=std)


class AttentionPool(nn.Module):
    """Pool a fixed token set through one learned cross-attention query."""

    def __init__(self, d_model: int, nhead: int, *, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.query_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.source_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            bias=True,
        )
        self.output_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize the query and task-specific attention projections."""
        initialize_module(self)
        nn.init.normal_(self.query, std=MODEL_INIT_STD)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.dim() != 3 or source.size(1) == 0 or source.size(2) != self.d_model:
            raise ValueError(
                f"source must have shape (batch, tokens > 0, {self.d_model}); got {source.shape}"
            )
        query = self.query.expand(source.size(0), -1, -1)
        normalized_source = self.source_norm(source)
        pooled, _ = self.attention(
            query=self.query_norm(query),
            key=normalized_source,
            value=normalized_source,
            need_weights=False,
        )
        return self.output_norm(query + pooled)[:, 0]


class SwiGLUEncoderLayer(nn.Module):
    """
    Pre-normalized self-attention and SwiGLU feedforward layer.

    Inputs use batch-first layout; attention has no dropout.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        *,
        bias: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.RMSNorm(d_model, eps=norm_eps)
        self.norm2 = nn.RMSNorm(d_model, eps=norm_eps)

        # fused qkv proj
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Match the parameter count of a conventional two-matrix FFN.
        swiglu_hidden = (2 * dim_feedforward) // 3
        swiglu_hidden = (swiglu_hidden + 7) & ~7  # round up to nearest 8

        self.swiglu_hidden = swiglu_hidden

        self.w13 = nn.Linear(d_model, 2 * swiglu_hidden, bias=bias)
        self.w2 = nn.Linear(swiglu_hidden, d_model, bias=bias)

    def _self_attention(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        output_slice: slice = slice(None),
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, sequence, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q[:, :, output_slice]

        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = ~src_key_padding_mask.view(batch, 1, 1, sequence)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(batch, q.size(2), self.d_model)
        return self.out_proj(x)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * val)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        *,
        output_slice: slice = slice(None),
    ) -> torch.Tensor:
        x = src[:, output_slice] + self._self_attention(
            self.norm1(src), src_key_padding_mask, output_slice
        )
        x = x + self._ffn(self.norm2(x))
        return x


class SwiGLUTransformerEncoder(nn.Module):
    """
    Stack encoder layers and apply a final RMS normalization.

    Inputs use batch-first layout with an optional key padding mask.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        *,
        bias: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SwiGLUEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    bias=bias,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize projections normally and scale residual outputs with depth."""
        initialize_module(self)
        residual_std = MODEL_INIT_STD / math.sqrt(2 * len(self.layers))
        for item in self.layers:
            layer = cast(SwiGLUEncoderLayer, item)
            nn.init.normal_(layer.out_proj.weight, std=residual_std)
            nn.init.normal_(layer.w2.weight, std=residual_std)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        *,
        output_slice: slice = slice(None),
    ) -> torch.Tensor:
        """Select final output rows; every input row still supplies keys and values."""
        x = src
        for index, layer in enumerate(self.layers):
            selected = output_slice if index == len(self.layers) - 1 else slice(None)
            x = layer(x, src_key_padding_mask=src_key_padding_mask, output_slice=selected)
        return self.norm(x)
