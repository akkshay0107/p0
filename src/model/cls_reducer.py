from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.structured_observation import SEQUENCE_LENGTH
from src.model.swiglu_encoder import SwiGLUTransformerEncoder


class CLSReducer(nn.Module):
    """Battle-level CLS reducer over already encoded token embeddings."""

    def __init__(
        self,
        seq_len: int = SEQUENCE_LENGTH,
        d_model: int = 512,
        nhead: int = 8,
        nlayer: int = 3,
        dim_feedforward: int | None = None,
        n_hg: int = 4,
        use_history: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_hg = n_hg if use_history else 0
        self.use_history = use_history
        dim_feedforward = dim_feedforward or 4 * d_model

        self.cls_base = nn.Parameter(torch.empty(1, 1, d_model))
        self.register_buffer("hg_init", torch.zeros(1, self.n_hg, d_model))

        self.encoder = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=nlayer,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        emb_gain = self.d_model**-0.5
        init.normal_(self.cls_base, std=emb_gain)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        B, S, D = tokens.shape
        if S != self.seq_len or D != self.d_model:
            raise ValueError(
                f"Got token shape ({S}, {D}). Expected ({self.seq_len}, {self.d_model})."
            )

        cls_tok = self.cls_base.expand(B, -1, -1)

        if state is None:
            hg_prev = self.hg_init.expand(B, -1, -1)
        else:
            _, hg_prev = state
            hg_prev = hg_prev.to(tokens.device)

        # hg_prev empty if use history false
        seq = torch.cat([cls_tok, hg_prev, tokens[:, 1:]], dim=1)
        enc = self.encoder(seq)

        cls = enc[:, 0]
        # empty if use history is false (n_hg = 0)
        hg = enc[:, 1 : 1 + self.n_hg]

        return cls, (cls, hg)
