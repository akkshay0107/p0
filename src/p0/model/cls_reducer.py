from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from p0.model.structured_observation import POKEMON_TOKENS
from p0.model.swiglu_encoder import SwiGLUEncoderLayer

_N_POKEMON_TOKENS = len(POKEMON_TOKENS)


class CLSReducer(nn.Module):
    """Battle-level CLS reducer over already encoded token embeddings."""

    hg_init: torch.Tensor

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        prelude_layers: int,
        dim_feedforward: int,
        n_hg: int,
        core_repeats: int = 1,
        coda_layers: int = 1,
        core_weights_tied: bool = False,
        use_history: bool = True,
    ):
        super().__init__()
        if prelude_layers != 1 or coda_layers != 1:
            raise ValueError("CLSReducer requires one prelude layer and one coda layer")
        if core_repeats <= 0:
            raise ValueError("CLSReducer.core_repeats must be positive")
        if core_weights_tied and core_repeats == 1:
            raise ValueError("CLSReducer.core_weights_tied requires repeated core layers")
        self.seq_len = seq_len
        self.d_model = d_model
        self.prelude_layers = prelude_layers
        self.core_repeats = core_repeats
        self.coda_layers = coda_layers
        self.core_weights_tied = core_weights_tied
        self.n_hg = n_hg if use_history else 0
        self.use_history = use_history
        self.cls_base = nn.Parameter(torch.empty(1, 1, d_model))
        # learned initial state with random per-slot init
        self.hg_init = nn.Parameter(torch.empty(1, self.n_hg, d_model))
        if self.use_history:
            # per-channel gate so each history dimension can keep or refresh independently
            self.hg_gate = nn.Parameter(torch.zeros(1, self.n_hg, d_model))

        self.prelude = SwiGLUEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        if core_weights_tied:
            core = SwiGLUEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            core_layers = [core] * core_repeats
        else:
            core_layers = [
                SwiGLUEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(core_repeats)
            ]
        self.core_layers = nn.ModuleList(core_layers)
        self.coda = SwiGLUEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        emb_gain = self.d_model**-0.5
        init.normal_(self.cls_base, std=emb_gain)
        init.normal_(self.hg_init, std=emb_gain)
        if self.use_history:
            # bias towards preserving history
            init.normal_(self.hg_gate, mean=1.0, std=0.3)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns cls, history, all other tokens"""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 3:
            raise ValueError("tokens must have shape (batch, sequence, d_model)")

        B, S, D = tokens.shape
        if S != self.seq_len or D != self.d_model:
            raise ValueError(
                f"Got token shape ({S}, {D}). Expected ({self.seq_len}, {self.d_model})."
            )
        if state.shape != (B, self.n_hg, self.d_model):
            raise ValueError(
                f"Got recurrent state shape {tuple(state.shape)}. "
                f"Expected ({B}, {self.n_hg}, {self.d_model})."
            )
        if padding_mask is not None and padding_mask.shape != (B, S):
            raise ValueError(f"Expected padding_mask shape ({B}, {S})")

        cls_tok = self.cls_base.to(device=tokens.device, dtype=tokens.dtype).expand(B, -1, -1)

        hg_prev = state.to(device=tokens.device, dtype=tokens.dtype)

        # hg_prev empty if use history false
        seq = torch.cat([cls_tok, hg_prev, tokens[:, 1:]], dim=1)

        enc_mask = None
        if padding_mask is not None:
            enc_mask = torch.zeros(B, seq.size(1), dtype=torch.bool, device=seq.device)
            # CLS and HG tokens are never padded. tokens[:, 1:] corresponds to padding_mask[:, 1:]
            enc_mask[:, 1 + self.n_hg :] = padding_mask.to(
                device=seq.device, dtype=torch.bool
            )[:, 1:]

        enc = self.prelude(seq, src_key_padding_mask=enc_mask)
        for layer in self.core_layers:
            enc = layer(enc, src_key_padding_mask=enc_mask)
        enc = self.norm(self.coda(enc, src_key_padding_mask=enc_mask))

        cls = enc[:, 0]
        # empty if use history is false (n_hg = 0)
        hg_candidate = enc[:, 1 : 1 + self.n_hg]

        if self.use_history:
            gate = torch.sigmoid(self.hg_gate)
            hg = gate * hg_prev.to(hg_candidate.device) + (1 - gate) * hg_candidate
        else:
            hg = hg_candidate

        # extract only the pokemon tokens (skipping CLS at idx 0, and owner tokens after them)
        pokemon_tokens = enc[:, 1 + self.n_hg : 1 + self.n_hg + _N_POKEMON_TOKENS]
        return cls, hg, pokemon_tokens
