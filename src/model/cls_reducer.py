import torch
import torch.nn as nn
import torch.nn.init as init

from lookups import OBS_DIM


class CLSReducer(nn.Module):
    # type emb indexes
    CLS = 0
    FIELD = 1
    TEAM_SLOT_0 = 2
    TEAM_SLOT_1 = 3
    TEAM_BENCH = 4
    TEAM_DROPPED = 5
    OPP_SLOT_0 = 6
    OPP_SLOT_1 = 7
    OPP_BENCH = 8
    OPP_DROPPED = 9
    H1 = 10
    H2 = 11
    H3 = 12
    HG = 13
    UNKNOWN = 14
    # part emb indexes
    TEXT_A, TEXT_B, NUM = 0, 1, 2

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        d_model: int,
        nhead: int,
        nlayer: int,
        dim_feedforward: int,
        n_hg: int,
    ):
        super().__init__()
        if seq_len != OBS_DIM[0]:
            raise ValueError(f"This CLSReducer assumes seq_len={OBS_DIM[0]}.")

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_hg = n_hg

        self.in_proj = nn.Linear(feat_dim, d_model)

        self.cls_base = nn.Parameter(torch.rand(1, 1, d_model) * self.d_model**-0.5)
        self.cls_conditioner = nn.Linear(d_model, d_model)

        self.type_emb = nn.Embedding(15, d_model)
        self.part_emb = nn.Embedding(3, d_model)

        # Transformer for history update: HG_curr = Transformer([HG_prev, H1_curr, CLS_prev])
        self.register_buffer("hg_init", torch.zeros(1, n_hg, d_model))
        self.history_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayer,
            enable_nested_tensor=False,
        )

        # 0: CLS
        # 1-4: HG (using default of 4)
        # -- Above injected, below from obs --
        # 5-7: H1, H2, H3 text
        # 8: Field Text
        # 9: Info Text
        # 10-21: P1 Text (12)
        # 22-33: Opp Text (12)
        # 34-39: P1 Num (6)
        # 40-45: Opp Num (6)
        # 46: Field Num (1)

        part_ids = torch.cat(
            [
                torch.tensor([self.TEXT_A, self.TEXT_B], dtype=torch.long).repeat(12),  # P1+Opp
                torch.tensor([self.NUM], dtype=torch.long).repeat(13),  # P1+Opp+Field
            ]
        )
        self.register_buffer("part_ids", part_ids, persistent=False)

        type_ids = torch.zeros(43 + n_hg, dtype=torch.long)
        type_ids[0] = self.CLS
        type_ids[1 : 1 + n_hg] = self.HG

        type_ids[n_hg + 1] = self.H1
        type_ids[n_hg + 2] = self.H2
        type_ids[n_hg + 3] = self.H3

        type_ids[n_hg + 4] = self.FIELD
        type_ids[n_hg + 5] = self.FIELD

        # P1 Texts
        type_ids[n_hg + 6 : n_hg + 8] = self.TEAM_SLOT_0
        type_ids[n_hg + 8 : n_hg + 10] = self.TEAM_SLOT_1
        type_ids[n_hg + 10 : n_hg + 14] = self.TEAM_BENCH
        type_ids[n_hg + 14 : n_hg + 18] = self.TEAM_DROPPED

        # Opp Texts
        type_ids[n_hg + 18 : n_hg + 20] = self.OPP_SLOT_0
        type_ids[n_hg + 20 : n_hg + 22] = self.OPP_SLOT_1
        type_ids[n_hg + 22 : n_hg + 26] = self.OPP_BENCH
        type_ids[n_hg + 26 : n_hg + 30] = self.OPP_DROPPED

        # P1 Num
        type_ids[n_hg + 30] = self.TEAM_SLOT_0
        type_ids[n_hg + 31] = self.TEAM_SLOT_1
        type_ids[n_hg + 32 : n_hg + 34] = self.TEAM_BENCH
        type_ids[n_hg + 34 : n_hg + 36] = self.TEAM_DROPPED

        # Opp Num
        type_ids[n_hg + 36] = self.OPP_SLOT_0
        type_ids[n_hg + 37] = self.OPP_SLOT_1
        type_ids[n_hg + 38 : n_hg + 40] = self.OPP_BENCH
        type_ids[n_hg + 40 : n_hg + 42] = self.OPP_DROPPED

        # Field Num
        type_ids[n_hg + 42] = self.FIELD
        self.register_buffer("type_ids", type_ids, persistent=False)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        init.orthogonal_(self.in_proj.weight, gain=1.0)
        init.zeros_(self.in_proj.bias)

        init.orthogonal_(self.cls_conditioner.weight, gain=1.0)
        init.zeros_(self.cls_conditioner.bias)

        emb_gain = self.d_model**-0.5
        init.normal_(self.cls_base, std=emb_gain)
        init.normal_(self.type_emb.weight, std=emb_gain)
        init.normal_(self.part_emb.weight, std=emb_gain)

    def forward(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        B, S, F = obs.shape
        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected ({self.seq_len}, {self.feat_dim})")

        x = self.in_proj(obs)
        cls_base = self.cls_base.expand(B, -1, -1)
        global_ctx = x.mean(dim=1, keepdim=True)
        cls_tok = cls_base + self.cls_conditioner(global_ctx)

        if state is None:
            cls_prev = cls_base
            hg_prev = self.hg_init.expand(B, -1, -1)
        else:
            cls_prev, hg_prev = state
            if cls_prev.dim() == 2:
                cls_prev = cls_prev.unsqueeze(1)

        h1_new = x[:, :1]
        hg_in = torch.cat([hg_prev, cls_prev, h1_new], dim=1)
        hg = self.history_transformer(hg_in)[:, : self.n_hg]

        seq = torch.cat([cls_tok, hg, x], dim=1)

        if seq.size(1) != self.type_ids.numel():
            raise RuntimeError(
                f"Final seq len {seq.size(1)} does not match type_ids len {self.type_ids.numel()}"
            )

        seq = seq + self.type_emb(self.type_ids).unsqueeze(0)

        part_start = 1 + self.n_hg + 5
        seq[:, part_start:] += self.part_emb(self.part_ids).unsqueeze(0)

        enc = self.encoder(seq)
        cls = enc[:, 0]

        next_state = (cls, hg)
        return cls, next_state
