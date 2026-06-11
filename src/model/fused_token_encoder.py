from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

from src.lookups import ACT_SIZE
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE_SUPER,
    TOKEN_IDX_GLOBAL_FIELD_SUPER,
    TOKEN_IDX_OPPONENT_SIDE_SUPER,
    StructuredObservation,
)
from src.model.swiglu_encoder import SwiGLUTransformerEncoder

NUM_COMPONENTS = 11
NUM_TOKEN_TYPES = 5
NUM_SIDES = 3
NUM_SLOTS = 7

# 0 CLS
# 1,3,...,23 pokemon super tokens
# 2,4,...,24 pokemon numeric tokens
# 25 Global-field super
# 26 Global-field numeric
# 27 Ally-side super
# 28 Ally-side numeric
# 29 Opponent-side super
# 30 Opponent-side numeric
_SUPER_POS = tuple(range(1, 24, 2))  # (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
_NUMERIC_POS = tuple(range(2, 25, 2))  # (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
_FIELD_SUPER_POS = (25, 27, 29)
_FIELD_NUMERIC_POS = (26, 28, 30)


def _load_vocab_sizes() -> dict[str, int]:
    path = Path(__file__).resolve().parents[2] / "data" / "vocab.json"
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {name: len(values) + 1 for name, values in vocab.items()}


MOVE_STATIC_WIDTH = 3  # base power, max pp, priority


def _load_move_statics() -> torch.Tensor:
    """Static per-move scalars indexed by vocab move id (row 0 = padding)."""
    from poke_env.battle.move import Move

    path = Path(__file__).resolve().parents[2] / "data" / "vocab.json"
    with path.open("r", encoding="utf-8") as f:
        moves_vocab = json.load(f)["moves"]

    table = torch.zeros(len(moves_vocab) + 1, MOVE_STATIC_WIDTH)
    for name, idx in moves_vocab.items():
        try:
            move = Move(name, gen=9)
            table[idx, 0] = move.base_power / 150.0
            table[idx, 1] = move.max_pp / 64.0
            table[idx, 2] = move.priority / 5.0
        except Exception:
            # pseudo-moves without full data stay zero
            pass
    return table


class MultiAggDeepSet(nn.Module):
    def __init__(self, in_features: int, d_model: int):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(in_features, d_model), nn.GELU())
        self.f = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        gx = self.g(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            sum_gx = torch.where(mask_expanded, gx, 0.0).sum(dim=-2)
            max_gx = gx.masked_fill(~mask_expanded, float("-inf")).max(dim=-2)[0]
            max_gx = torch.where(max_gx == float("-inf"), 0.0, max_gx)
        else:
            sum_gx = gx.sum(dim=-2)
            max_gx = gx.max(dim=-2)[0]

        return self.f(torch.cat([sum_gx, max_gx], dim=-1))


# Also considering removing the unnecessary interleaving and just stacking
# all embeddings together and all numerical rows together. Would make
# downstream slicing much easier. More effort to rewrite / test the code tho
class FusedTokenEncoder(nn.Module):
    # type annotations for pyright
    _super_pos: torch.Tensor
    _other_super_pos: torch.Tensor
    _numeric_pos: torch.Tensor
    _field_super_pos: torch.Tensor
    _field_numeric_pos: torch.Tensor
    _component_ids: torch.Tensor
    _move_statics: torch.Tensor

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.d_model = d_model
        d_raw = 128  # lower dim for reduced memory

        sizes = _load_vocab_sizes()
        self.species_emb = nn.Embedding(sizes.get("species", 1), d_raw)
        self.ability_emb = nn.Embedding(sizes.get("abilities", 1), d_raw)
        self.item_emb = nn.Embedding(sizes.get("items", 1), d_raw)
        self.move_emb = nn.Embedding(sizes.get("moves", 1), d_raw)
        self.type_emb = nn.Embedding(sizes.get("types", 1), d_raw)
        self.category_emb = nn.Embedding(sizes.get("categories", 1), d_raw)
        self.status_emb = nn.Embedding(sizes.get("status", 1), d_raw)
        self.volatile_emb = nn.Embedding(sizes.get("volatiles", 1), d_raw)

        self.weather_emb = nn.Embedding(sizes.get("weathers", 1), d_raw)
        self.trickroom_emb = nn.Embedding(sizes.get("trickroom", 1), d_raw)
        self.side_condition_emb = nn.Embedding(sizes.get("side_conditions", 1), d_raw)
        self.nature_emb = nn.Embedding(25, d_raw)

        self.species_proj = nn.Linear(d_raw, d_model)
        self.ability_proj = nn.Linear(d_raw, d_model)
        self.item_proj = nn.Linear(d_raw, d_model)
        self.status_proj = nn.Linear(d_raw, d_model)
        self.weather_proj = nn.Linear(d_raw, d_model)
        self.trickroom_proj = nn.Linear(d_raw, d_model)
        self.nature_proj = nn.Linear(d_raw, d_model)

        self.volatile_set = MultiAggDeepSet(d_raw, d_model)
        self.side_condition_set = MultiAggDeepSet(d_raw, d_model)

        self.component_emb = nn.Embedding(NUM_COMPONENTS, d_model)
        self.move_pos_emb = nn.Embedding(4, d_model)
        self.token_type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.side_emb = nn.Embedding(NUM_SIDES, d_model)
        self.slot_emb = nn.Embedding(NUM_SLOTS, d_model)

        # each move gets a move embedding, a type embedding, a category embedding,
        # and a static scalar vector (base power, max pp, priority)
        self.move_proj = nn.Sequential(nn.Linear(3 * d_raw + MOVE_STATIC_WIDTH, d_model), nn.GELU())
        self.type_set = MultiAggDeepSet(d_raw, d_model)
        self.numeric_proj = nn.Sequential(
            nn.Linear(NUMERICAL_WIDTH, d_model),
            nn.GELU(),
        )
        self.field_numeric_proj = nn.Sequential(
            nn.Linear(NUMERICAL_WIDTH, d_model),
            nn.GELU(),
        )
        self.action_mask_proj = nn.Sequential(
            nn.Linear(2 * ACT_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_mask_token = nn.Parameter(torch.empty(1, 1, d_model))

        self.mon_fusion = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=1,
        )
        self.mon_fusion_token = nn.Parameter(torch.empty(1, 1, d_model))

        # cache component ids instead of creating them every forward pass
        self.register_buffer("_component_ids", torch.arange(NUM_COMPONENTS))
        self.register_buffer("_move_statics", _load_move_statics())
        # cache fixed sequence-position indices so advanced indexing uses pre-allocated
        # device tensors rather than constructing a new index tensor on every forward pass.
        self.register_buffer("_super_pos", torch.tensor(_SUPER_POS, dtype=torch.long))
        self.register_buffer("_numeric_pos", torch.tensor(_NUMERIC_POS, dtype=torch.long))
        self.register_buffer("_field_super_pos", torch.tensor(_FIELD_SUPER_POS, dtype=torch.long))
        self.register_buffer(
            "_field_numeric_pos", torch.tensor(_FIELD_NUMERIC_POS, dtype=torch.long)
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        emb_gain = self.d_model**-0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, std=emb_gain)
        init.normal_(self.mon_fusion_token, std=emb_gain)
        init.normal_(self.action_mask_token, std=emb_gain)

    def _append_action_mask_token(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = tokens.size(0)
        if action_mask.shape != (B, 2, ACT_SIZE):
            raise ValueError(
                f"Expected action mask ({B}, 2, {ACT_SIZE}); got {tuple(action_mask.shape)}."
            )
        flat_mask = action_mask.reshape(B, -1).to(tokens.dtype)

        mask_token = self.action_mask_token.expand(B, -1, -1)
        mask_token = mask_token + self.action_mask_proj(flat_mask).unsqueeze(1)
        return torch.cat([tokens, mask_token], dim=1)

    def _embed_pokemon_super(
        self, categorical: torch.Tensor, aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # see _pokemon_categorical in observation_builder
        # for the order of the categorical features
        species = self.species_proj(self.species_emb(categorical[..., 0]))
        ability = self.ability_proj(self.ability_emb(categorical[..., 1]))
        item = self.item_proj(self.item_emb(categorical[..., 2]))

        type_summary = self.type_set(self.type_emb(categorical[..., 3:5]))

        move_ids = categorical[..., 5:9]
        move_parts = torch.cat(
            [
                self.move_emb(move_ids),
                self.type_emb(categorical[..., 9:13]),
                self.category_emb(categorical[..., 13:17]),
                self._move_statics[move_ids],
            ],
            dim=-1,
        )

        # pos emb added here to break permutation invariance
        # the model downstream needs to know which slot is which
        # for a1 to choose the indices
        move_embs = self.move_proj(move_parts) + self.move_pos_emb.weight

        status = self.status_proj(self.status_emb(categorical[..., 17]))

        # masked multi-agg over volatile slots; zero-vector when no volatiles present
        v_cat = categorical[..., 18:24]
        v_mask = v_cat != 0
        volatile = self.volatile_set(self.volatile_emb(v_cat), mask=v_mask)

        nature = self.nature_proj(self.nature_emb(categorical[..., 24]))

        # combine all components into (N, NUM_COMPONENTS, d_model)
        # Note: move_embs is (..., 4, d_model), others are (..., d_model)
        components = torch.cat(
            [
                species.unsqueeze(-2),
                ability.unsqueeze(-2),
                item.unsqueeze(-2),
                type_summary.unsqueeze(-2),
                move_embs,
                status.unsqueeze(-2),
                volatile.unsqueeze(-2),
                nature.unsqueeze(-2),
            ],
            dim=-2,
        )
        components = components + self.component_emb(self._component_ids)

        # input from boolean masking is always (N, C), so components is (N, NUM_COMPONENTS, d_model)
        # prepend the fusion token (cls) and run through the mon_fusion transformer
        N = components.shape[0]
        fusion_token = self.mon_fusion_token.expand(N, 1, -1)
        fused = self.mon_fusion(torch.cat([fusion_token, components], dim=1))

        # fused[:, 0] is the cls => "super token" for a pokemon
        # aux channel returns move_embs which is needed downstream for
        # embedding the action of the first pokemon
        if aux:
            return fused[:, 0], move_embs
        return fused[:, 0]

    def _embed_global_field_cond(self, categorical: torch.Tensor) -> torch.Tensor:
        """Returns the categorical-side embedding only (numeric added in forward)."""
        weather_emb = self.weather_proj(self.weather_emb(categorical[..., 0]))
        trickroom_emb = self.trickroom_proj(self.trickroom_emb(categorical[..., 1]))
        return weather_emb + trickroom_emb

    def _embed_side_field_cond(self, categorical: torch.Tensor) -> torch.Tensor:
        """Returns the categorical-side embedding only (numeric added in forward)."""
        s_cat = categorical[..., :4]
        s_mask = s_cat != 0
        return self.side_condition_set(self.side_condition_emb(s_cat), mask=s_mask)

    def forward(
        self,
        obs: StructuredObservation,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        categorical = obs.categorical.long()
        numerical = obs.numerical.float()
        token_type_ids = obs.token_type_ids.long()
        side_ids = obs.side_ids.long()
        slot_ids = obs.slot_ids.long()

        if categorical.dim() != 3:
            raise ValueError(
                f"Expected a batched categorical tensor with 3 dimensions; "
                f"got {tuple(categorical.shape)}."
            )

        B, S, C = categorical.shape
        if S != SEQUENCE_LENGTH or C != CATEGORICAL_WIDTH or numerical.shape[-1] != NUMERICAL_WIDTH:
            raise ValueError(
                f"Expected categorical ({SEQUENCE_LENGTH}, {CATEGORICAL_WIDTH}) and "
                f"numerical ({SEQUENCE_LENGTH}, {NUMERICAL_WIDTH}); got {tuple(categorical.shape)} "
                f"and {tuple(numerical.shape)}."
            )

        device = self.mon_fusion_token.device
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        token_type_ids = token_type_ids.to(device)
        side_ids = side_ids.to(device)
        slot_ids = slot_ids.to(device)
        action_mask = action_mask.to(device)

        x = torch.zeros(B, S, self.d_model, device=device, dtype=self.mon_fusion_token.dtype)

        n_super = len(_SUPER_POS)
        super_cats = categorical[:, self._super_pos, :].flatten(0, 1)
        super_out, all_move_embs = self._embed_pokemon_super(super_cats, aux=True)

        x[:, self._super_pos, :] = super_out.unflatten(0, (B, n_super)).to(x.dtype)
        aux_moves = all_move_embs.unflatten(0, (B, n_super))[:, :2]

        x[:, self._numeric_pos, :] = self.numeric_proj(numerical[:, self._numeric_pos, :]).to(x.dtype)

        x[:, TOKEN_IDX_GLOBAL_FIELD_SUPER, :] = self._embed_global_field_cond(
            categorical[:, TOKEN_IDX_GLOBAL_FIELD_SUPER, :]
        ).to(x.dtype)

        x[:, (TOKEN_IDX_ALLY_SIDE_SUPER, TOKEN_IDX_OPPONENT_SIDE_SUPER), :] = (
            self._embed_side_field_cond(
                categorical[:, (TOKEN_IDX_ALLY_SIDE_SUPER, TOKEN_IDX_OPPONENT_SIDE_SUPER), :]
            ).to(x.dtype)
        )

        x[:, self._field_numeric_pos, :] = self.field_numeric_proj(
            numerical[:, self._field_numeric_pos, :]
        ).to(x.dtype)

        out_tokens = (
            x
            + self.token_type_emb(token_type_ids)
            + self.side_emb(side_ids)
            + self.slot_emb(slot_ids)
        )
        out_tokens = self._append_action_mask_token(out_tokens, action_mask)
        return out_tokens, aux_moves
