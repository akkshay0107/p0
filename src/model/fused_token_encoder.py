from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
    TokenType,
)
from src.model.swiglu_encoder import SwiGLUTransformerEncoder

NUM_COMPONENTS = 7
NUM_TOKEN_TYPES = 6
NUM_SIDES = 3
NUM_SLOTS = 7


def _load_vocab_sizes() -> dict[str, int]:
    path = Path(__file__).resolve().parents[2] / "data" / "vocab.json"
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {name: len(values) + 1 for name, values in vocab.items()}


class FusedTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.d_model = d_model
        d_raw = 64  # lower dim for reduced memory

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

        self.species_proj = nn.Linear(d_raw, d_model)
        self.ability_proj = nn.Linear(d_raw, d_model)
        self.item_proj = nn.Linear(d_raw, d_model)
        self.status_proj = nn.Linear(d_raw, d_model)
        self.volatile_proj = nn.Linear(d_raw, d_model)
        self.weather_proj = nn.Linear(d_raw, d_model)
        self.trickroom_proj = nn.Linear(d_raw, d_model)
        self.side_condition_proj = nn.Linear(d_raw, d_model)
        self.component_emb = nn.Embedding(NUM_COMPONENTS, d_model)
        self.token_type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.side_emb = nn.Embedding(NUM_SIDES, d_model)
        self.slot_emb = nn.Embedding(NUM_SLOTS, d_model)

        # each move gets a move embedding, a type embedding, and a category embedding
        self.move_proj = nn.Sequential(
            nn.Linear(3 * d_raw, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_proj = nn.Sequential(nn.Linear(d_raw, d_model), nn.GELU())
        self.moveset_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(NUMERICAL_WIDTH),
            nn.Linear(NUMERICAL_WIDTH, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.field_numeric_proj = nn.Sequential(
            nn.LayerNorm(NUMERICAL_WIDTH),
            nn.Linear(NUMERICAL_WIDTH, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.mon_fusion = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=1,
        )
        self.mon_fusion_token = nn.Parameter(torch.empty(1, 1, d_model))

        # cache component ids instead of creating them every forward pass
        self.register_buffer("_component_ids", torch.arange(NUM_COMPONENTS))
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

    def _embed_pokemon_super(self, categorical: torch.Tensor) -> torch.Tensor:
        # see _pokemon_categorical in observation_builder
        # for the order of the categorical features
        species = self.species_proj(self.species_emb(categorical[..., 0]))
        ability = self.ability_proj(self.ability_emb(categorical[..., 1]))
        item = self.item_proj(self.item_emb(categorical[..., 2]))

        # proj before sum with the hope that sum in the higher dim space
        # is able to represent the true combination of types
        # used for all permutation invariant combinations below.
        type_summary = self.type_proj(self.type_emb(categorical[..., 3:5])).sum(dim=-2)

        move_parts = torch.cat(
            [
                self.move_emb(categorical[..., 5:9]),
                self.type_emb(categorical[..., 9:13]),
                self.category_emb(categorical[..., 13:17]),
            ],
            dim=-1,
        )
        moveset = self.moveset_proj(self.move_proj(move_parts)).mean(dim=-2)

        status = self.status_proj(self.status_emb(categorical[..., 17]))

        # masked mean over volatile slots; zero-vector when no volatiles present
        v_cat = categorical[..., 18:24]
        v_mask = v_cat != 0
        v_emb = self.volatile_proj(self.volatile_emb(v_cat))
        v_sum = (v_emb * v_mask.unsqueeze(-1).float()).sum(dim=-2)
        v_count = v_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        volatile = v_sum / v_count

        # combine all components into (N, NUM_COMPONENTS, d_model)
        components = torch.stack(
            [species, ability, item, type_summary, moveset, status, volatile],
            dim=-2,
        )
        components = components + self.component_emb(self._component_ids)

        # input from boolean masking is always (N, C), so components is (N, NUM_COMPONENTS, d_model)
        # prepend the fusion token (cls) and run through the mon_fusion transformer
        N = components.shape[0]
        fusion_token = self.mon_fusion_token.expand(N, 1, -1)
        fused = self.mon_fusion(torch.cat([fusion_token, components], dim=1))

        # extract cls "super token"
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
        s_emb = self.side_condition_proj(self.side_condition_emb(s_cat))
        s_sum = (s_emb * s_mask.unsqueeze(-1).float()).sum(dim=-2)
        s_count = s_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
        return s_sum / s_count  # zero-vector when no conditions present (#2)

    def forward(self, obs: StructuredObservation) -> torch.Tensor:
        categorical = obs.categorical.long()
        numerical = obs.numerical.float()
        token_type_ids = obs.token_type_ids.long()
        side_ids = obs.side_ids.long()
        slot_ids = obs.slot_ids.long()

        if categorical.dim() == 2:
            categorical = categorical.unsqueeze(0)
            numerical = numerical.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            side_ids = side_ids.unsqueeze(0)
            slot_ids = slot_ids.unsqueeze(0)

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

        x = torch.zeros(B, S, self.d_model, device=device, dtype=self.mon_fusion_token.dtype)

        # sparse execution: only compute embeddings for the slots that actually need them
        super_mask = token_type_ids == TokenType.POKEMON_SUPER
        if super_mask.any():
            x[super_mask] = self._embed_pokemon_super(categorical[super_mask])

        numeric_mask = token_type_ids == TokenType.POKEMON_NUMERIC
        if numeric_mask.any():
            x[numeric_mask] = self.numeric_proj(numerical[numeric_mask])

        global_mask = token_type_ids == TokenType.GLOBAL_FIELD
        side_mask = (token_type_ids == TokenType.ALLY_SIDE) | (
            token_type_ids == TokenType.OPPONENT_SIDE
        )
        field_mask = global_mask | side_mask
        if field_mask.any():
            # one batched layer norm and linear pass for all 3 field tokens
            field_num = self.field_numeric_proj(numerical[field_mask])

            # categorical embeddings still differ by token type — fill per type
            field_cat_emb = torch.zeros_like(field_num)
            global_in_field = global_mask[field_mask]
            if global_in_field.any():
                field_cat_emb[global_in_field] = self._embed_global_field_cond(
                    categorical[global_mask]
                )
            side_in_field = side_mask[field_mask]
            if side_in_field.any():
                field_cat_emb[side_in_field] = self._embed_side_field_cond(categorical[side_mask])
            x[field_mask] = field_cat_emb + field_num

        return (
            x
            + self.token_type_emb(token_type_ids)
            + self.side_emb(side_ids)
            + self.slot_emb(slot_ids)
        )
