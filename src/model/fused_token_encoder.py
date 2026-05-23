from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TokenType,
)


def _load_vocab_sizes() -> dict[str, int]:
    path = Path(__file__).resolve().parents[2] / "data" / "vocab.json"
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {name: len(values) + 1 for name, values in vocab.items()}


def as_obs_dict(obs: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obs, Mapping):
        return obs
    as_dict = getattr(obs, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError("Structured observation must be a mapping or expose as_dict().")


class FusedTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.d_model = d_model

        sizes = _load_vocab_sizes()
        self.roster_emb = nn.Embedding(sizes.get("roster", 1), d_model)
        self.species_emb = nn.Embedding(sizes.get("species", 1), d_model)
        self.ability_emb = nn.Embedding(sizes.get("abilities", 1), d_model)
        self.item_emb = nn.Embedding(sizes.get("items", 1), d_model)
        self.move_emb = nn.Embedding(sizes.get("moves", 1), d_model)
        self.type_emb = nn.Embedding(sizes.get("types", 1), d_model)
        self.status_emb = nn.Embedding(sizes.get("status", 1), d_model)
        self.volatile_emb = nn.Embedding(sizes.get("volatiles", 1), d_model)

        self.global_condition_emb = nn.Embedding(10, d_model)
        self.side_condition_emb = nn.Embedding(6, d_model)
        self.component_emb = nn.Embedding(8, d_model)
        self.token_type_emb = nn.Embedding(6, d_model)
        self.side_emb = nn.Embedding(3, d_model)
        self.slot_emb = nn.Embedding(7, d_model)

        self.move_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
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

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.mon_fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=1, enable_nested_tensor=False
        )
        self.mon_fusion_token = nn.Parameter(torch.empty(1, 1, d_model))
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
        roster = self.roster_emb(categorical[..., 0])
        species = self.species_emb(categorical[..., 1])
        ability = self.ability_emb(categorical[..., 2])
        item = self.item_emb(categorical[..., 3])

        type_summary = self.type_proj(self.type_emb(categorical[..., 4:6]).sum(dim=-2))

        move_parts = torch.cat(
            [
                self.move_emb(categorical[..., 6:10]),
                self.type_emb(categorical[..., 10:14]),
            ],
            dim=-1,
        )
        moveset = self.moveset_proj(self.move_proj(move_parts).mean(dim=-2))

        status = self.status_emb(categorical[..., 14])
        volatile = self.volatile_emb(categorical[..., 15:21]).mean(dim=-2)

        components = torch.stack(
            [species, roster, ability, item, type_summary, moveset, status, volatile],
            dim=-2,
        )
        components = components + self.component_emb(torch.arange(8, device=categorical.device))

        B, S, _, D = components.shape
        fusion_token = self.mon_fusion_token.expand(B, S, -1, -1)
        fused = self.mon_fusion(torch.cat([fusion_token, components], dim=-2).flatten(0, 1))
        return fused[:, 0].view(B, S, D)

    def _embed_fields(
        self, categorical: torch.Tensor, numerical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_field = self.global_condition_emb(categorical[..., :6].clamp_max(9)).mean(dim=-2)
        global_field = global_field + self.field_numeric_proj(numerical)

        side_field = self.side_condition_emb(categorical[..., :6].clamp_max(5)).mean(dim=-2)
        side_field = side_field + self.field_numeric_proj(numerical)
        return global_field, side_field

    def forward(self, obs: Any) -> torch.Tensor:
        obs_dict = as_obs_dict(obs)
        categorical = obs_dict["categorical"].long()
        numerical = obs_dict["numerical"].float()
        token_type_ids = obs_dict["token_type_ids"].long()
        side_ids = obs_dict["side_ids"].long()
        slot_ids = obs_dict["slot_ids"].long()

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

        pokemon_super = self._embed_pokemon_super(categorical)
        pokemon_numeric = self.numeric_proj(numerical)
        global_field, side_field = self._embed_fields(categorical, numerical)

        x = torch.zeros(B, S, self.d_model, device=device, dtype=pokemon_super.dtype)
        x = torch.where(
            (token_type_ids == TokenType.POKEMON_SUPER).unsqueeze(-1),
            pokemon_super,
            x,
        )
        x = torch.where(
            (token_type_ids == TokenType.POKEMON_NUMERIC).unsqueeze(-1),
            pokemon_numeric,
            x,
        )
        x = torch.where(
            (token_type_ids == TokenType.GLOBAL_FIELD).unsqueeze(-1),
            global_field,
            x,
        )
        x = torch.where(
            (
                (token_type_ids == TokenType.ALLY_SIDE)
                | (token_type_ids == TokenType.OPPONENT_SIDE)
            ).unsqueeze(-1),
            side_field,
            x,
        )

        return (
            x
            + self.token_type_emb(token_type_ids)
            + self.side_emb(side_ids)
            + self.slot_emb(slot_ids)
        )
