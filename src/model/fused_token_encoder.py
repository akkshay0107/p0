from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

from src.format_config import FORMAT
from src.model.event_builder import EVENT_TYPE_COUNT
from src.model.structured_observation import (
    ALLY_NUM_TOKENS,
    CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    EVENT_ORDER_VOCAB_SIZE,
    MOVE_SLOTS,
    NUM_IDX_MOVE_LAST,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_MOVE_PP,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE_SUPER,
    TOKEN_IDX_GLOBAL_FIELD_SUPER,
    TOKEN_IDX_OPPONENT_SIDE_SUPER,
    StructuredObservation,
    TokenType,
)
from src.model.swiglu_encoder import SwiGLUTransformerEncoder

ACT_SIZE = FORMAT.action_size
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NUM_COMPONENTS = 11
NUM_TOKEN_TYPES = 6
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
_FIELD_NUMERIC_POS = (26, 28, 30)


def _load_vocab_sizes() -> dict[str, int]:
    path = DATA_DIR / "vocab.json"
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {name: len(values) + 1 for name, values in vocab.items()}


MOVE_DYNAMIC_WIDTH = 3  # pp fraction, last-move flag, legal-this-step
SPECIES_STATIC_WIDTH = 9  # six base stats, weight, mega flag, forme relationship

_TARGET_CLASSES = (
    "self",
    "adjacentally",
    "adjacentallyorself",
    "selectedpokemon",
    "adjacentfoe",
    "all",
    "alladjacent",
    "alladjacentfoes",
    "allies",
    "allyside",
    "allyteam",
    "foeside",
    "randomnormal",
    "scripted",
)
_TARGET_CLASS_INDEX = {name: index for index, name in enumerate(_TARGET_CLASSES)}
_TARGET_CLASS_ALIASES = {
    # Distance is irrelevant with two active slots per side, so these have the
    # same selectable Pokemon in doubles. Other Showdown target types remain
    # distinct, including `all` (includes the user) and `allAdjacent` (does not).
    "normal": "selectedpokemon",
    "any": "selectedpokemon",
}
MOVE_STATIC_WIDTH = 7 + len(_TARGET_CLASSES)


def _load_species_statics() -> torch.Tensor:
    with (DATA_DIR / "vocab.json").open("r", encoding="utf-8") as f:
        species_vocab = json.load(f)["species"]
    with (DATA_DIR / "champions_dex.json").open("r", encoding="utf-8") as f:
        dex_species = {entry["id"]: entry for entry in json.load(f)["species"]}

    table = torch.zeros(len(species_vocab) + 1, SPECIES_STATIC_WIDTH)
    for name, idx in species_vocab.items():
        species = dex_species.get(name)
        if species is None:
            raise ValueError(f"Missing Champions mechanics for vocabulary species: {name}")
        stats = species.get("baseStats", {})
        for offset, stat in enumerate(("hp", "atk", "def", "spa", "spd", "spe")):
            table[idx, offset] = float(stats.get(stat, 0)) / 255.0
        table[idx, 6] = float(species.get("weightkg", 0.0) or 0.0) / 1000.0
        table[idx, 7] = float(bool(species.get("isMega")))
        table[idx, 8] = float(
            bool(species.get("baseSpecies") and species.get("baseSpecies") != species.get("name"))
        )
    return table


def _load_move_statics() -> torch.Tensor:
    """Static per-move scalars indexed by vocab move id (row 0 = padding)."""
    with (DATA_DIR / "vocab.json").open("r", encoding="utf-8") as f:
        moves_vocab = json.load(f)["moves"]
    with (DATA_DIR / "champions_dex.json").open("r", encoding="utf-8") as f:
        dex_moves = {entry["id"]: entry for entry in json.load(f)["moves"]}

    table = torch.zeros(len(moves_vocab) + 1, MOVE_STATIC_WIDTH)
    for name, idx in moves_vocab.items():
        move = dex_moves.get(name)
        if move is None:
            # Explicit runtime pseudo-actions are allowed to remain zero-valued;
            # every ordinary vocabulary move must be present in the dex dump.
            if name not in {"struggle", "recharge"}:
                raise ValueError(f"Missing Champions mechanics for vocabulary move: {name}")
            continue
        table[idx, 0] = float(move.get("basePower", 0)) / 150.0
        table[idx, 1] = float(move.get("pp", 0)) / 64.0
        table[idx, 2] = float(move.get("priority", 0)) / 5.0
        accuracy = move.get("accuracy", 100)
        table[idx, 3] = float(accuracy) / 100.0 if isinstance(accuracy, (int, float)) else 0.0
        target = str(move.get("target", "")).lower()
        target_class = _TARGET_CLASS_ALIASES.get(target, target)
        target_index = _TARGET_CLASS_INDEX.get(target_class)
        if target_index is None:
            raise ValueError(f"Unknown Showdown target class for move {name}: {target!r}")
        table[idx, 4 + target_index] = 1.0

        flag_offset = 4 + len(_TARGET_CLASSES)
        table[idx, flag_offset] = float(bool(move.get("spreadHit")))
        flags = move.get("flags", {})
        table[idx, flag_offset + 1] = float(bool(flags.get("protect")))
        table[idx, flag_offset + 2] = float(bool(move.get("selfSwitch")))
    return table


def _load_mechanic_tag_tables() -> dict[str, torch.Tensor]:
    """Load audited item and ability hook tables in one data-file pass."""
    with (DATA_DIR / "vocab.json").open("r", encoding="utf-8") as stream:
        vocab_data = json.load(stream)
    with (DATA_DIR / "champions_dex.json").open("r", encoding="utf-8") as stream:
        dex_data = json.load(stream)

    tables: dict[str, torch.Tensor] = {}
    for table_name in ("items", "abilities"):
        vocab = vocab_data[table_name]
        entries = {entry["id"]: entry for entry in dex_data[table_name]}
        missing = sorted(set(vocab) - entries.keys())
        if missing:
            raise ValueError(
                f"Missing Champions mechanics for vocabulary {table_name}: {', '.join(missing[:8])}"
            )
        tags = sorted({tag for entry in entries.values() for tag in entry.get("mechanicTags", [])})
        tag_index = {tag: index for index, tag in enumerate(tags)}
        result = torch.zeros((len(vocab) + 1, len(tags)), dtype=torch.float32)
        for name, row in vocab.items():
            mechanic_tags = entries[name].get("mechanicTags")
            if not isinstance(mechanic_tags, list):
                raise ValueError(
                    f"Missing mechanicTags metadata for legal {table_name} entry: {name}"
                )
            for tag in mechanic_tags:
                result[row, tag_index[tag]] = 1.0
        tables[table_name] = result
    return tables


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
            max_gx = gx.masked_fill(~mask_expanded, float("-inf")).amax(dim=-2)
            max_gx = torch.where(max_gx == float("-inf"), 0.0, max_gx)
        else:
            sum_gx = gx.sum(dim=-2)
            max_gx = gx.max(dim=-2)[0]

        return self.f(torch.cat([sum_gx, max_gx], dim=-1))


class FusedTokenEncoder(nn.Module):
    # Buffers registered dynamically by torch need explicit declarations for Pyright.
    _super_pos: torch.Tensor
    _numeric_pos: torch.Tensor
    _field_numeric_pos: torch.Tensor
    _active_num_pos: torch.Tensor
    _component_ids: torch.Tensor
    _species_statics: torch.Tensor
    _move_statics: torch.Tensor
    _item_mechanic_tags: torch.Tensor
    _ability_mechanic_tags: torch.Tensor

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
        self.species_emb = nn.Embedding(sizes["species"], d_raw)
        self.ability_emb = nn.Embedding(sizes["abilities"], d_raw)
        self.item_emb = nn.Embedding(sizes["items"], d_raw)
        self.move_emb = nn.Embedding(sizes["moves"], d_raw)
        self.type_emb = nn.Embedding(sizes["types"], d_raw)
        self.category_emb = nn.Embedding(sizes["categories"], d_raw)
        self.status_emb = nn.Embedding(sizes["status"], d_raw)
        self.volatile_emb = nn.Embedding(sizes["volatiles"], d_raw)

        self.weather_emb = nn.Embedding(sizes["weathers"], d_raw)
        self.trickroom_emb = nn.Embedding(sizes["trickroom"], d_raw)
        self.side_condition_emb = nn.Embedding(sizes["side_conditions"], d_raw)
        self.nature_emb = nn.Embedding(25, d_raw)

        self.species_proj = nn.Linear(d_raw, d_model)
        self.species_static_proj = nn.Linear(SPECIES_STATIC_WIDTH, d_model)
        self.ability_proj = nn.Linear(d_raw, d_model)
        self.item_proj = nn.Linear(d_raw, d_model)
        mechanic_tags = _load_mechanic_tag_tables()
        item_tags = mechanic_tags["items"]
        ability_tags = mechanic_tags["abilities"]
        self.item_mechanic_proj = nn.Linear(item_tags.shape[1], d_model, bias=False)
        self.ability_mechanic_proj = nn.Linear(ability_tags.shape[1], d_model, bias=False)
        self.register_buffer("_item_mechanic_tags", item_tags)
        self.register_buffer("_ability_mechanic_tags", ability_tags)
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

        self.event_type_emb = nn.Embedding(EVENT_TYPE_COUNT, d_model)
        self.order_pos_emb = nn.Embedding(EVENT_ORDER_VOCAB_SIZE, d_model)
        self.event_proj = nn.Linear(3 * d_raw + EVENT_NUMERICAL_WIDTH, d_model)

        # each move gets a move embedding, a type embedding, a category embedding,
        # and a static scalar vector (base power, max pp, priority, accuracy)
        self.move_proj = nn.Linear(3 * d_raw + MOVE_STATIC_WIDTH, d_model)
        # mixes the per-slot dynamic move state into the aux channel so the
        # pointer move keys see more than the static move identity
        self.move_dyn_proj = nn.Linear(MOVE_DYNAMIC_WIDTH, d_model)
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
        self.register_buffer("_species_statics", _load_species_statics())
        self.register_buffer("_move_statics", _load_move_statics())
        # cache fixed sequence-position indices so advanced indexing uses pre-allocated
        # device tensors rather than constructing a new index tensor on every forward pass.
        self.register_buffer("_super_pos", torch.tensor(_SUPER_POS, dtype=torch.long))
        self.register_buffer("_numeric_pos", torch.tensor(_NUMERIC_POS, dtype=torch.long))
        self.register_buffer(
            "_field_numeric_pos", torch.tensor(_FIELD_NUMERIC_POS, dtype=torch.long)
        )
        self.register_buffer("_active_num_pos", torch.tensor(ALLY_NUM_TOKENS[:2], dtype=torch.long))
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        emb_gain = self.d_model**-0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
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

    def _embed_pokemon_components(
        self, categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # see _pokemon_categorical in observation_builder
        # for the order of the categorical features
        species_ids = categorical[..., 0]
        species = self.species_proj(self.species_emb(species_ids))
        species = species + self.species_static_proj(self._species_statics[species_ids])
        ability_ids = categorical[..., 1]
        item_ids = categorical[..., 2]
        ability = self.ability_proj(self.ability_emb(ability_ids))
        ability = ability + self.ability_mechanic_proj(self._ability_mechanic_tags[ability_ids])
        item = self.item_proj(self.item_emb(item_ids))
        item = item + self.item_mechanic_proj(self._item_mechanic_tags[item_ids])

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

        return fused[:, 0], move_embs

    def _embed_pokemon_super(self, categorical: torch.Tensor) -> torch.Tensor:
        """Embed Pokemon rows without exposing pointer-head auxiliary features."""
        super_token, _ = self._embed_pokemon_components(categorical)
        return super_token

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

        batch_size, sequence_length, categorical_width = categorical.shape
        expected_numerical_shape = (batch_size, SEQUENCE_LENGTH, NUMERICAL_WIDTH)
        if (
            sequence_length != SEQUENCE_LENGTH
            or categorical_width != CATEGORICAL_WIDTH
            or numerical.shape != expected_numerical_shape
        ):
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

        x = torch.zeros(
            batch_size,
            sequence_length,
            self.d_model,
            device=device,
            dtype=self.mon_fusion_token.dtype,
        )

        n_super = len(_SUPER_POS)
        super_cats = categorical[:, self._super_pos, :].flatten(0, 1)
        super_out, all_move_embs = self._embed_pokemon_components(super_cats)

        x[:, self._super_pos, :] = super_out.unflatten(0, (batch_size, n_super)).to(x.dtype)
        aux_moves = all_move_embs.unflatten(0, (batch_size, n_super))[:, :2]

        # the move embeddings above are state-blind; without this the pointer
        # head only learns pp / encore / choice-lock nuance indirectly through z
        active_num = numerical[:, self._active_num_pos, :]
        move_dyn = torch.stack(
            [
                active_num[..., NUM_IDX_MOVE_PP : NUM_IDX_MOVE_PP + MOVE_SLOTS],
                active_num[..., NUM_IDX_MOVE_LAST : NUM_IDX_MOVE_LAST + MOVE_SLOTS],
                active_num[..., NUM_IDX_MOVE_LEGAL : NUM_IDX_MOVE_LEGAL + MOVE_SLOTS],
            ],
            dim=-1,
        )
        aux_moves = aux_moves + self.move_dyn_proj(move_dyn)

        x[:, self._numeric_pos, :] = self.numeric_proj(numerical[:, self._numeric_pos, :]).to(
            x.dtype
        )

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

        events_cat = obs.events_cat.long().to(device)
        events_num = obs.events_num.float().to(device)
        events_side_ids = obs.events_side_ids.long().to(device)
        events_slot_ids = obs.events_slot_ids.long().to(device)

        event_feats = torch.cat(
            [
                self.move_emb(events_cat[..., 1]),
                self.item_emb(events_cat[..., 2]),
                self.status_emb(events_cat[..., 3]),
                events_num,
            ],
            dim=-1,
        )

        event_tokens = self.event_proj(event_feats)
        event_tokens = (
            event_tokens
            + self.event_type_emb(events_cat[..., 0])
            + self.side_emb(events_side_ids)
            + self.slot_emb(events_slot_ids)
            + self.order_pos_emb(events_cat[..., 4])
            + self.token_type_emb(
                torch.full(
                    (batch_size, EVENT_COUNT),
                    int(TokenType.EVENT),
                    dtype=torch.long,
                    device=device,
                )
            )
        )

        # zero out embeddings for padded event slots to prevent slot/pos/side embedding bleeding
        event_mask = (events_cat[..., 0] != 0).to(event_tokens.dtype).unsqueeze(-1)
        event_tokens = event_tokens * event_mask

        out_tokens = torch.cat([out_tokens, event_tokens.to(out_tokens.dtype)], dim=1)

        return out_tokens, aux_moves
