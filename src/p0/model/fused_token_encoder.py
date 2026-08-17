"""Fused token encoder that turns categorical/numerical battle features into mixed context tokens.

Produces the casual observation sequence consumed by the memory reducer, including side-owned
scalars (turn, team-preview flag, fainted counts), entity rows, and pooled battle events.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from p0.battle.events import (
    NUM_ACTION_TYPES,
    NUM_TARGET_SLOTS,
    SPATIAL_NUMERICAL_WIDTH,
    SPATIAL_SLOT_COUNT,
)
from p0.format_config import FORMAT
from p0.model.architecture_contract import EVENT_RAW_WIDTH, POOLED_EVENT_COUNT
from p0.model.resources import RuntimeResources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_MECHANIC_STATE,
    CAT_IDX_NATURE,
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    CAT_IDX_STATUS,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    MAX_EFFECTS,
    MOVE_SLOTS,
    NUM_BASE_WIDTH,
    NUM_EFFECT_START,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_MOVE_LAST,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_MOVE_PP,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_STATUS_COUNTER,
    NUMERICAL_WIDTH,
    OWNER_TOKENS,
    POKEMON_TOKENS,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    IdentityKnownness,
    MechanicState,
    PresenceStatus,
    StatProvenance,
    StructuredObservation,
    TokenType,
)
from p0.model.swiglu_encoder import SwiGLUTransformerEncoder

ACT_SIZE = FORMAT.action_size
NUM_COMPONENTS = 14
NUM_TOKEN_TYPES = 3
NUM_SIDES = 3
NUM_SLOTS = 7

# 0-11 pokemon tokens (one fused token per Pokemon)
# 12 Global-field, 13 Ally-side, 14 Opponent-side (one fused token per owner)
_POKE_POS = POKEMON_TOKENS
_OWNER_POS = OWNER_TOKENS

# pp fraction, last-move flag, legal-this-step, legality-proven gate
MOVE_DYNAMIC_WIDTH = 4
STATUS_DYNAMIC_WIDTH = 1  # status counter (turns asleep / toxic stage)
SPECIES_STATIC_WIDTH = 9  # six base stats, weight, mega flag, forme relationship

# Pokemon-owned scalars: everything in the base+provenance numeric row except the
# fields owned by a narrower record (move dynamics -> MoveRecord, status counter
# -> StatusRecord). Effects live past NUM_EFFECT_START and are owned by the
# typed-effect deepset, so they are excluded structurally.
_MOVE_DYN_IDX = frozenset(
    index
    for start in (NUM_IDX_MOVE_PP, NUM_IDX_MOVE_LAST, NUM_IDX_MOVE_LEGAL)
    for index in range(start, start + MOVE_SLOTS)
)
_POKEMON_SCALAR_IDX = tuple(
    index
    for index in range(NUM_EFFECT_START)
    if index not in _MOVE_DYN_IDX and index != NUM_IDX_STATUS_COUNTER
)
POKEMON_SCALAR_WIDTH = len(_POKEMON_SCALAR_IDX)

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
    # distinct, including all (includes the user) and allAdjacent (does not).
    "normal": "selectedpokemon",
    "any": "selectedpokemon",
}
MOVE_STATIC_WIDTH = 7 + len(_TARGET_CLASSES)


def _load_vocab_sizes(resources: RuntimeResources) -> dict[str, int]:
    return {name: len(values) + 1 for name, values in resources.vocab.items()}


def _load_species_statics(resources: RuntimeResources) -> torch.Tensor:
    species_vocab = resources.vocab["species"]
    dex_species = {entry["id"]: entry for entry in resources.dex["species"]}

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


def _load_move_statics(resources: RuntimeResources) -> torch.Tensor:
    """Static per-move scalars indexed by vocab move id (row 0 = padding)."""
    moves_vocab = resources.vocab["moves"]
    dex_moves = {entry["id"]: entry for entry in resources.dex["moves"]}

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


def _load_mechanic_tag_tables(
    resources: RuntimeResources,
) -> dict[str, torch.Tensor]:
    """Load audited item and ability hook tables in one data-file pass."""
    vocab_data = resources.vocab
    dex_data = resources.dex

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
    _pokemon_scalar_idx: torch.Tensor
    _event_effect_namespace: torch.Tensor
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
        resources: RuntimeResources,
    ):
        super().__init__()
        self.resources = resources
        self.d_model = d_model
        d_raw = EVENT_RAW_WIDTH

        sizes = _load_vocab_sizes(self.resources)
        self.species_emb = nn.Embedding(sizes["species"], d_raw)
        self.ability_emb = nn.Embedding(sizes["abilities"], d_raw)
        self.item_emb = nn.Embedding(sizes["items"], d_raw)
        self.move_emb = nn.Embedding(sizes["moves"], d_raw)
        self.type_emb = nn.Embedding(sizes["types"], d_raw)
        self.category_emb = nn.Embedding(sizes["categories"], d_raw)
        self.status_emb = nn.Embedding(sizes["status"], d_raw)
        self.nature_emb = nn.Embedding(25, d_raw)
        effect_vocab_size = max(
            sizes["volatiles"], sizes["side_conditions"], sizes["fields"], sizes["weathers"]
        )
        self.effect_emb = nn.Embedding(effect_vocab_size, d_raw)
        self.counter_kind_emb = nn.Embedding(5, 16)
        self.effect_namespace_emb = nn.Embedding(5, 16)
        self.identity_knownness_emb = nn.Embedding(len(IdentityKnownness), 16)
        self.stat_provenance_emb = nn.Embedding(len(StatProvenance), 16)
        self.presence_status_emb = nn.Embedding(len(PresenceStatus), 16)
        self.mechanic_state_emb = nn.Embedding(len(MechanicState), 16)
        self.provenance_proj = nn.Linear(16 * 4, d_model)

        self.species_proj = nn.Linear(d_raw, d_model)
        self.species_static_proj = nn.Linear(SPECIES_STATIC_WIDTH, d_model)
        self.register_buffer(
            "_species_statics", _load_species_statics(self.resources), persistent=False
        )

        self.ability_proj = nn.Linear(d_raw, d_model)
        self.item_proj = nn.Linear(d_raw, d_model)
        mechanic_tags = _load_mechanic_tag_tables(self.resources)
        item_tags = mechanic_tags["items"]
        ability_tags = mechanic_tags["abilities"]
        self.item_mechanic_proj = nn.Linear(item_tags.shape[1], d_model, bias=False)
        self.ability_mechanic_proj = nn.Linear(ability_tags.shape[1], d_model, bias=False)
        self.register_buffer("_item_mechanic_tags", item_tags, persistent=False)
        self.register_buffer("_ability_mechanic_tags", ability_tags, persistent=False)

        # pooled type summary plus a non-pooled primary-type signal, so
        # order-sensitive mechanics (Revelation Dance) have a slot-aware channel
        self.type_set = MultiAggDeepSet(d_raw, d_model)
        self.primary_type_proj = nn.Linear(d_raw, d_model)

        # each move fuses its identity (move/type/category embeddings), static dex scalars,
        # and its own dynamics (pp fraction, last-used flag, legal-this-step)
        # in one projection. The same record is pooled into the Pokemon token (query side)
        # and down-projected for the pointer keys (key side)
        self.move_proj = nn.Linear(3 * d_raw + MOVE_STATIC_WIDTH + MOVE_DYNAMIC_WIDTH, d_model)
        self.move_pos_emb = nn.Embedding(4, d_model)
        self.register_buffer("_move_statics", _load_move_statics(self.resources), persistent=False)

        # the status owns its identity and its counter dynamics as one record
        self.status_proj = nn.Linear(d_raw + 16 + STATUS_DYNAMIC_WIDTH, d_model)

        self.nature_proj = nn.Linear(d_raw, d_model)

        self.typed_effect_set = MultiAggDeepSet(d_raw + 16 + 16 + EFFECT_NUMERICAL_WIDTH, d_model)

        # Pokemon-owned dynamics (boosts, hp, protect counter, ...) are one more
        # component of the Pokemon fusion kernel, not a second sequence token.
        self.pokemon_scalar_proj = nn.Sequential(
            nn.Linear(POKEMON_SCALAR_WIDTH, d_model),
            nn.GELU(),
        )
        self.register_buffer(
            "_pokemon_scalar_idx", torch.tensor(_POKEMON_SCALAR_IDX, dtype=torch.long)
        )

        # one internal fusion pass over all of the components above
        self.component_emb = nn.Embedding(NUM_COMPONENTS, d_model)
        # cache component ids instead of creating them every forward pass
        self.register_buffer("_component_ids", torch.arange(NUM_COMPONENTS))
        self.mon_fusion = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=1,
        )
        self.mon_fusion_token = nn.Parameter(torch.empty(1, 1, d_model))

        # field/side-owned scalars (turn, team-preview flag, fainted count,
        # mega availability) fused into the single owner token
        self.owner_scalar_proj = nn.Sequential(
            nn.Linear(NUM_BASE_WIDTH, d_model),
            nn.GELU(),
        )

        self.token_type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.side_emb = nn.Embedding(NUM_SIDES, d_model)
        self.slot_emb = nn.Embedding(NUM_SLOTS, d_model)
        self.action_mask_proj = nn.Sequential(
            nn.Linear(2 * ACT_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_mask_token = nn.Parameter(torch.empty(1, 1, d_model))
        # One learned marker per active slot, added in place of a mask the data source
        # could not prove. Keeps unknown as a distinct state instead of a mask value.
        self.unknown_legality_emb = nn.Parameter(torch.empty(2, d_model))

        # Spatial interaction token builder (4 slots: P1A, P1B, P2A, P2B)
        self.spatial_action_emb = nn.Embedding(NUM_ACTION_TYPES, d_model)
        self.spatial_target_emb = nn.Embedding(NUM_TARGET_SLOTS, d_model)
        self.spatial_move_proj = nn.Linear(d_raw, d_model)
        self.spatial_num_proj = nn.Linear(SPATIAL_NUMERICAL_WIDTH, d_model)
        self.spatial_slot_pos_emb = nn.Parameter(torch.empty(SPATIAL_SLOT_COUNT, d_model))

        # 4 learned spatial event queries
        self.spatial_event_queries = nn.Parameter(torch.empty(1, POOLED_EVENT_COUNT, d_model))

        # Single-layer cross-attention over the 4 spatial slots
        self.spatial_cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.spatial_norm1 = nn.LayerNorm(d_model)
        self.spatial_norm2 = nn.LayerNorm(d_model)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

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
        init.normal_(self.unknown_legality_emb, std=emb_gain)
        init.normal_(self.spatial_slot_pos_emb, std=emb_gain)
        init.normal_(self.spatial_event_queries, std=emb_gain)

    def _embed_pokemon_components(
        self, categorical: torch.Tensor, numerical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # see _pokemon_categorical_into / _pokemon_numeric_into in
        # observation_builder for the order of the per-Pokemon features
        species_ids = categorical[..., 0]
        species = self.species_proj(self.species_emb(species_ids))
        species = species + self.species_static_proj(self._species_statics[species_ids])

        ability_ids = categorical[..., 1]
        item_ids = categorical[..., 2]
        ability = self.ability_proj(self.ability_emb(ability_ids))
        ability = ability + self.ability_mechanic_proj(self._ability_mechanic_tags[ability_ids])
        item = self.item_proj(self.item_emb(item_ids))
        item = item + self.item_mechanic_proj(self._item_mechanic_tags[item_ids])

        # non-pooled primary-type channel
        # some moves like revelation dance rely on the primary type
        type_summary = self.type_set(self.type_emb(categorical[..., 3:5]))
        primary_type = self.primary_type_proj(self.type_emb(categorical[..., 3]))

        # MoveRecord: identity + static dex scalars + move-owned dynamics fused once
        move_ids = categorical[..., 5:9]
        # A row whose legality is unproven carries zeros in the legal-this-step channel;
        # the proven gate keeps that from reading as a proven illegal action this step.
        legality_proven = (
            1.0 - numerical[..., NUM_IDX_LEGALITY_UNKNOWN : NUM_IDX_LEGALITY_UNKNOWN + 1]
        )
        move_dynamics = torch.stack(
            [
                numerical[..., NUM_IDX_MOVE_PP : NUM_IDX_MOVE_PP + MOVE_SLOTS],
                numerical[..., NUM_IDX_MOVE_LAST : NUM_IDX_MOVE_LAST + MOVE_SLOTS],
                numerical[..., NUM_IDX_MOVE_LEGAL : NUM_IDX_MOVE_LEGAL + MOVE_SLOTS],
                legality_proven.expand(*legality_proven.shape[:-1], MOVE_SLOTS),
            ],
            dim=-1,
        )
        move_parts = torch.cat(
            [
                self.move_emb(move_ids),
                self.type_emb(categorical[..., 9:13]),
                self.category_emb(categorical[..., 13:17]),
                self._move_statics[move_ids],
                move_dynamics,
            ],
            dim=-1,
        )
        # pos emb added here to break permutation invariance
        # the model downstream needs to know which slot is which
        # for a1 to choose the indices
        move_embs = self.move_proj(move_parts) + self.move_pos_emb.weight

        # StatusRecord: identity + counter semantics + counter value fused once
        status = self.status_proj(
            torch.cat(
                [
                    self.status_emb(categorical[..., CAT_IDX_STATUS]),
                    self.counter_kind_emb(categorical[..., CAT_IDX_STATUS_COUNTER_KIND]),
                    numerical[..., NUM_IDX_STATUS_COUNTER : NUM_IDX_STATUS_COUNTER + 1],
                ],
                dim=-1,
            )
        )

        nature = self.nature_proj(self.nature_emb(categorical[..., CAT_IDX_NATURE]))

        effects = self._embed_typed_effects(categorical, numerical)
        scalars = self.pokemon_scalar_proj(numerical[..., self._pokemon_scalar_idx])
        provenance = self.provenance_proj(
            torch.cat(
                [
                    self.identity_knownness_emb(categorical[..., CAT_IDX_IDENTITY_KNOWNNESS]),
                    self.stat_provenance_emb(categorical[..., CAT_IDX_STAT_PROVENANCE]),
                    self.presence_status_emb(categorical[..., CAT_IDX_PRESENCE_STATUS]),
                    self.mechanic_state_emb(categorical[..., CAT_IDX_MECHANIC_STATE]),
                ],
                dim=-1,
            )
        )

        # combine all components into (N, NUM_COMPONENTS, d_model)
        # Note: move_embs is (..., 4, d_model), others are (..., d_model)
        components = torch.cat(
            [
                species.unsqueeze(-2),
                ability.unsqueeze(-2),
                item.unsqueeze(-2),
                type_summary.unsqueeze(-2),
                primary_type.unsqueeze(-2),
                move_embs,
                status.unsqueeze(-2),
                nature.unsqueeze(-2),
                effects.unsqueeze(-2),
                scalars.unsqueeze(-2),
                provenance.unsqueeze(-2),
            ],
            dim=-2,
        )
        components = components + self.component_emb(self._component_ids)

        # The internal fusion query remains the Pokemon super-token. It is not
        # an observation-level row and is therefore unaffected by CLS removal.
        N = components.shape[0]
        fusion_token = self.mon_fusion_token.expand(N, 1, -1)
        fused = self.mon_fusion(torch.cat([fusion_token, components], dim=1))

        return fused[:, 0], move_embs

    def _embed_pokemon_super(
        self, categorical: torch.Tensor, numerical: torch.Tensor
    ) -> torch.Tensor:
        """Embed Pokemon rows without exposing pointer-head auxiliary features."""
        fused_token, _ = self._embed_pokemon_components(categorical, numerical)
        return fused_token

    def _embed_typed_effects(
        self, categorical: torch.Tensor, numerical: torch.Tensor
    ) -> torch.Tensor:
        effect_cat = categorical[..., CAT_EFFECT_START:].unflatten(
            -1, (MAX_EFFECTS, EFFECT_CATEGORICAL_WIDTH)
        )
        effect_num = numerical[
            ..., NUM_EFFECT_START : NUM_EFFECT_START + MAX_EFFECTS * EFFECT_NUMERICAL_WIDTH
        ].unflatten(-1, (MAX_EFFECTS, EFFECT_NUMERICAL_WIDTH))
        features = torch.cat(
            (
                self.effect_emb(effect_cat[..., 0]),
                self.counter_kind_emb(effect_cat[..., 1]),
                self.effect_namespace_emb(effect_cat[..., 2]),
                effect_num,
            ),
            dim=-1,
        )
        return self.typed_effect_set(features, mask=effect_num[..., 0] > 0.5)

    def _encode_events(self, obs: StructuredObservation, device: torch.device) -> torch.Tensor:
        spatial_cat = obs.spatial_cat.long().to(device)
        spatial_num = obs.spatial_num.float().to(device)
        B = spatial_cat.size(0)

        # Embed raw spatial features per slot
        action_emb = self.spatial_action_emb(spatial_cat[..., 0])
        move_emb = self.spatial_move_proj(self.move_emb(spatial_cat[..., 1]))
        target_emb = self.spatial_target_emb(spatial_cat[..., 2])
        num_emb = self.spatial_num_proj(spatial_num)
        slot_emb = self.spatial_slot_pos_emb.unsqueeze(0)

        raw_kv = action_emb + move_emb + target_emb + num_emb + slot_emb

        # Single-layer cross-attention over the 4 spatial slots
        queries = self.spatial_event_queries.expand(B, -1, -1)
        attn_out, _ = self.spatial_cross_attn(
            query=queries,
            key=raw_kv,
            value=raw_kv,
            need_weights=False,
        )
        x = self.spatial_norm1(queries + attn_out)
        event_tokens = self.spatial_norm2(x + self.spatial_ffn(x))
        return event_tokens

    def _append_action_mask_token(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        """Append the joint action-mask token, gated by per-slot legality provenance."""
        B = tokens.size(0)
        if action_mask.shape != (B, 2, ACT_SIZE):
            raise ValueError(
                f"Expected action mask ({B}, 2, {ACT_SIZE}); got {tuple(action_mask.shape)}."
            )
        slot_unknown = numerical[
            :,
            TOKEN_IDX_ALLY_SIDE,
            NUM_IDX_SLOT_LEGALITY_UNKNOWN : NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2,
        ].to(tokens.dtype)
        # An unproven slot contributes no mask magnitude; its learned marker carries the
        # a fall back to state signal instead.
        gated_mask = action_mask.to(tokens.dtype) * (1.0 - slot_unknown).unsqueeze(-1)

        unknown_marker = slot_unknown @ self.unknown_legality_emb.to(tokens.dtype)
        mask_token = (
            self.action_mask_token.expand(B, -1, -1)
            + self.action_mask_proj(gated_mask.reshape(B, -1)).unsqueeze(1)
            + unknown_marker.unsqueeze(1)
        )
        return torch.cat([tokens, mask_token], dim=1)

    def forward(
        self,
        obs: StructuredObservation,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched observation into token and auxiliary-context tensors.

        Arguments:
          obs: the batched structured observation to encode
          action_mask: per-slot action mask (batch, 2, action_size) guiding masking

        Returns:
          a (tokens, aux) pair of encoded observation tensors for the reducer/policy
        """
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

        n_poke = len(_POKE_POS)
        poke_cats = categorical[:, :n_poke, :].flatten(0, 1)
        poke_nums = numerical[:, :n_poke, :].flatten(0, 1)
        poke_out, all_move_embs = self._embed_pokemon_components(poke_cats, poke_nums)

        x[:, :n_poke, :] = poke_out.unflatten(0, (batch_size, n_poke)).to(x.dtype)
        # the two active allies' MoveRecords double as the pointer-head move
        # keys; the records already carry pp/legality state, so no extra patch
        aux_moves = all_move_embs.unflatten(0, (batch_size, n_poke))[:, :2]

        n_owner = len(_OWNER_POS)
        owner_end = n_poke + n_owner
        # field / ally-side / opponent-side owners: one fused token each, from
        # the owner's typed effects plus its own scalars
        x[:, n_poke:owner_end, :] = (
            self._embed_typed_effects(
                categorical[:, n_poke:owner_end, :],
                numerical[:, n_poke:owner_end, :],
            )
            + self.owner_scalar_proj(numerical[:, n_poke:owner_end, :NUM_BASE_WIDTH])
        ).to(x.dtype)

        out_tokens = (
            x
            + self.token_type_emb(token_type_ids)
            + self.side_emb(side_ids)
            + self.slot_emb(slot_ids)
        )
        out_tokens = self._append_action_mask_token(out_tokens, action_mask, numerical)

        event_tokens = self._encode_events(obs, device)
        event_tokens = event_tokens + self.token_type_emb.weight[int(TokenType.EVENT)]

        out_tokens = torch.cat([out_tokens, event_tokens.to(out_tokens.dtype)], dim=1)
        return out_tokens, aux_moves
