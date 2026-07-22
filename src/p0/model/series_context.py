"""Series-context featurization for Bo3 games.

Turns the causal per-game summaries from p0.battle.series into padded id and
scalar tensors viewed from one player's perspective. Tensorization is pure and
grad-free so BC dataloaders can run it offline and the live runtime can run it
once at game start; the encoder consuming these features re-encodes them inside
the current game's graph, so no autograd graph ever crosses a game boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from p0.battle.series import MAX_PRIOR_GAMES, GameSummary, SideGameSummary
from p0.model.swiglu_encoder import SwiGLUTransformerEncoder
from p0.model.tokenizer import PokemonTokenizer

SERIES_SIDES = 2
SERIES_POKEMON_SLOTS = 4
SERIES_MOVE_SLOTS = 4

# Model-side vocabulary for SideGameSummary.plan_tags. Unknown tags are ignored
# rather than raising: the tag grammar is a producer-side convention that may
# grow, and a newer summary must not crash an older model.
PLAN_TAGS: tuple[str, ...] = (
    "protectloop",
    "setup",
    "weather",
    "trickroom",
    "perish",
    "redirection",
)

_PLAN_TAG_INDEX = {tag: index for index, tag in enumerate(PLAN_TAGS)}

POKE_SCALAR_WIDTH = 5
SIDE_SCALAR_WIDTH = 2 + len(PLAN_TAGS)
GAME_SCALAR_WIDTH = 5

_TURNS_SCALE = 50.0
_SWITCH_SCALE = 10.0
_SPEED_OBS_SCALE = 8.0


@dataclass(frozen=True, slots=True)
class SeriesFeatures:
    """Padded per-series tensors; side slot 0 is self, slot 1 the opponent.

    Unbatched shapes use G = MAX_PRIOR_GAMES game slots, S = SERIES_SIDES,
    P = SERIES_POKEMON_SLOTS, and M = SERIES_MOVE_SLOTS. Id zero is padding or
    OOV, matching the battle tokenizer convention.
    """

    species: Tensor  # (G, S, P) long
    item: Tensor  # (G, S, P) long
    ability: Tensor  # (G, S, P) long
    moves: Tensor  # (G, S, P, M) long
    poke_scalars: Tensor  # (G, S, P, POKE_SCALAR_WIDTH) float
    side_scalars: Tensor  # (G, S, SIDE_SCALAR_WIDTH) float
    game_scalars: Tensor  # (G, GAME_SCALAR_WIDTH) float
    game_number: Tensor  # (G,) long, zero for padded game slots
    game_mask: Tensor  # (G,) bool, True for real prior games

    @classmethod
    def stack(cls, batch: Sequence[SeriesFeatures]) -> SeriesFeatures:
        if not batch:
            raise ValueError("SeriesFeatures.stack requires at least one element")
        return cls(
            **{
                field.name: torch.stack([getattr(item, field.name) for item in batch])
                for field in fields(cls)
            }
        )

    def to(self, device: torch.device | str) -> SeriesFeatures:
        """Move feature tensors without changing their grad-free contract."""
        return SeriesFeatures(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


def _side_species(side: SideGameSummary) -> tuple[str, ...]:
    # brought lists observed members only and always includes the leads when
    # nonempty; leads-first ordering keeps slot 0/1 meaning "led this game".
    ordered = list(side.leads)
    for species in side.brought:
        if species not in ordered:
            ordered.append(species)
    return tuple(ordered[:SERIES_POKEMON_SLOTS])


def _tensorize_side(
    side: SideGameSummary,
    game_slot: int,
    side_slot: int,
    tokenizer: PokemonTokenizer,
    out: SeriesFeatures,
) -> None:
    for poke_slot, species in enumerate(_side_species(side)):
        out.species[game_slot, side_slot, poke_slot] = tokenizer.id_for("species", species)
        out.item[game_slot, side_slot, poke_slot] = tokenizer.id_for(
            "items", side.revealed_items.get(species)
        )
        out.ability[game_slot, side_slot, poke_slot] = tokenizer.id_for(
            "abilities", side.revealed_abilities.get(species)
        )
        moves = side.moves_used.get(species, ())
        for move_slot, move in enumerate(moves[:SERIES_MOVE_SLOTS]):
            out.moves[game_slot, side_slot, poke_slot, move_slot] = tokenizer.id_for("moves", move)
        out.poke_scalars[game_slot, side_slot, poke_slot] = torch.tensor(
            (
                float(species in side.leads),
                float(species == side.mega_species),
                float(species in side.revealed_formes),
                1.0,
                min(len(moves) / SERIES_MOVE_SLOTS, 1.0),
            )
        )
    side_row = out.side_scalars[game_slot, side_slot]
    side_row[0] = min(side.switch_count / _SWITCH_SCALE, 1.0)
    side_row[1] = min(side.pivot_count / _SWITCH_SCALE, 1.0)
    for tag in side.plan_tags:
        tag_index = _PLAN_TAG_INDEX.get(tag)
        if tag_index is not None:
            side_row[2 + tag_index] = 1.0


def _empty_features() -> SeriesFeatures:
    shape = (MAX_PRIOR_GAMES, SERIES_SIDES, SERIES_POKEMON_SLOTS)
    return SeriesFeatures(
        species=torch.zeros(shape, dtype=torch.long),
        item=torch.zeros(shape, dtype=torch.long),
        ability=torch.zeros(shape, dtype=torch.long),
        moves=torch.zeros((*shape, SERIES_MOVE_SLOTS), dtype=torch.long),
        poke_scalars=torch.zeros((*shape, POKE_SCALAR_WIDTH)),
        side_scalars=torch.zeros((MAX_PRIOR_GAMES, SERIES_SIDES, SIDE_SCALAR_WIDTH)),
        game_scalars=torch.zeros((MAX_PRIOR_GAMES, GAME_SCALAR_WIDTH)),
        game_number=torch.zeros(MAX_PRIOR_GAMES, dtype=torch.long),
        game_mask=torch.zeros(MAX_PRIOR_GAMES, dtype=torch.bool),
    )


def empty_series_features() -> SeriesFeatures:
    """Return the canonical all-padding series input for training adapters."""
    return _empty_features()


def tensorize_series(
    prior_games: Sequence[GameSummary],
    player_index: int,
    tokenizer: PokemonTokenizer,
) -> SeriesFeatures:
    """Featurize up to MAX_PRIOR_GAMES summaries from one player's perspective.

    Zero prior games is the canonical Game 1 input and yields all-padded
    features with an all-False game mask.
    """
    if player_index not in (0, 1):
        raise ValueError(f"player_index must be 0 or 1, got {player_index!r}")
    if len(prior_games) > MAX_PRIOR_GAMES:
        raise ValueError(f"At most {MAX_PRIOR_GAMES} prior games can be encoded")
    for slot, game in enumerate(prior_games):
        if game.game_number != slot + 1:
            raise ValueError(
                f"prior_games must be games 1..{len(prior_games)} in order; "
                f"slot {slot} holds game {game.game_number}"
            )

    out = _empty_features()
    for slot, game in enumerate(prior_games):
        if game.winner == player_index:
            winner_sign = 1.0
        elif game.winner == 1 - player_index:
            winner_sign = -1.0
        else:
            winner_sign = 0.0
        out.game_scalars[slot] = torch.tensor(
            (
                winner_sign,
                game.series_score[player_index] / 2.0,
                game.series_score[1 - player_index] / 2.0,
                min(game.turns / _TURNS_SCALE, 1.0),
                min(len(game.speed_observations) / _SPEED_OBS_SCALE, 1.0),
            )
        )
        out.game_number[slot] = game.game_number
        out.game_mask[slot] = True
        _tensorize_side(game.sides[player_index], slot, 0, tokenizer, out)
        _tensorize_side(game.sides[1 - player_index], slot, 1, tokenizer, out)
    return out


# Tokens per game slot: one record per Pokemon slot on each side, one record
# per side, and one game-level record.
_TOKENS_PER_GAME = SERIES_SIDES * (SERIES_POKEMON_SLOTS + 1) + 1


class SeriesContextEncoder(nn.Module):
    """Encode batched SeriesFeatures into a fixed number of context tokens.

    Entity embeddings are shared with the battle encoder by module identity,
    so the shared tables appear under two state_dict prefixes; strict loading
    handles tied weights and parameters() deduplicates them for optimizers.
    Batches with zero prior games return a learned empty context exactly, so
    Game 1 conditioning is a trained prior independent of transformer weights.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        species_emb: nn.Embedding,
        move_emb: nn.Embedding,
        item_emb: nn.Embedding,
        ability_emb: nn.Embedding,
    ):
        super().__init__()
        d_raw = species_emb.embedding_dim
        for name, emb in (
            ("move_emb", move_emb),
            ("item_emb", item_emb),
            ("ability_emb", ability_emb),
        ):
            if emb.embedding_dim != d_raw:
                raise ValueError(f"{name} width {emb.embedding_dim} does not match {d_raw}")
        self.d_model = d_model
        self.species_emb = species_emb
        self.move_emb = move_emb
        self.item_emb = item_emb
        self.ability_emb = ability_emb

        self.poke_proj = nn.Linear(4 * d_raw + POKE_SCALAR_WIDTH, d_model)
        self.side_scalar_proj = nn.Linear(SIDE_SCALAR_WIDTH, d_model)
        self.game_scalar_proj = nn.Linear(GAME_SCALAR_WIDTH, d_model)
        self.game_number_emb = nn.Embedding(MAX_PRIOR_GAMES + 1, d_model)
        self.game_pos_emb = nn.Embedding(MAX_PRIOR_GAMES, d_model)
        # 2-row self/opponent table local to this module; the battle side table
        # is 3-row with different index semantics (field/ally/opponent)
        self.side_emb = nn.Embedding(SERIES_SIDES, d_model)
        self.series_queries = nn.Parameter(torch.empty(1, MAX_PRIOR_GAMES, d_model))
        self.empty_context = nn.Parameter(torch.empty(1, MAX_PRIOR_GAMES, d_model))
        self.encoder = SwiGLUTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=2,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        emb_gain = self.d_model**-0.5
        init.normal_(self.series_queries, std=emb_gain)
        init.normal_(self.empty_context, std=emb_gain)
        shared = {self.species_emb, self.move_emb, self.item_emb, self.ability_emb}
        for module in self.modules():
            if module in shared:
                continue
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, std=emb_gain)

    def _validate_batch(self, features: SeriesFeatures) -> int:
        # series context is re-encoded from grad-free summaries inside the
        # current game's graph; a grad-carrying input would smuggle an autograd
        # edge across the game boundary
        for field in fields(SeriesFeatures):
            if getattr(features, field.name).requires_grad:
                raise ValueError(f"SeriesFeatures.{field.name} must not require grad")
        shape = features.species.shape
        expected = (MAX_PRIOR_GAMES, SERIES_SIDES, SERIES_POKEMON_SLOTS)
        if len(shape) != 4 or shape[1:] != expected:
            raise ValueError(
                f"Expected batched species features (B, {MAX_PRIOR_GAMES}, {SERIES_SIDES}, "
                f"{SERIES_POKEMON_SLOTS}); got {tuple(shape)}"
            )
        batch_size = shape[0]
        if features.moves.shape != (*shape, SERIES_MOVE_SLOTS):
            raise ValueError(f"Unexpected moves shape {tuple(features.moves.shape)}")
        if features.game_mask.shape != (batch_size, MAX_PRIOR_GAMES):
            raise ValueError(f"Unexpected game_mask shape {tuple(features.game_mask.shape)}")
        return batch_size

    def forward(self, features: SeriesFeatures) -> Tensor:
        batch_size = self._validate_batch(features)
        device = self.series_queries.device

        species = features.species.to(device)
        item = features.item.to(device)
        ability = features.ability.to(device)
        moves = features.moves.to(device)
        poke_scalars = features.poke_scalars.to(device)
        side_scalars = features.side_scalars.to(device)
        game_scalars = features.game_scalars.to(device)
        game_number = features.game_number.to(device)
        game_mask = features.game_mask.to(device)

        move_vectors = self.move_emb(moves)
        move_present = (moves != 0).unsqueeze(-1).to(move_vectors.dtype)
        move_mean = (move_vectors * move_present).sum(dim=-2) / move_present.sum(dim=-2).clamp_min(
            1.0
        )

        poke_tokens = self.poke_proj(
            torch.cat(
                [
                    self.species_emb(species),
                    self.item_emb(item),
                    self.ability_emb(ability),
                    move_mean,
                    poke_scalars,
                ],
                dim=-1,
            )
        )
        side_tokens = self.side_scalar_proj(side_scalars)

        side_ids = torch.arange(SERIES_SIDES, device=device)
        poke_tokens = poke_tokens + self.side_emb(side_ids)[None, None, :, None, :]
        side_tokens = side_tokens + self.side_emb(side_ids)[None, None, :, :]

        game_tokens = self.game_scalar_proj(game_scalars) + self.game_number_emb(game_number)

        # Encode each completed game independently so each fixed series slot
        # contains exactly one summary and no game can attend to another.
        per_game = torch.cat(
            [
                poke_tokens.flatten(2, 3),
                side_tokens,
                game_tokens.unsqueeze(2),
            ],
            dim=2,
        )
        per_game = per_game + self.game_pos_emb(
            torch.arange(MAX_PRIOR_GAMES, device=device)
        ).unsqueeze(1)
        queries = (
            self.series_queries.expand(batch_size, -1, -1)
            + self.game_pos_emb(torch.arange(MAX_PRIOR_GAMES, device=device))[None]
        )
        sequence = torch.cat([queries.unsqueeze(2), per_game], dim=2)
        sequence = sequence.reshape(
            batch_size * MAX_PRIOR_GAMES, 1 + _TOKENS_PER_GAME, self.d_model
        )

        poke_padding = (poke_scalars[..., 3] < 0.5).flatten(2, 3)
        record_padding = torch.zeros(
            batch_size,
            MAX_PRIOR_GAMES,
            SERIES_SIDES + 1,
            dtype=torch.bool,
            device=device,
        )
        game_padding = torch.cat([poke_padding, record_padding], dim=2) | ~game_mask[:, :, None]
        padding = torch.cat(
            [
                torch.zeros(batch_size, MAX_PRIOR_GAMES, 1, dtype=torch.bool, device=device),
                game_padding,
            ],
            dim=2,
        ).reshape(batch_size * MAX_PRIOR_GAMES, 1 + _TOKENS_PER_GAME)

        encoded = self.encoder(sequence, src_key_padding_mask=padding)[:, 0]
        encoded = encoded.reshape(batch_size, MAX_PRIOR_GAMES, self.d_model)
        return torch.where(
            game_mask[:, :, None],
            encoded,
            self.empty_context.expand(batch_size, -1, -1).to(encoded.dtype),
        )
