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
from torch import Tensor

from p0.battle.series import MAX_PRIOR_GAMES, GameSummary, SideGameSummary
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
