import pytest
import torch

from p0.battle.series import MAX_PRIOR_GAMES, GameSummary, SideGameSummary
from p0.model.resources import default_runtime_resources
from p0.model.series_context import (
    GAME_SCALAR_WIDTH,
    PLAN_TAGS,
    POKE_SCALAR_WIDTH,
    SERIES_MOVE_SLOTS,
    SERIES_POKEMON_SLOTS,
    SERIES_SIDES,
    SIDE_SCALAR_WIDTH,
    SeriesFeatures,
    tensorize_series,
)

def _side(lead_a: str, lead_b: str, extra: str) -> SideGameSummary:
    return SideGameSummary(
        leads=(lead_a, lead_b),
        brought=(lead_a, lead_b, extra),
        mega_species="",
        moves_used={lead_a: ("flamethrower", "protect")},
        revealed_items={lead_a: "leftovers"},
        revealed_abilities={lead_a: "blaze"},
        revealed_formes=(),
        switch_count=3,
        pivot_count=1,
        plan_tags=("weather", "notatag"),
    )


def _game(game_number: int, winner: int, score: tuple[int, int]) -> GameSummary:
    return GameSummary(
        game_number=game_number,
        winner=winner,
        series_score=score,
        turns=12,
        sides=(
            _side("charizard", "garchomp", "pikachu"),
            _side("incineroar", "pikachu", "charizard"),
        ),
        speed_observations=("charizard>incineroar",),
    )


def _tokenizer():
    return default_runtime_resources().tokenizer


def test_empty_series_is_all_padding() -> None:
    features = tensorize_series((), player_index=0, tokenizer=_tokenizer())
    shape = (MAX_PRIOR_GAMES, SERIES_SIDES, SERIES_POKEMON_SLOTS)
    assert features.species.shape == shape
    assert features.moves.shape == (*shape, SERIES_MOVE_SLOTS)
    assert features.poke_scalars.shape == (*shape, POKE_SCALAR_WIDTH)
    assert features.side_scalars.shape == (MAX_PRIOR_GAMES, SERIES_SIDES, SIDE_SCALAR_WIDTH)
    assert features.game_scalars.shape == (MAX_PRIOR_GAMES, GAME_SCALAR_WIDTH)
    assert not features.game_mask.any()
    assert features.species.eq(0).all() and features.game_scalars.eq(0.0).all()


def test_perspective_flip() -> None:
    tokenizer = _tokenizer()
    game = _game(1, winner=1, score=(0, 1))
    p0_view = tensorize_series((game,), player_index=0, tokenizer=tokenizer)
    p1_view = tensorize_series((game,), player_index=1, tokenizer=tokenizer)

    charizard = tokenizer.id_for("species", "charizard")
    incineroar = tokenizer.id_for("species", "incineroar")
    assert p0_view.species[0, 0, 0] == charizard and p0_view.species[0, 1, 0] == incineroar
    assert p1_view.species[0, 0, 0] == incineroar and p1_view.species[0, 1, 0] == charizard

    assert p0_view.game_scalars[0, 0] == -1.0 and p1_view.game_scalars[0, 0] == 1.0
    assert p0_view.game_scalars[0, 1] == 0.0 and p0_view.game_scalars[0, 2] == 0.5
    assert p1_view.game_scalars[0, 1] == 0.5 and p1_view.game_scalars[0, 2] == 0.0
    assert p0_view.game_mask[0] and not p0_view.game_mask[1]
    assert p0_view.game_number.tolist() == [1, 0]


def test_side_features() -> None:
    tokenizer = _tokenizer()
    features = tensorize_series((_game(1, 0, (1, 0)),), player_index=0, tokenizer=tokenizer)

    assert features.item[0, 0, 0] == tokenizer.id_for("items", "leftovers")
    assert features.ability[0, 0, 0] == tokenizer.id_for("abilities", "blaze")
    assert features.moves[0, 0, 0, 0] == tokenizer.id_for("moves", "flamethrower")
    assert features.moves[0, 0, 0, 1] == tokenizer.id_for("moves", "protect")
    assert features.moves[0, 0, 0, 2] == 0

    lead_scalars = features.poke_scalars[0, 0, 0]
    bench_scalars = features.poke_scalars[0, 0, 2]
    assert lead_scalars.tolist() == [1.0, 0.0, 0.0, 1.0, 0.5]
    assert bench_scalars.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert features.poke_scalars[0, 0, 3].eq(0.0).all()

    side_row = features.side_scalars[0, 0]
    assert side_row[0] == pytest.approx(0.3) and side_row[1] == pytest.approx(0.1)
    weather_slot = 2 + PLAN_TAGS.index("weather")
    assert side_row[weather_slot] == 1.0
    assert side_row[2:].sum() == 1.0


def test_tensorize_validation() -> None:
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="player_index"):
        tensorize_series((), player_index=2, tokenizer=tokenizer)
    games = (_game(1, 0, (1, 0)), _game(2, 1, (1, 1)), _game(3, 0, (2, 1)))
    with pytest.raises(ValueError, match="At most"):
        tensorize_series(games, player_index=0, tokenizer=tokenizer)
    with pytest.raises(ValueError, match="in order"):
        tensorize_series((_game(2, 0, (1, 0)),), player_index=0, tokenizer=tokenizer)


def test_stack_batches_features() -> None:
    tokenizer = _tokenizer()
    single = tensorize_series((_game(1, 0, (1, 0)),), player_index=0, tokenizer=tokenizer)
    empty = tensorize_series((), player_index=0, tokenizer=tokenizer)
    batch = SeriesFeatures.stack([single, empty])
    assert batch.species.shape[0] == 2
    assert batch.game_mask.tolist() == [[True, False], [False, False]]
    assert torch.equal(batch.species[0], single.species)
    with pytest.raises(ValueError, match="at least one"):
        SeriesFeatures.stack([])
