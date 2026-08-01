import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore


def test_series_perspective_key_rejects_invalid_players():
    with pytest.raises(ValueError, match="canonical_player"):
        SeriesPerspectiveKey("series-1", 2)
    with pytest.raises(ValueError, match="canonical_player"):
        SeriesPerspectiveKey("series-1", True)


def test_token_store_initialization():
    store = SeriesTokenStore(d_model=64, max_games=2)
    assert store.d_model == 64
    assert store.max_games == 2
    assert store._store == {}


def test_token_store_append_and_get():
    store = SeriesTokenStore(d_model=16, max_games=2)

    tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    store.append("series-A", tokens1)

    out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
    assert out_tokens.shape == (1, SERIES_SLOTS, 16)
    assert out_mask.shape == (1, SERIES_SLOTS)

    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
    assert torch.all(out_tokens[0, SERIES_TOKENS_PER_GAME:] == 0)
    assert torch.all(out_mask[0, :SERIES_TOKENS_PER_GAME])
    assert not torch.any(out_mask[0, SERIES_TOKENS_PER_GAME:])

    tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    store.append("series-A", tokens2)

    out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
    assert torch.allclose(
        out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens2
    )
    assert torch.all(out_mask[0, : 2 * SERIES_TOKENS_PER_GAME])


def test_token_store_max_games_truncation():
    store = SeriesTokenStore(d_model=16, max_games=2)

    tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    tokens3 = torch.randn(SERIES_TOKENS_PER_GAME, 16)

    store.append("series-B", tokens1)
    store.append("series-B", tokens2)
    store.append("series-B", tokens3)

    out_tokens, out_mask = store.get_tokens(["series-B"], device=torch.device("cpu"))

    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens2)
    assert torch.allclose(
        out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens3
    )


def test_token_store_batching_and_missing():
    store = SeriesTokenStore(d_model=8)

    t1 = torch.randn(SERIES_TOKENS_PER_GAME, 8)
    store.append("s1", t1)

    out_tokens, out_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
    assert out_tokens.shape == (2, SERIES_SLOTS, 8)

    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], t1)
    assert out_mask[0, 0]

    assert torch.all(out_tokens[1] == 0)
    assert not torch.any(out_mask[1])


def test_token_store_isolates_canonical_player_perspectives():
    store = SeriesTokenStore(d_model=8)
    first_player = SeriesPerspectiveKey("series-1", 0)
    second_player = SeriesPerspectiveKey("series-1", 1)
    tokens = torch.randn(SERIES_TOKENS_PER_GAME, 8)

    store.append(first_player, tokens)

    out_tokens, out_mask = store.get_tokens(
        [first_player, second_player],
        device=torch.device("cpu"),
    )
    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens)
    assert torch.all(out_mask[0, :SERIES_TOKENS_PER_GAME])
    assert not torch.any(out_mask[1])


def test_token_store_drop_and_clear():
    store = SeriesTokenStore(d_model=8)
    store.append("s1", torch.randn(SERIES_TOKENS_PER_GAME, 8))

    store.drop("s1")
    out_tokens, out_mask = store.get_tokens(["s1"], device=torch.device("cpu"))
    assert not torch.any(out_mask)

    store.append("s2", torch.randn(SERIES_TOKENS_PER_GAME, 8))
    store.clear()
    out_tokens, out_mask = store.get_tokens(["s2"], device=torch.device("cpu"))
    assert not torch.any(out_mask)
