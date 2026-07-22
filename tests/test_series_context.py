import torch

from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.series_context import DynamicSeriesResampler

D_MODEL = 32


def _resampler() -> DynamicSeriesResampler:
    torch.manual_seed(0)
    return DynamicSeriesResampler(
        d_model=D_MODEL,
        nhead=4,
        dim_feedforward=64,
        num_summary_tokens=SERIES_TOKENS_PER_GAME,
        num_layers=2,
    )


def test_resample_single_game_shape() -> None:
    resampler = _resampler()
    batch_size = 2
    turns = 15
    history = torch.randn(batch_size, turns, D_MODEL)
    output = resampler.resample_single_game(history)
    assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.isfinite(output).all()


def test_resample_empty_game() -> None:
    resampler = _resampler()
    batch_size = 2
    empty_history = torch.zeros(batch_size, 0, D_MODEL)
    output = resampler.resample_single_game(empty_history)
    assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.equal(output[0], resampler.empty_game_context[0])


def test_series_context_encoding_shapes() -> None:
    resampler = _resampler()
    batch_size = 2
    game1 = torch.randn(batch_size, 10, D_MODEL)
    game2 = torch.randn(batch_size, 20, D_MODEL)

    # 2 completed prior games
    series_tokens, series_mask = resampler([game1, game2])
    assert series_tokens.shape == (batch_size, SERIES_SLOTS, D_MODEL)
    assert series_mask.shape == (batch_size, SERIES_SLOTS)
    assert series_mask.all()

    # 1 completed prior game
    series_tokens_1, series_mask_1 = resampler([game1])
    assert series_tokens_1.shape == (batch_size, SERIES_SLOTS, D_MODEL)
    assert series_mask_1[:, :4].all()
    assert not series_mask_1[:, 4:].any()


def _policy() -> PolicyNet:
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        nhead=4,
        reducer_layers=1,
        dim_feedforward=128,
    )
    return build_policy(config, default_runtime_resources())


def test_policy_series_resampler() -> None:
    policy = _policy()
    game1 = torch.randn(1, 12, policy.d_model)
    tokens, mask = policy.encode_series([game1])
    assert tokens.shape == (1, SERIES_SLOTS, policy.d_model)
    assert mask.shape == (1, SERIES_SLOTS)
