from dataclasses import fields

import pytest
import torch
import torch.nn as nn

from p0.battle.series import MAX_PRIOR_GAMES, GameSummary, SideGameSummary
from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.series_context import (
    GAME_SCALAR_WIDTH,
    PLAN_TAGS,
    POKE_SCALAR_WIDTH,
    SERIES_MOVE_SLOTS,
    SERIES_POKEMON_SLOTS,
    SERIES_SIDES,
    SIDE_SCALAR_WIDTH,
    SeriesContextEncoder,
    SeriesFeatures,
    SeriesStateConditioner,
    tensorize_series,
)
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
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


D_MODEL = 32
D_RAW = 16
SERIES_TOKENS = 3


def _encoder() -> SeriesContextEncoder:
    vocab = default_runtime_resources().vocab
    torch.manual_seed(0)
    return SeriesContextEncoder(
        d_model=D_MODEL,
        nhead=4,
        dim_feedforward=64,
        series_tokens=SERIES_TOKENS,
        species_emb=nn.Embedding(len(vocab["species"]) + 1, D_RAW),
        move_emb=nn.Embedding(len(vocab["moves"]) + 1, D_RAW),
        item_emb=nn.Embedding(len(vocab["items"]) + 1, D_RAW),
        ability_emb=nn.Embedding(len(vocab["abilities"]) + 1, D_RAW),
    )


def _mixed_batch() -> SeriesFeatures:
    tokenizer = _tokenizer()
    with_games = tensorize_series(
        (_game(1, 0, (1, 0)), _game(2, 1, (1, 1))), player_index=0, tokenizer=tokenizer
    )
    empty = tensorize_series((), player_index=0, tokenizer=tokenizer)
    return SeriesFeatures.stack([with_games, empty])


def test_encoder_output_shape() -> None:
    context = _encoder()(_mixed_batch())
    assert context.shape == (2, SERIES_TOKENS, D_MODEL)
    assert torch.isfinite(context).all()


def test_empty_rows_return_learned_empty_context() -> None:
    encoder = _encoder()
    context = encoder(_mixed_batch())
    assert torch.equal(context[1], encoder.empty_context[0])
    assert not torch.equal(context[0], encoder.empty_context[0])


def test_encoder_rejects_unbatched_features() -> None:
    single = tensorize_series((), player_index=0, tokenizer=_tokenizer())
    with pytest.raises(ValueError, match="batched"):
        _encoder()(single)


def test_encoder_rejects_grad_carrying_features() -> None:
    batch = _mixed_batch()
    leaky = SeriesFeatures(
        **{
            field.name: getattr(batch, field.name)
            for field in fields(SeriesFeatures)
            if field.name != "poke_scalars"
        },
        poke_scalars=batch.poke_scalars.clone().requires_grad_(True),
    )
    with pytest.raises(ValueError, match="must not require grad"):
        _encoder()(leaky)


def test_encoder_grads_reach_projections() -> None:
    encoder = _encoder()
    context = encoder(_mixed_batch())
    # a plain sum has near-zero gradient through the encoder's final LayerNorm
    # (the normalized output sums to a constant), so square first
    context.pow(2).sum().backward()
    assert encoder.series_queries.grad is not None
    assert encoder.poke_proj.weight.grad is not None
    assert encoder.poke_proj.weight.grad.abs().sum() > 0


def test_conditioner_zero_gate_is_identity() -> None:
    torch.manual_seed(0)
    conditioner = SeriesStateConditioner(D_MODEL, nhead=4, history_tokens=8)
    hg_init = torch.randn(1, 8, D_MODEL)
    context = torch.randn(5, SERIES_TOKENS, D_MODEL)
    state = conditioner(hg_init, context)
    assert torch.equal(state, hg_init.expand(5, -1, -1))
    with pytest.raises(ValueError, match="hg_init"):
        conditioner(torch.randn(1, 4, D_MODEL), context)


def test_conditioner_gate_admits_context() -> None:
    torch.manual_seed(0)
    conditioner = SeriesStateConditioner(D_MODEL, nhead=4, history_tokens=8)
    with torch.no_grad():
        conditioner.gate.fill_(1.0)
    hg_init = torch.randn(1, 8, D_MODEL)
    state_a = conditioner(hg_init, torch.randn(2, SERIES_TOKENS, D_MODEL))
    state_b = conditioner(hg_init, torch.randn(2, SERIES_TOKENS, D_MODEL))
    assert not torch.equal(state_a, state_b)


def _policy(enabled: bool) -> PolicyNet:
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        nhead=4,
        prelude_layers=1,
        history_tokens=8,
        dim_feedforward=128,
        series_context_enabled=enabled,
        series_tokens=SERIES_TOKENS,
    )
    return build_policy(config, default_runtime_resources())


def _dummy_obs(batch_size: int) -> StructuredObservation:
    return StructuredObservation(
        token_type_ids=torch.zeros((batch_size, SEQUENCE_LENGTH), dtype=torch.long),
        side_ids=torch.zeros((batch_size, SEQUENCE_LENGTH), dtype=torch.long),
        slot_ids=torch.zeros((batch_size, SEQUENCE_LENGTH), dtype=torch.long),
        categorical=torch.zeros((batch_size, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        numerical=torch.zeros((batch_size, SEQUENCE_LENGTH, NUMERICAL_WIDTH)),
        events_cat=torch.zeros(
            (batch_size, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH), dtype=torch.long
        ),
        events_num=torch.zeros((batch_size, EVENT_COUNT, EVENT_NUMERICAL_WIDTH)),
        events_side_ids=torch.zeros((batch_size, EVENT_COUNT), dtype=torch.long),
        events_slot_ids=torch.zeros((batch_size, EVENT_COUNT), dtype=torch.long),
    )


def test_disabled_policy_has_no_series_modules() -> None:
    policy = _policy(enabled=False)
    assert not any(name.startswith("series") for name in policy.state_dict())
    empty = SeriesFeatures.stack([tensorize_series((), player_index=0, tokenizer=_tokenizer())])
    with pytest.raises(ValueError, match="series_context_enabled"):
        policy.initial_series_state(empty)


def test_series_embeddings_shared_with_battle_encoder() -> None:
    policy = _policy(enabled=True)
    assert policy.series.species_emb is policy.encoder.species_emb
    assert policy.series.move_emb is policy.encoder.move_emb
    assert policy.series.item_emb is policy.encoder.item_emb
    assert policy.series.ability_emb is policy.encoder.ability_emb
    # tied tables appear under both prefixes in the state dict but only once
    # in the optimizer-facing parameter list
    named = dict(policy.named_parameters())
    assert "encoder.species_emb.weight" in named
    assert "series.species_emb.weight" not in named
    assert "series.species_emb.weight" in policy.state_dict()


def test_initial_series_state_matches_initial_state_at_init() -> None:
    policy = _policy(enabled=True)
    tokenizer = _tokenizer()
    empty = tensorize_series((), player_index=0, tokenizer=tokenizer)
    with_game = tensorize_series((_game(1, 0, (1, 0)),), player_index=0, tokenizer=tokenizer)
    features = SeriesFeatures.stack([empty, with_game])
    # the zero-initialized gate makes conditioning an exact no-op until trained
    assert torch.equal(policy.initial_series_state(features), policy.initial_state(2))


def test_game2_gradients_reach_series_encoder_but_not_game1() -> None:
    policy = _policy(enabled=True)
    with torch.no_grad():
        policy.series_conditioner.gate.fill_(0.5)
    obs = _dummy_obs(1)
    action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.uint8)

    game1_out = policy.act_obs(obs, action_mask, policy.initial_state(1))
    # retain_grad so grads would be observable if the Game 2 graph reached
    # these Game 1 execution tensors
    game1_out.value.retain_grad()
    game1_out.state.retain_grad()
    features = SeriesFeatures.stack(
        [tensorize_series((_game(1, 0, (1, 0)),), player_index=0, tokenizer=_tokenizer())]
    )
    for field in fields(SeriesFeatures):
        assert not getattr(features, field.name).requires_grad

    state = policy.initial_series_state(features)
    game2_out = policy.act_obs(obs, action_mask, state)
    loss = game2_out.value.mean() - game2_out.log_probs.mean()
    loss.backward()

    assert policy.series.series_queries.grad is not None
    assert policy.series.poke_proj.weight.grad is not None
    assert policy.series.poke_proj.weight.grad.abs().sum() > 0
    assert policy.series_conditioner.gate.grad is not None
    assert game1_out.value.grad is None and game1_out.state.grad is None
