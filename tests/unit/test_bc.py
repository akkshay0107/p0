from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_TOKENS_PER_GAME
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.replays.compile import compile_payloads, compile_to_shards, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, ReplayGameChunk, assign_series_splits
from p0.replays.schema import LabelKind
from p0.replays.scrape import HttpResponse, ReplayFetcher, ScrapeConfig, load_raw_replay
from p0.training.bc import BCGameWindow, BCTrainer, collate_bc_batches, compute_bc_objective
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from tests.unit.test_replay import _payload_replay_dataset as _payload


def test_exact_and_partial_losses_match_probability_definitions() -> None:
    log_probs = torch.tensor([math.log(0.25), math.log(0.5), math.log(0.25)])
    offsets = torch.tensor([0, 1, 3, 3], dtype=torch.long)
    labels = torch.tensor([int(LabelKind.EXACT), int(LabelKind.PARTIAL), int(LabelKind.UNKNOWN)])
    loss_mask = torch.tensor([1.0, 1.0, 0.0])

    result = compute_bc_objective(log_probs, offsets, labels, loss_mask)

    expected_exact = -math.log(0.25)
    expected_partial = -math.log(0.75)
    assert result.exact_count == 1 and result.partial_count == 1
    assert result.labeled_count == 2
    assert result.loss_weight == 2.0
    assert result.exact_nll.item() == pytest.approx(expected_exact)
    assert result.partial_nll.item() == pytest.approx(expected_partial)
    assert result.loss.item() == pytest.approx((expected_exact + expected_partial) / 2)
    assert result.marginal_log_probs[2].isneginf()


def test_fractional_loss_weights_do_not_change_labeled_counts() -> None:
    result = compute_bc_objective(
        torch.log(torch.tensor([0.25, 0.75])),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([int(LabelKind.EXACT), int(LabelKind.EXACT)]),
        torch.tensor([0.25, 0.75]),
    )

    assert result.labeled_count == 2
    assert result.loss_weight == 1.0
    assert result.loss.item() == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))


def test_unknown_steps_have_zero_loss_and_preserve_boundaries() -> None:
    log_probs = torch.empty(0, requires_grad=True)
    offsets = torch.tensor([0, 0, 0], dtype=torch.long)
    labels = torch.tensor([int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)])
    loss_mask = torch.zeros(2)

    result = compute_bc_objective(log_probs, offsets, labels, loss_mask)
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert log_probs.grad is not None and log_probs.grad.numel() == 0


def test_partial_loss_is_candidate_order_invariant() -> None:
    first = compute_bc_objective(
        torch.log(torch.tensor([0.2, 0.3, 0.5])),
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([int(LabelKind.PARTIAL)]),
        torch.ones(1),
    )
    second = compute_bc_objective(
        torch.log(torch.tensor([0.5, 0.2, 0.3])),
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([int(LabelKind.PARTIAL)]),
        torch.ones(1),
    )
    torch.testing.assert_close(first.loss, second.loss)


def test_candidate_objective_preserves_gradients() -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    log_probs = probabilities.log()
    result = compute_bc_objective(
        log_probs,
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([int(LabelKind.PARTIAL)]),
        torch.ones(1),
    )
    result.loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.parametrize(
    "labels, offsets, mask, message",
    [
        ([int(LabelKind.EXACT)], [0, 2], [1.0], "EXACT"),
        ([int(LabelKind.PARTIAL)], [0, 1], [1.0], "PARTIAL"),
        ([int(LabelKind.UNKNOWN)], [0, 1], [0.0], "UNKNOWN"),
        ([99], [0, 0], [0.0], "unsupported"),
    ],
)
def test_invalid_label_and_candidate_shapes_are_rejected(labels, offsets, mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_bc_objective(
            torch.full((offsets[-1],), math.log(0.5)),
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(labels),
            torch.tensor(mask),
        )


def test_replay_to_series_bc_checkpoint_smoke(tmp_path) -> None:
    result = compile_payloads((_payload("game-1"), _payload("game-2")))
    built = write_tensor_shards(
        result,
        tmp_path / "shards",
        max_decisions_per_shard=8,
        created_at="2026-01-01T00:00:00Z",
    )
    dataset = LazyReplayDataset(built.manifest_path)
    policy = build_policy(
        ModelConfig(
            d_model=64,
            nhead=4,
            reducer_layers=1,
            dim_feedforward=128,
        ),
        default_runtime_resources(),
    )
    trainer = BCTrainer(
        policy,
        dataset,
        BCConfig(
            batch_decisions=2,
            learning_rate=1e-3,
            epochs=1,
            amp=False,
        ),
        device="cpu",
    )

    metrics = trainer.train()

    assert metrics["decisions"] == 8
    assert metrics["labeled_decisions"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    checkpoint = tmp_path / "bc.pt"
    trainer.save_checkpoint(checkpoint, epoch=1)

    restored = build_policy(trainer.policy.config, default_runtime_resources())
    restored_trainer = BCTrainer(
        restored,
        (),
        trainer.config,
        device="cpu",
    )
    assert restored_trainer.load_checkpoint(checkpoint) == 1
    for name, parameter in trainer.policy.state_dict().items():
        torch.testing.assert_close(parameter, restored_trainer.policy.state_dict()[name])


def _chunk(
    label_kind: list[int],
    candidate_values: list[tuple[int, int]],
    offsets: list[int],
    *,
    game_number: int = 1,
    is_series_end: bool = False,
):
    length = len(label_kind)
    observations = StructuredObservation.empty_batch(length)
    action_mask = torch.zeros((length, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True
    return ReplayGameChunk(
        series_id="series-1",
        game_number=game_number,
        player=0,
        canonical_player=0,
        observations=observations,
        action_mask=action_mask,
        mask_provenance=torch.ones(length, dtype=torch.long),
        label_kind=torch.tensor(label_kind, dtype=torch.long),
        label_confidence=torch.ones(length),
        loss_mask=torch.tensor([float(kind != int(LabelKind.UNKNOWN)) for kind in label_kind]),
        decision_type=torch.ones(length, dtype=torch.long),
        exact_action=torch.tensor([(7, 8)] * length, dtype=torch.long),
        candidate_values=torch.tensor(candidate_values, dtype=torch.long).reshape(-1, 2),
        candidate_offsets=torch.tensor(offsets, dtype=torch.long),
        outcome=torch.ones(length),
        is_series_end=is_series_end,
    )


def _trainer(chunk: ReplayGameChunk, *, minibatch_size: int = 2) -> BCTrainer:
    policy = build_policy(
        ModelConfig(64, 4, 1, 128),
        default_runtime_resources(),
    )
    return BCTrainer(
        policy,
        (chunk,),
        BCConfig(
            batch_decisions=minibatch_size,
            learning_rate=1e-3,
            epochs=1,
            num_workers=0,
            amp=False,
        ),
        device="cpu",
    )


def test_bc_trainer_updates_policy_in_game_local_chunks() -> None:
    chunk = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    trainer = _trainer(chunk, minibatch_size=1)
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.policy.named_parameters()
    }

    metrics = trainer.train()

    assert metrics["decisions"] == 2
    assert metrics["labeled_decisions"] == 2
    assert metrics["updates"] == 1
    assert metrics["games"] == 1
    assert metrics["decisions_per_update"] == 2
    assert metrics["games_per_update"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.policy.named_parameters()
    )


def test_unknown_decision_is_excluded_without_breaking_game_context() -> None:
    chunk = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT)],
        [(7, 8)],
        [0, 0, 1],
    )
    metrics = _trainer(chunk).train()
    assert metrics["decisions"] == 2
    assert metrics["labeled_decisions"] == 1
    assert metrics["exact_decisions"] == 1


def test_unknown_only_game_does_not_report_an_optimizer_update() -> None:
    chunk = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)],
        [],
        [0, 0, 0],
    )

    metrics = _trainer(chunk, minibatch_size=1).train()

    assert metrics["updates"] == 0
    assert metrics["games"] == 1
    assert metrics["decisions_per_update"] == 0.0
    assert metrics["games_per_update"] == 0.0


def test_game_boundary_accumulation_uses_total_loss_weight() -> None:
    chunk = _chunk(
        [int(LabelKind.EXACT)] * 4,
        [(7, 8)] * 4,
        [0, 1, 2, 3, 4],
    )
    whole_game = _trainer(chunk, minibatch_size=4)
    decision_chunks = _trainer(chunk, minibatch_size=1)

    with patch.object(
        whole_game,
        "_step_optimizer",
        wraps=whole_game._step_optimizer,
    ) as whole_step:
        whole_metrics = whole_game.train()
    with patch.object(
        decision_chunks,
        "_step_optimizer",
        wraps=decision_chunks._step_optimizer,
    ) as chunk_step:
        chunk_metrics = decision_chunks.train()

    assert whole_metrics["updates"] == chunk_metrics["updates"] == 1
    assert whole_step.call_args.args == chunk_step.call_args.args == (4.0,)


def test_bc_target_windows_keep_only_past_48_local_tokens() -> None:
    length = 52
    chunk = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    _, batch = collate_bc_batches((chunk,), 48)

    assert batch.target_indices.tolist() == [48, 49, 50, 51]
    for target, indices, mask in zip(
        batch.target_indices,
        batch.history_indices,
        batch.history_mask,
        strict=True,
    ):
        expected = torch.arange(max(0, int(target) - HISTORY_WINDOW), int(target))
        torch.testing.assert_close(indices[mask], expected)
    assert 0 not in batch.history_indices[-1, batch.history_mask[-1]]


def test_collated_context_is_compact_and_never_crosses_game_boundaries() -> None:
    first = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = replace(second, series_id="series-2")

    batch = next(collate_bc_batches((first, second), 4))

    assert all(isinstance(window, BCGameWindow) for window in batch.windows)
    assert batch.target_indices.tolist() == [0, 1, 2, 3]
    assert not torch.any(batch.history_mask[2])
    assert batch.history_indices[3, -1] == 2
    assert batch.history_mask[3, -1]
    for tensor in batch.observations.tensors():
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_model_inputs_encode_all_game_windows_once() -> None:
    first = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = replace(second, canonical_player=1)
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))

    with (
        torch.inference_mode(),
        patch.object(trainer.policy, "encode", wraps=trainer.policy.encode) as encode,
    ):
        model_inputs = trainer._prepare_model_inputs(batch).model_inputs

    assert encode.call_count == 1
    assert model_inputs[0].tokens.size(0) == batch.decisions
    assert model_inputs[4].shape[:2] == (batch.decisions, HISTORY_WINDOW)


def test_continued_chunk_matches_per_window_reference_inputs() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    trainer = _trainer(game, minibatch_size=64)
    trainer.policy.eval()
    batch = list(collate_bc_batches((game,), 64))[1]

    with torch.inference_mode():
        actual = trainer._prepare_model_inputs(batch).model_inputs
        context_start = 64 - HISTORY_WINDOW
        encoded = trainer.policy.encode(
            game.observations[context_start:],
            game.action_mask[context_start:],
        )
        local_tokens = trainer.policy.local_history_tokens(encoded)
        expected_history_tokens = local_tokens[
            batch.history_indices
        ] * batch.history_mask.unsqueeze(-1)

    torch.testing.assert_close(actual[0].tokens, encoded.tokens[batch.target_indices])
    torch.testing.assert_close(actual[0].aux, encoded.aux[batch.target_indices])
    torch.testing.assert_close(actual[0].numerical, encoded.numerical[batch.target_indices])
    torch.testing.assert_close(actual[4], expected_history_tokens)
    torch.testing.assert_close(actual[5], batch.history_mask)
    torch.testing.assert_close(actual[6], batch.history_age_ids)


def test_completed_game_does_not_emit_a_second_observation_payload() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )

    first, second = collate_bc_batches((game,), 64)

    assert not first.windows[0].is_game_end
    assert second.windows[0].is_game_end
    assert not hasattr(second, "completed_games")
    assert second.observations.categorical.size(0) == HISTORY_WINDOW + length - 64


def test_raw_game_history_caches_each_target_token_once() -> None:
    length = 100
    first = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=64)

    with patch.object(
        trainer.policy.series,
        "resample_single_game",
        wraps=trainer.policy.series.resample_single_game,
    ) as resample:
        trainer.evaluate((first, second))

    assert resample.call_count == 1
    history, history_mask = resample.call_args.args
    assert history.shape == (1, length, trainer.policy.d_model)
    assert history_mask.shape == (1, length)
    assert history_mask.all()


def test_ordered_history_validation_logic():
    def _test_ordered_history_allows_skipped_game_numbers():
        first = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            game_number=1,
        )
        third = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            game_number=3,
            is_series_end=True,
        )
        trainer = _trainer(first, minibatch_size=2)
        batch = next(collate_bc_batches((first, third), 2))

        prepared = trainer._prepare_model_inputs(batch)

        assert not prepared.model_inputs[3][0].any()
        assert prepared.model_inputs[3][1, :SERIES_TOKENS_PER_GAME].all()

    _test_ordered_history_allows_skipped_game_numbers()

    def _test_ordered_history_keeps_interleaved_series_independent():
        first_a = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1], game_number=1)
        first_b = replace(
            first_a,
            series_id="series-2",
        )
        third_a = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            game_number=3,
            is_series_end=True,
        )
        second_b = replace(
            _chunk(
                [int(LabelKind.EXACT)],
                [(7, 8)],
                [0, 1],
                game_number=2,
                is_series_end=True,
            ),
            series_id="series-2",
        )
        trainer = _trainer(first_a, minibatch_size=4)
        batch = next(collate_bc_batches((first_a, first_b, third_a, second_b), 4))

        series_mask = trainer._prepare_model_inputs(batch).model_inputs[3]

        assert not series_mask[:2].any()
        assert series_mask[2:, :SERIES_TOKENS_PER_GAME].all()
        assert not series_mask[2:, SERIES_TOKENS_PER_GAME:].any()

    _test_ordered_history_keeps_interleaved_series_independent()

    def _test_ordered_history_rejects_repeated_or_decreasing_games():
        for game_numbers in [(2, 2), (2, 1)]:
            first = _chunk(
                [int(LabelKind.EXACT)],
                [(7, 8)],
                [0, 1],
                game_number=game_numbers[0],
            )
            invalid = _chunk(
                [int(LabelKind.EXACT)],
                [(7, 8)],
                [0, 1],
                game_number=game_numbers[1],
                is_series_end=True,
            )
            trainer = _trainer(first, minibatch_size=2)
            batch = next(collate_bc_batches((first, invalid), 2))

            with pytest.raises(ValueError, match="must increase"):
                trainer._prepare_model_inputs(batch)

    _test_ordered_history_rejects_repeated_or_decreasing_games()

    def _test_ordered_history_rejects_data_after_series_end():
        ended = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            game_number=1,
            is_series_end=True,
        )
        extra = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            game_number=2,
        )
        trainer = _trainer(ended, minibatch_size=2)
        batch = next(collate_bc_batches((ended, extra), 2))

        with pytest.raises(ValueError, match="after a perspective-series ended"):
            trainer._prepare_model_inputs(batch)

    _test_ordered_history_rejects_data_after_series_end()


def test_ordered_history_tracks_an_incomplete_final_game() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    trainer = _trainer(game, minibatch_size=64)
    first = next(collate_bc_batches((game,), 64))

    prepared = trainer._prepare_model_inputs(first)
    trainer._series_history.apply(prepared.history_updates)

    assert trainer._series_history.has_partial_games


def test_same_batch_next_game_receives_differentiable_series_context() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))

    prepared = trainer._prepare_model_inputs(batch)
    series_mask = prepared.model_inputs[3]

    assert not series_mask[:2].any()
    assert series_mask[2:, :4].all()
    assert not series_mask[2:, 4:].any()
    assert prepared.model_inputs[2].requires_grad


def test_cross_batch_series_history_truncates_encoder_gradients() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=2)
    first_batch, second_batch = collate_bc_batches((first, second), 2)
    first_tokens = torch.randn(2, trainer.policy.d_model, requires_grad=True)

    with patch.object(
        trainer.policy,
        "local_history_tokens",
        return_value=first_tokens,
    ):
        first_prepared = trainer._prepare_model_inputs(first_batch)
    trainer._series_history.apply(first_prepared.history_updates)

    second_prepared = trainer._prepare_model_inputs(second_batch)
    second_prepared.model_inputs[2].sum().backward()

    assert first_tokens.grad is None


def test_bc_loss_trains_series_resampler() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))
    totals: dict[str, float | int] = {
        "loss": 0.0,
        "loss_weight": 0.0,
        "exact_nll": 0.0,
        "partial_nll": 0.0,
        "decisions": 0,
        "labeled_decisions": 0,
        "exact_decisions": 0,
        "partial_decisions": 0,
        "updates": 0,
        "games": 0,
    }

    trainer._backward_chunk(batch, totals)

    gradients = [
        parameter.grad
        for parameter in trainer.policy.series.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_evaluate_encodes_each_collated_batch_once() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 3,
        [(7, 8)] * 3,
        [0, 1, 2, 3],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 3,
        [(7, 8)] * 3,
        [0, 1, 2, 3],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=2)

    with patch.object(trainer.policy, "encode", wraps=trainer.policy.encode) as encode:
        trainer.evaluate((first, second))

    assert encode.call_count == 3


def test_multi_epoch_training_rejects_one_shot_dataset() -> None:
    chunk = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1])
    trainer = _trainer(chunk)
    trainer.dataset = iter((chunk,))
    trainer.config = BCConfig(epochs=2, amp=False)

    with pytest.raises(ValueError, match="re-iterable"):
        trainer.train()


def test_scrape_soft_limit_completes_the_final_linked_series(tmp_path: Path) -> None:
    format_id = FORMAT.bo3_format
    seeds = [f"{format_id}-{number}" for number in (100, 200, 300)]
    siblings = [f"{format_id}-{number}" for number in (101, 201, 301)]
    bodies: dict[str, bytes] = {}
    for seed, sibling in zip(seeds, siblings, strict=True):
        first = _payload(seed, parent=f"series-{seed}")
        first["log"] = f'|uhtml|next|<a href="/battle-{sibling}">Game 2</a>\n{first["log"]}'
        bodies[seed] = json.dumps(first).encode()
        bodies[sibling] = json.dumps(_payload(sibling, parent=f"series-{seed}")).encode()

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(
                200,
                json.dumps([{"id": seed, "formatid": format_id} for seed in seeds]).encode(),
            )
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        page_size=50,
        limit_games=3,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config, transport=transport).acquire()

    assert {entry.replay_id for entry in entries} == {
        seeds[0],
        siblings[0],
        seeds[1],
        siblings[1],
    }


def test_raw_cache_keeps_malformed_bytes_for_later_quality_rejection(
    tmp_path: Path,
) -> None:
    replay_id = f"{FORMAT.bo3_format}-malformed"

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(200, b"not-json")

    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((replay_id,))

    assert (
        load_raw_replay(tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz")
        == b"not-json"
    )


def test_fetcher_skips_404_without_writing_cache_entry(tmp_path: Path, caplog) -> None:
    replay_id = f"{FORMAT.bo3_format}-missing"

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(200, json.dumps([{"id": replay_id}]).encode())
        return HttpResponse(404, b"not found")

    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    with caplog.at_level("WARNING", logger="p0.replays.scrape"):
        assert ReplayFetcher(config, transport=transport).acquire() == ()
    assert not (tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz").exists()
    assert not (tmp_path / FORMAT.bo3_format / "metadata" / f"{replay_id}.json").exists()
    assert "skipping unavailable replay" in caplog.text


def test_cache_build_is_dataset_bound_and_preserves_bo3_series(
    tmp_path: Path,
) -> None:
    good_id = f"{FORMAT.bo3_format}-good"
    bad_id = f"{FORMAT.bo3_format}-bad"
    good = _payload(good_id, parent="source-series")
    bad = _payload(bad_id, parent="source-series")
    bad["log"] = "\n".join(
        line for line in str(bad["log"]).splitlines() if "|showteam|" not in line
    )
    bodies = {good_id: json.dumps(good).encode(), bad_id: json.dumps(bad).encode()}

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    cache = tmp_path / "replays"
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=cache,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((good_id, bad_id))
    from p0.replays.protocol import parse_replay_payload
    from p0.replays.scrape import load_raw_replay

    docs = [
        parse_replay_payload(load_raw_replay(p))
        for p in (cache / FORMAT.bo3_format / "raw").glob("*.json.gz")
    ]
    first = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)
    second = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)

    assert first.manifest_path == second.manifest_path
    assert first.manifest.dataset_hash == second.manifest.dataset_hash
    assert first.manifest.source_games == 2
    assert first.manifest.accepted_games == 1
    assert first.manifest.rejected_games == 1
    chunks = list(LazyReplayDataset(first.manifest_path))
    assert len(chunks) == 2


def test_collator_fills_budget_across_games_and_rebases_candidates() -> None:
    first = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.PARTIAL)],
        [(7, 8), (7, 8), (9, 10)],
        [0, 1, 3],
    )
    second = replace(second, series_id="series-2")

    batches = list(collate_bc_batches((first, second), 4))

    assert [batch.decisions for batch in batches] == [4, 1]
    assert batches[0].games == 2
    assert batches[0].candidate_offsets.tolist() == [0, 0, 1, 2, 3]
    assert batches[1].candidate_offsets.tolist() == [0, 2]


def test_split_assignment_populates_all_requested_splits_when_possible() -> None:
    manifest = assign_series_splits(
        ("a", "b", "c", "d", "e"),
        runtime_contract_sha256="a" * 64,
        dataset_hash="b" * 64,
    )

    assert set(manifest.assignments.values()) == {"train", "validation", "test"}


def test_validation_is_deterministic_inference_only_and_reports_all_counts() -> None:
    game = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT), int(LabelKind.PARTIAL)],
        [(7, 8), (7, 8), (9, 10)],
        [0, 0, 1, 3],
    )
    game = replace(
        game,
        decision_type=torch.tensor((1, 2, 2)),
        label_confidence=torch.tensor((0.1, 0.3, 0.8)),
    )
    policy = build_policy(
        ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    trainer = BCTrainer(
        policy,
        (game,),
        BCConfig(batch_decisions=3, amp=False),
        device="cpu",
    )
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

    first = trainer.evaluate()
    second = trainer.evaluate()

    assert first.to_dict() == second.to_dict()
    assert first.decisions == 3
    assert first.labeled_decisions == 2
    assert first.unknown_decisions == 1
    assert first.exact_decisions == 1
    assert first.partial_decisions == 1
    assert first.non_finite_values == 0
    assert first.illegal_predictions == 0
    assert first.by_decision_type["1"] == {
        "decisions": 1,
        "labeled": 0,
        "nll": 0.0,
    }
    assert first.by_decision_type["2"]["decisions"] == 2
    assert first.by_decision_type["2"]["labeled"] == 2
    assert first.confidence_buckets["[0,.25)"]["decisions"] == 1
    assert first.confidence_buckets["[.25,.5)"]["labeled"] == 1
    assert first.confidence_buckets["[.75,1]"]["labeled"] == 1
    assert first.candidate_set_sizes == {"0": 1, "1": 1, "2": 1}
    assert all(
        torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
    )


def test_bc_training_state_cannot_resume_ppo_but_policy_weights_can_transfer(
    tmp_path: Path,
) -> None:
    policy = build_policy(
        ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    store = CheckpointStore()
    training_path = tmp_path / "bc-training.pt"
    policy_path = tmp_path / "bc-policy.pt"
    store.save_training_state(
        training_path,
        1,
        policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        trainer_kind="bc",
    )
    store.save_policy(policy_path, policy, metadata={"dataset_hash": "a" * 64})
    restored = store.load_policy(policy_path, "cpu")

    with pytest.raises(ValueError, match="trainer"):
        store.load_training_state(
            training_path,
            restored,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
    with pytest.raises(ValueError, match="weights-only"):
        store.load_training_state(
            policy_path,
            restored,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
