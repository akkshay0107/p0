"""Tests for behavior-cloning objectives, batching, and training."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_TEAM_PREVIEW,
    TOKEN_IDX_ALLY_SIDE,
    TOKEN_IDX_GLOBAL_FIELD,
    StructuredObservation,
)
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.bc import (
    BCGameWindow,
    BCTrainer,
    collate_bc_batches,
    compute_bc_objective,
)
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from tests.unit.replay_fixtures import sample_replay_payload


class TestBCObjectives:
    def test_exact_and_partial_losses_match_probability_definitions(self) -> None:
        """Check exact, candidate-marginal, and unknown-label loss definitions."""
        log_probs = torch.tensor([math.log(0.25), math.log(0.5), math.log(0.25)])
        offsets = torch.tensor([0, 1, 3, 3], dtype=torch.long)
        labels = torch.tensor(
            [int(LabelKind.EXACT), int(LabelKind.PARTIAL), int(LabelKind.UNKNOWN)]
        )
        loss_mask = torch.tensor([1.0, 1.0, 0.0])

        result = compute_bc_objective(log_probs, offsets, labels, loss_mask)

        expected_exact = -math.log(0.25)
        # Partial candidate set contains probs 0.5 and 0.25 -> sum = 0.75
        expected_partial = -math.log(0.75)
        assert result.exact_count == 1 and result.partial_count == 1
        assert result.labeled_count == 2
        assert result.loss_weight == 2.0
        assert result.exact_nll.item() == pytest.approx(expected_exact)
        assert result.partial_nll.item() == pytest.approx(expected_partial)
        assert result.loss.item() == pytest.approx((expected_exact + expected_partial) / 2)
        assert result.marginal_log_probs[2].isneginf()

    def test_fractional_loss_weights_do_not_change_labeled_counts(self) -> None:
        """Verify fractional sample weights scale total loss without distorting discrete labeled count metrics."""
        result = compute_bc_objective(
            torch.log(torch.tensor([0.25, 0.75])),
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([int(LabelKind.EXACT), int(LabelKind.EXACT)]),
            torch.tensor([0.25, 0.75]),
        )

        assert result.labeled_count == 2
        assert result.loss_weight == 1.0
        assert result.loss.item() == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))

    def test_fractional_weights_use_the_true_weighted_mean(self) -> None:
        """Verify fractional sample weights compute the mathematically exact weighted mean and gradients."""
        log_probs = torch.log(torch.tensor([0.25, 0.5])).requires_grad_()

        result = compute_bc_objective(
            log_probs,
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([int(LabelKind.EXACT), int(LabelKind.EXACT)]),
            torch.tensor([0.1, 0.15]),
        )
        result.loss.backward()

        expected = (-0.1 * math.log(0.25) - 0.15 * math.log(0.5)) / 0.25
        assert result.loss_weight.item() == pytest.approx(0.25)
        assert result.loss.item() == pytest.approx(expected)
        torch.testing.assert_close(log_probs.grad, torch.tensor([-0.4, -0.6]))

    def test_unknown_steps_have_zero_loss_and_preserve_boundaries(self) -> None:
        """Verify UNKNOWN labels produce zero loss and empty gradients without breaking backprop graph."""
        log_probs = torch.empty(0, requires_grad=True)
        offsets = torch.tensor([0, 0, 0], dtype=torch.long)
        labels = torch.tensor([int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)])
        loss_mask = torch.zeros(2)

        result = compute_bc_objective(log_probs, offsets, labels, loss_mask)
        assert result.loss.item() == 0.0
        result.loss.backward()
        assert log_probs.grad is not None and log_probs.grad.numel() == 0

    def test_partial_loss_is_candidate_order_invariant(self) -> None:
        """Verify marginal log-sum-exp over candidate actions is invariant to internal candidate permutation."""
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

    def test_partial_loss_allows_an_impossible_candidate_when_total_mass_is_valid(self) -> None:
        result = compute_bc_objective(
            torch.tensor([math.log(0.5), float("-inf")]),
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([int(LabelKind.PARTIAL)], dtype=torch.long),
            torch.ones(1),
        )

        assert result.loss.item() == pytest.approx(math.log(2.0))

    def test_candidate_objective_preserves_gradients(self) -> None:
        """Verify backward gradient flow through marginal candidate loss calculations."""
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
        ("log_probs", "offsets", "labels", "message"),
        [
            ([float("nan")], [0, 1], [int(LabelKind.EXACT)], "valid log probabilities"),
            ([float("-inf")], [0, 1], [int(LabelKind.EXACT)], "positive finite"),
            (
                [math.log(0.75), math.log(0.75)],
                [0, 2],
                [int(LabelKind.PARTIAL)],
                "must not exceed one",
            ),
        ],
    )
    def test_invalid_candidate_probability_mass_is_rejected(
        self,
        log_probs: list[float],
        offsets: list[int],
        labels: list[int],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            compute_bc_objective(
                torch.tensor(log_probs),
                torch.tensor(offsets, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.ones(len(labels)),
            )

    def test_nonfinite_loss_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compute_bc_objective(
                torch.tensor([math.log(0.5)]),
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([int(LabelKind.EXACT)], dtype=torch.long),
                torch.tensor([float("nan")]),
            )

    @pytest.mark.parametrize(
        ("labels", "offsets", "mask", "message"),
        [
            ([int(LabelKind.EXACT)], [0, 2], [1.0], "EXACT"),
            ([int(LabelKind.PARTIAL)], [0, 1], [1.0], "PARTIAL"),
            ([int(LabelKind.UNKNOWN)], [0, 1], [0.0], "UNKNOWN"),
            ([99], [0, 0], [0.0], "unsupported"),
        ],
    )
    def test_invalid_label_and_candidate_shapes_are_rejected(
        self, labels, offsets, mask, message
    ) -> None:
        """Verify compute_bc_objective detects candidate count and label type mismatches."""
        with pytest.raises(ValueError, match=message):
            compute_bc_objective(
                torch.full((offsets[-1],), math.log(0.5)),
                torch.tensor(offsets, dtype=torch.long),
                torch.tensor(labels),
                torch.tensor(mask),
            )


def _chunk(
    label_kind: list[int],
    candidate_values: list[tuple[int, int]],
    offsets: list[int],
    *,
    game_number: int = 1,
    is_series_end: bool = False,
) -> ReplayGameChunk:
    """Helper creating a synthetic ReplayGameChunk for BC trainer unit tests."""
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
    """Helper building a minimal BCTrainer instance over a single game chunk."""
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
            enable_optim=False,
        ),
        device="cpu",
    )


class TestBCTrainer:
    def test_complete_game_reaches_optimizer_before_next_game_is_requested(self) -> None:
        first = _chunk(
            [int(LabelKind.EXACT)] * 5,
            [(7, 8)] * 5,
            list(range(6)),
            is_series_end=True,
        )
        second = replace(
            _chunk(
                [int(LabelKind.EXACT)] * 4,
                [(7, 8)] * 4,
                list(range(5)),
                is_series_end=True,
            ),
            series_id="series-2",
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}

        def games():
            yield first
            assert any(
                not torch.equal(before[name], value) for name, value in policy.named_parameters()
            )
            yield second

        trainer = BCTrainer(
            policy,
            games(),
            BCConfig(
                batch_decisions=4,
                max_chunk_size=4,
                learning_rate=1e-3,
                enable_optim=False,
            ),
            device="cpu",
        )

        metrics = trainer.train_epoch()

        assert metrics["decisions"] == 9
        assert metrics["games"] == 2
        assert metrics["updates"] == 2

    def test_multiworker_training_rejects_an_unsharded_iterable(self) -> None:
        game = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1], is_series_end=True)
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=1, num_workers=2, enable_optim=False),
            device="cpu",
        )

        with pytest.raises(ValueError, match="worker-sharded LazyReplayDataset"):
            trainer.train_epoch()

    def test_preview_candidate_orbits_are_unique_and_accuracy_ignores_pair_order(self) -> None:
        game = _chunk([int(LabelKind.EXACT)], [(6, 20)], [0, 1], is_series_end=True)
        game.observations.numerical[:, TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] = 1.0
        valid_pairs = torch.tensor([first != second for first in range(6) for second in range(6)])
        action_mask = torch.zeros((1, 2, FORMAT.action_size), dtype=torch.bool)
        action_mask[:, :, :36] = valid_pairs
        game = replace(
            game,
            action_mask=action_mask,
            exact_action=torch.tensor(((6, 20),), dtype=torch.long),
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=1, enable_optim=False),
            device="cpu",
        )

        metrics = trainer.evaluate()

        assert metrics.exact_nll == pytest.approx(math.log(90.0), rel=1e-5)
        assert metrics.exact_joint_accuracy == 1.0

    def test_real_shards_close_all_worker_game_boundaries(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (
                sample_replay_payload("worker-a", parent="worker-series-a"),
                sample_replay_payload("worker-b", parent="worker-series-b"),
            )
        )
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            max_decisions_per_shard=1,
            created_at="2026-01-01T00:00:00Z",
        )
        dataset = LazyReplayDataset(built.manifest_path)
        policy = build_policy(
            ModelConfig(64, 4, 1, 128),
            default_runtime_resources(),
        )
        trainer = BCTrainer(
            policy,
            dataset,
            BCConfig(
                batch_decisions=3,
                max_chunk_size=3,
                num_workers=2,
                learning_rate=1e-3,
                enable_optim=False,
            ),
            device="cpu",
        )

        metrics = trainer.train_epoch()

        assert metrics["decisions"] > 0
        assert metrics["games"] == 4

    def test_replay_to_series_bc_checkpoint_smoke(self, tmp_path: Path) -> None:
        result = compile_payloads(
            (sample_replay_payload("game-1"), sample_replay_payload("game-2"))
        )
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
                enable_optim=False,
            ),
            device="cpu",
        )

        metrics = trainer.train()

        assert metrics["decisions"] == 8
        assert metrics["games"] == 4
        assert torch.isfinite(torch.tensor(metrics["overall_nll"]))
        checkpoint = tmp_path / "bc.pt"
        store = CheckpointStore()
        store.save_training(
            checkpoint,
            1,
            trainer.policy,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
            trainer_kind="bc",
        )

        restored = build_policy(trainer.policy.config, default_runtime_resources())
        restored_trainer = BCTrainer(
            restored,
            (),
            trainer.config,
            device="cpu",
        )
        assert (
            store.load_training(
                checkpoint,
                restored_trainer.policy,
                optimizer=restored_trainer.optimizer,
                scaler=restored_trainer.scaler,
                expected_trainer_kind="bc",
                require_training_state=True,
            )
            == 1
        )
        for name, parameter in trainer.policy.state_dict().items():
            torch.testing.assert_close(parameter, restored_trainer.policy.state_dict()[name])

    def test_bc_trainer_updates_policy_in_game_local_chunks(self) -> None:
        chunk = _chunk(
            [int(LabelKind.EXACT), int(LabelKind.EXACT)],
            [(7, 8), (7, 8)],
            [0, 1, 2],
        )
        trainer = _trainer(chunk, minibatch_size=1)
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.policy.named_parameters()
        }

        metrics = trainer.train()

        assert metrics["decisions"] == 2
        assert metrics["updates"] == 1
        assert metrics["games"] == 1
        assert metrics["grad_norm"] >= 0.0
        assert torch.isfinite(torch.tensor(metrics["overall_nll"]))
        # Verify policy parameters were modified by optimizer step
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in trainer.policy.named_parameters()
        )

    def test_bc_trainer_resamples_prior_game_history_with_live_gradients(self) -> None:
        first = _chunk(
            [int(LabelKind.EXACT), int(LabelKind.EXACT)],
            [(7, 8), (7, 8)],
            [0, 1, 2],
        )
        second = _chunk(
            [int(LabelKind.EXACT), int(LabelKind.EXACT)],
            [(7, 8), (7, 8)],
            [0, 1, 2],
            game_number=2,
            is_series_end=True,
        )
        policy = build_policy(
            ModelConfig(64, 4, 1, 128),
            default_runtime_resources(),
        )
        trainer = BCTrainer(
            policy,
            (first, second),
            BCConfig(batch_decisions=2, learning_rate=1e-3, enable_optim=False),
            device="cpu",
        )
        before = {
            name: parameter.detach().clone() for name, parameter in policy.series.named_parameters()
        }

        metrics = trainer.train()

        assert metrics["games"] == 2
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in policy.series.named_parameters()
        )

    def test_unknown_decision_is_excluded_without_breaking_game_context(self) -> None:
        chunk = _chunk(
            [int(LabelKind.UNKNOWN), int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 0, 1],
        )
        metrics = _trainer(chunk).train()
        assert metrics["decisions"] == 2
        assert metrics["games"] == 1

    def test_unknown_only_game_does_not_report_an_optimizer_update(self) -> None:
        chunk = _chunk(
            [int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)],
            [],
            [0, 0, 0],
        )

        metrics = _trainer(chunk, minibatch_size=1).train()

        assert metrics["updates"] == 0
        assert metrics["games"] == 1

    def test_unknown_policy_labels_with_known_outcome_train_the_value_head(self) -> None:
        game = replace(
            _chunk(
                [int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)],
                [],
                [0, 0, 0],
                is_series_end=True,
            ),
            outcome_valid=True,
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=2, learning_rate=1e-3, enable_optim=False),
            device="cpu",
        )
        before = {
            name: parameter.detach().clone() for name, parameter in policy.critic.named_parameters()
        }

        metrics = trainer.train_epoch()

        assert metrics["updates"] == 1
        assert metrics["overall_nll"] == 0.0
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in policy.critic.named_parameters()
        )

    @pytest.mark.parametrize(
        ("gamma", "expected_loss"),
        [
            (0.5, (0.25**2 + 0.5**2 + 1.0) / 3.0),
            (0.0, 1.0 / 3.0),
        ],
    )
    def test_value_targets_are_literal_discounted_recorded_decisions(
        self,
        gamma: float,
        expected_loss: float,
    ) -> None:
        game = replace(
            _chunk(
                [int(LabelKind.EXACT)] * 3,
                [(7, 8)] * 3,
                [0, 1, 2, 3],
                is_series_end=True,
            ),
            outcome_valid=True,
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=3, gamma=gamma, enable_optim=False),
            device="cpu",
        )

        metrics = trainer.evaluate()

        assert metrics.value_count == 3
        assert metrics.value_loss == pytest.approx(expected_loss)

    def test_nonfinite_update_preserves_parameters_and_optimizer_state(self) -> None:
        invalid = _chunk(
            [int(LabelKind.EXACT)],
            [(1, 2)],
            [0, 1],
            is_series_end=True,
        )
        valid = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            is_series_end=True,
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        trainer = BCTrainer(
            policy,
            (invalid,),
            BCConfig(batch_decisions=1, learning_rate=1e-3, enable_optim=False),
            device="cpu",
        )
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}

        with pytest.raises(FloatingPointError, match="evaluation contains non-finite"):
            trainer.evaluate()
        with pytest.raises(FloatingPointError, match="non-finite loss"):
            trainer.train_epoch()

        assert not trainer.optimizer.state
        assert all(parameter.grad is None for parameter in policy.parameters())
        assert all(
            torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
        )

        trainer.dataset = (valid,)
        metrics = trainer.train_epoch()
        assert metrics["updates"] == 1

    def test_nonfinite_gradient_discards_the_complete_update(self) -> None:
        game = _chunk(
            [int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 1],
            is_series_end=True,
        )
        policy = build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=1, learning_rate=1e-3, enable_optim=False),
            device="cpu",
        )
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}
        handle = policy.actor.q_proj1.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )

        with pytest.raises(FloatingPointError, match="non-finite gradients"):
            trainer.train_epoch()
        handle.remove()

        assert not trainer.optimizer.state
        assert all(parameter.grad is None for parameter in policy.parameters())
        assert all(
            torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
        )
        assert trainer.train_epoch()["updates"] == 1

    def test_bc_target_windows_keep_only_past_48_local_tokens(self) -> None:
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

    def test_collated_context_is_compact_and_never_crosses_game_boundaries(self) -> None:
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
        # Decision 2 (start of second game) must not attend to history from first game
        assert not torch.any(batch.history_mask[2])
        assert batch.history_indices[3, -1] == 2
        assert batch.history_mask[3, -1]
        for tensor in batch.observations.tensors():
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()

    def test_completed_game_sets_terminal_window_flag(self) -> None:
        length = 100
        game = _chunk(
            [int(LabelKind.EXACT)] * length,
            [(7, 8)] * length,
            list(range(length + 1)),
        )

        first, second = collate_bc_batches((game,), 64)

        assert not first.windows[0].is_game_end
        assert second.windows[0].is_game_end
        assert second.observations.categorical.size(0) == HISTORY_WINDOW + length - 64

    def test_multi_epoch_training_rejects_one_shot_dataset(self) -> None:
        chunk = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1])
        trainer = _trainer(chunk)
        trainer.dataset = iter((chunk,))
        trainer.config = BCConfig(epochs=2, enable_optim=False)

        with pytest.raises(ValueError, match="re-iterable"):
            trainer.train()

    def test_collator_fills_budget_across_games_and_rebases_candidates(self) -> None:
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

    def test_evaluation_reports_legality_diagnostics(self) -> None:
        game = _chunk([int(LabelKind.EXACT)] * 2, [(7, 8), (7, 8)], [0, 1, 2])
        gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
        game.observations.numerical[1, TOKEN_IDX_ALLY_SIDE, gates] = 1.0
        policy = build_policy(
            ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
            default_runtime_resources(),
        )
        trainer = BCTrainer(
            policy,
            (game,),
            BCConfig(batch_decisions=2, enable_optim=False),
            device="cpu",
        )

        metrics = trainer.evaluate()

        assert 0.0 < metrics.illegal_probability_mass < 1.0
        assert metrics.to_dict()["illegal_probability_mass"] == metrics.illegal_probability_mass

    def test_validation_is_deterministic_inference_only_and_reports_all_counts(self) -> None:
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
            BCConfig(batch_decisions=3, enable_optim=False),
            device="cpu",
        )
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

        first = trainer.evaluate()
        second = trainer.evaluate()

        assert first.to_dict() == second.to_dict()
        assert first.unknown_label_fraction == pytest.approx(1 / 3)
        assert first.non_finite_values == 0
        assert set(first.to_dict()) == {
            "overall_nll",
            "exact_nll",
            "partial_nll",
            "exact_joint_accuracy",
            "value_loss",
            "illegal_probability_mass",
            "unknown_label_fraction",
            "non_finite_values",
            "decisions",
            "labeled_count",
            "exact_count",
            "partial_count",
            "value_count",
        }
        assert all(
            torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
        )

    def test_bc_training_state_cannot_resume_ppo_but_policy_weights_can_transfer(
        self,
        tmp_path: Path,
    ) -> None:
        policy = build_policy(
            ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
            default_runtime_resources(),
        )
        store = CheckpointStore()
        training_path = tmp_path / "bc-training.pt"
        policy_path = tmp_path / "bc-policy.pt"
        store.save_training(
            training_path,
            1,
            policy,
            optimizer=torch.optim.AdamW(policy.parameters()),
            trainer_kind="bc",
        )
        store.save_policy(policy_path, policy, metadata={"dataset_hash": "a" * 64})
        restored = store.load_policy(policy_path, "cpu")

        with pytest.raises(ValueError, match="trainer"):
            store.load_training(
                training_path,
                restored,
                expected_trainer_kind="ppo",
                require_training_state=True,
            )
        with pytest.raises(ValueError, match="weights-only"):
            store.load_training(
                policy_path,
                restored,
                expected_trainer_kind="ppo",
                require_training_state=True,
            )
