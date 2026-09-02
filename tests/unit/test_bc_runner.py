"""Tests for behavioral-cloning training orchestration."""

from __future__ import annotations

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
    TOKEN_IDX_ALLY_SIDE,
    StructuredObservation,
)
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.bc import (
    BCGameWindow,
    BCTrainer,
    collate_bc_batches,
)
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from tests.unit.replay_fixtures import sample_replay_payload


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


class TestBCRunner:
    def test_replay_to_series_bc_checkpoint_smoke(self, tmp_path: Path) -> None:
        """End-to-end smoke test: replay JSON compilation -> tensor shards -> BC training -> checkpoint save/load."""
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

    def test_bc_trainer_updates_policy_in_game_local_chunks(self) -> None:
        """Verify BCTrainer updates model parameters on game chunks and records valid training loss and decision accounting."""
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
        """Verify BC trains the series resampler from a detached prior-game snapshot."""
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
        """Verify UNKNOWN decisions are excluded from labeled metrics while preserving full observation history for subsequent turns."""
        chunk = _chunk(
            [int(LabelKind.UNKNOWN), int(LabelKind.EXACT)],
            [(7, 8)],
            [0, 0, 1],
        )
        metrics = _trainer(chunk).train()
        assert metrics["decisions"] == 2
        assert metrics["games"] == 1

    def test_unknown_only_game_does_not_report_an_optimizer_update(self) -> None:
        """Verify a game consisting exclusively of UNKNOWN decisions performs no parameter updates."""
        chunk = _chunk(
            [int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)],
            [],
            [0, 0, 0],
        )

        metrics = _trainer(chunk, minibatch_size=1).train()

        assert metrics["updates"] == 0
        assert metrics["games"] == 1

    def test_bc_target_windows_keep_only_past_48_local_tokens(self) -> None:
        """Verify collator truncates intra-game memory history to HISTORY_WINDOW=48 past tokens per decision."""
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
        """Verify batches packed with multiple games preserve strict memory isolation across series/game boundaries."""
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
        """Verify is_game_end flag is set on the terminal batch window of a completed game."""
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
        """Verify BCTrainer rejects single-pass generators/iterators when multiple training epochs are requested."""
        chunk = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1])
        trainer = _trainer(chunk)
        trainer.dataset = iter((chunk,))
        trainer.config = BCConfig(epochs=2, enable_optim=False)

        with pytest.raises(ValueError, match="re-iterable"):
            trainer.train()

    def test_collator_fills_budget_across_games_and_rebases_candidates(self) -> None:
        """Verify collate_bc_batches packs multiple games up to minibatch budget and rebases candidate offsets to 0."""
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
        """Verify evaluate() computes illegal probability mass and counts decisions with unproven legality gates."""
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
        """Verify evaluate() is deterministic, does not modify policy parameters, and breaks down metrics by decision type and confidence."""
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
        }
        assert all(
            torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
        )

    def test_bc_training_state_cannot_resume_ppo_but_policy_weights_can_transfer(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify trainer kind safety: BC checkpoint cannot resume PPO training state directly, but policy weights can transfer."""
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
