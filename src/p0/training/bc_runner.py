"""Behavior cloning training and evaluation runner."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from p0.format_config import sha256_file
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.resources import default_runtime_resources
from p0.persistence import atomic_json_save
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest
from p0.training.bc import BCCancelled, BCEvaluationMetrics, BCTrainer
from p0.training.checkpoint import CheckpointStore, LoadedCheckpoint
from p0.training.config import BCConfig
from p0.training.files import TrainingRun, training_run
from p0.training.utils import default_device, seed_everything


def _training_metadata(
    config: BCConfig,
    *,
    dataset_hash: str,
    split_manifest: Path,
    overfit: bool,
) -> dict[str, Any]:
    trainer_config = asdict(config)
    for name in ("epochs", "shard_manifest", "split_manifest", "output_dir", "resume_checkpoint"):
        del trainer_config[name]
    trainer_config["overfit"] = overfit
    return {
        "dataset_hash": dataset_hash,
        "split_manifest_sha256": sha256_file(split_manifest),
        "trainer_config": trainer_config,
        "epoch_budget": config.epochs,
        "gamma": config.gamma,
        "value_target_semantics": "discounted_terminal_outcome.v1",
    }


def _validation_is_failed(
    metrics: BCEvaluationMetrics,
    *,
    require_policy_support: bool = False,
) -> bool:
    return (
        metrics.non_finite_values > 0
        or (require_policy_support and metrics.labeled_count == 0)
        or not all(math.isfinite(value) for value in metrics.to_dict().values())
    )


_BC_TRAIN_BOARD_METRICS = ("overall_nll", "grad_norm", "learning_rate")


def _has_accepted_series(
    split_manifest: SeriesSplitManifest,
    accepted_series: frozenset[str],
    split: str,
) -> bool:
    return any(
        assigned_split == split and series_id in accepted_series
        for series_id, assigned_split in split_manifest.assignments.items()
    )


def _restore_best_policy(
    files: TrainingRun,
    selection: Mapping[str, Any],
    step: int,
    expected_metadata: Mapping[str, Any],
) -> None:
    """Restore the embedded best policy from the resume checkpoint."""
    score, epoch = selection.get("best_validation_nll"), selection.get("selected_epoch")
    if (
        not isinstance(score, (float, int))
        or not math.isfinite(score)
        or type(epoch) is not int
        or epoch <= 0
    ):
        raise ValueError("BC resume checkpoint has incomplete selected-policy state")
    if epoch > step:
        raise ValueError("BC resume checkpoint has stale selected-policy state")
    best = files.state.get("best_policy")
    if best is None:
        raise ValueError("BC resume checkpoint has no embedded selected policy")
    source = files.source
    assert source is not None
    files.store.validate_artifact(best, source.path)
    files.store.load_policy(
        LoadedCheckpoint(source.path, best, ""),
        "cpu",
        expected_metadata={**expected_metadata, "selected_epoch": epoch},
    )


def _overfit_succeeded(metrics: BCEvaluationMetrics | None, initial_nll: float) -> bool:
    return bool(
        metrics is not None
        and metrics.overall_nll <= initial_nll * 0.2
        and metrics.exact_joint_accuracy >= 0.9
    )


def _reported_path(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return str(path.resolve())


def train_bc(
    config: BCConfig,
    *,
    overfit: bool = False,
    device: torch.device | str | None = None,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """
    Train one epoch at a time, validate, and checkpoint completed epochs.

    Arguments:
        config: Behavior-cloning dataset, optimizer, and output configuration.
        overfit: Use the bounded overfit mode intended for data-pipeline checks.
        device: Device override; the runtime default is used when omitted.
        cancel_requested: Callback polled between training units.

    Returns:
        The final training and validation metrics plus selected checkpoint state.
    """
    store = CheckpointStore()
    dataset = LazyReplayDataset(
        config.shard_manifest, split_manifest=config.split_manifest, verify_hashes=True
    )
    shard_manifest, split_manifest = dataset.manifest, dataset.split_manifest
    assert split_manifest is not None
    accepted_series = frozenset(dataset.accepted_series_ids())
    for split in ("train", "validation"):
        if not _has_accepted_series(split_manifest, accepted_series, split):
            raise ValueError(f"BC {split} split has no accepted series")
    train_dataset = dataset.for_split("train")
    validation_dataset = dataset.for_split("validation")
    dataset_output = config.output_dir / shard_manifest.dataset_hash
    latest_path = dataset_output / "bc_latest_training.pt"
    best_path = dataset_output / "bc_best_policy.pt"
    metrics_path = dataset_output / "metrics.json"
    metadata = _training_metadata(
        config,
        dataset_hash=shard_manifest.dataset_hash,
        split_manifest=config.split_manifest,
        overfit=overfit,
    )
    with training_run(
        store,
        latest_path,
        dataset_output,
        trainer_kind="bc",
        settings=metadata["trainer_config"],
        source_path=config.resume_checkpoint,
        resume=config.resume_checkpoint is not None,
    ) as files:
        files.metadata["inputs"] = {
            "shard_manifest": str(config.shard_manifest.resolve()),
            "split_manifest": str(config.split_manifest.resolve()),
        }
        selected_device = default_device() if device is None else torch.device(device)
        seed_everything(config.seed)
        policy = (
            store.load_policy(
                files.source,
                selected_device,
                expected_metadata={
                    "gamma": config.gamma,
                    "value_target_semantics": "discounted_terminal_outcome.v1",
                },
            )
            if files.source is not None
            else build_policy(ModelConfig.baseline(), default_runtime_resources())
        )
        trainer = BCTrainer(
            policy,
            train_dataset,
            config,
            device=selected_device,
            cancel_requested=cancel_requested,
        )
        if files.source is not None:
            resume_metadata = {
                key: value for key, value in metadata.items() if key != "epoch_budget"
            }
            selection_state = dict(store.load_metadata(files.source).get("selection_state", {}))
            completed_epoch = store.load_episode(files.source)
            _restore_best_policy(files, selection_state, completed_epoch, resume_metadata)
            # Restore randomness after constructing the selected-policy validation model.
            store.load_training(
                files.source,
                trainer.policy,
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
                expected_trainer_kind="bc",
                expected_metadata=resume_metadata,
                require_training_state=True,
            )
            latest_artifact: Path | None = files.source.path
        else:
            completed_epoch = 0
            selection_state = {}
            latest_artifact = None
        policy = compile_policy(
            policy,
            enable=config.enable_optim and selected_device.type == "cuda",
        )
        initial_training = trainer.evaluate(train_dataset)
        if _validation_is_failed(initial_training, require_policy_support=True):
            raise RuntimeError(
                "Initial BC training evaluation contains invalid predictions or values"
            )
        if overfit and initial_training.exact_count == 0:
            raise RuntimeError("BC overfit acceptance requires at least one exact policy label")
        saved_initial_nll = selection_state.get("overfit_initial_nll")
        initial_nll = (
            float(saved_initial_nll)
            if overfit
            and isinstance(saved_initial_nll, (int, float))
            and math.isfinite(float(saved_initial_nll))
            else initial_training.overall_nll
        )
        final_training_evaluation: BCEvaluationMetrics | None = (
            initial_training if overfit else None
        )
        last_training_update: dict[str, float | int] | None = None
        final_validation: BCEvaluationMetrics | None = None
        files.start(completed_epoch)
        cancelled = False
        for epoch in range(completed_epoch + 1, config.epochs + 1):
            if cancel_requested():
                cancelled = True
                break
            try:
                training = trainer.train_epoch()
            except BCCancelled:
                cancelled = True
                break
            if training["updates"] == 0:
                raise RuntimeError(f"BC training made no successful updates at epoch {epoch}")
            validation = trainer.evaluate(validation_dataset)
            if _validation_is_failed(validation, require_policy_support=True):
                raise RuntimeError(f"BC validation failed at epoch {epoch}")
            if overfit:
                final_training_evaluation = trainer.evaluate(train_dataset)
                if _validation_is_failed(final_training_evaluation):
                    raise RuntimeError(f"BC training evaluation failed at epoch {epoch}")
            last_training_update = training
            final_validation = validation
            if validation.overall_nll < selection_state.get("best_validation_nll", float("inf")):
                files.state["best_policy"] = store.snapshot_policy(
                    trainer.policy,
                    {
                        **metadata,
                        "trainer_kind": "bc",
                        "selected_epoch": epoch,
                        "run": files.metadata,
                    },
                )
                selection_state = {
                    "best_validation_nll": validation.overall_nll,
                    "selected_epoch": epoch,
                }
            selection_state["overfit_initial_nll"] = initial_nll if overfit else None
            validation_values = validation.to_dict()
            record = {
                "epoch": epoch,
                "dataset_hash": shard_manifest.dataset_hash,
                "train_update": training,
                "training": (
                    training
                    if final_training_evaluation is None
                    else final_training_evaluation.to_dict()
                ),
                "validation": validation_values,
            }
            files.record(
                epoch,
                record,
                {
                    "train": {name: training[name] for name in _BC_TRAIN_BOARD_METRICS},
                    "validation": validation_values,
                },
            )
            files.save(
                epoch,
                trainer.policy,
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
                metadata={**metadata, "selection_state": selection_state},
            )
            latest_artifact = latest_path.resolve()
            completed_epoch = epoch
            if overfit and _overfit_succeeded(final_training_evaluation, initial_nll):
                break
        overfit_passed = (
            _overfit_succeeded(final_training_evaluation, initial_nll) if overfit else None
        )
        result = {
            "dataset_hash": shard_manifest.dataset_hash,
            "global_hash": shard_manifest.global_contract_sha256,
            "completed_epoch": completed_epoch,
            "cancelled": cancelled,
            "overfit_passed": overfit_passed,
            "initial_training": initial_training.to_dict(),
            "final_training": (
                last_training_update
                if final_training_evaluation is None
                else final_training_evaluation.to_dict()
            ),
            "final_validation": (None if final_validation is None else final_validation.to_dict()),
            "latest_training_checkpoint": _reported_path(latest_artifact),
            "best_policy_checkpoint": _reported_path(best_path),
            "metrics_path": _reported_path(metrics_path),
        }
        atomic_json_save(dataset_output / "bc-result.json", result)
        if overfit and not overfit_passed and not cancelled:
            raise RuntimeError(
                "BC overfit acceptance failed: training NLL did not fall by 80% "
                "with at least 90% exact accuracy"
            )
        return result


@torch.inference_mode()
def evaluate_bc(
    config: BCConfig,
    checkpoint: Path,
    *,
    split: str = "validation",
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Evaluate a weights-only or BC training checkpoint on one bound split."""
    selected_device = default_device() if device is None else torch.device(device)
    store = CheckpointStore()
    objective = {
        "gamma": config.gamma,
        "value_target_semantics": "discounted_terminal_outcome.v1",
    }
    policy = store.load_policy(
        checkpoint,
        selected_device,
        expected_metadata=objective,
    )
    dataset = LazyReplayDataset(
        config.shard_manifest, split_manifest=config.split_manifest, verify_hashes=True
    )
    shard_manifest, split_manifest = dataset.manifest, dataset.split_manifest
    assert split_manifest is not None
    accepted_series = frozenset(dataset.accepted_series_ids())
    dataset = dataset.for_split(split)
    if not _has_accepted_series(split_manifest, accepted_series, split):
        raise ValueError(f"BC {split} split has no accepted series")
    trainer = BCTrainer(policy, dataset, config, device=selected_device)
    metrics = trainer.evaluate()
    if _validation_is_failed(metrics, require_policy_support=True):
        raise RuntimeError("BC evaluation contains invalid predictions or non-finite values")
    return {
        "dataset_hash": shard_manifest.dataset_hash,
        "global_hash": shard_manifest.global_contract_sha256,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "objective": objective,
        "metrics": metrics.to_dict(),
    }
