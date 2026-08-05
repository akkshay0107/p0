"""Composition root for streaming replay behaviour cloning and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.resources import default_runtime_resources
from p0.persistence import atomic_json_save
from p0.replays.dataset import LazyReplayDataset, load_split_manifest
from p0.replays.shards import load_shard_manifest
from p0.training.bc import BCCancelled, BCEvaluationMetrics, BCTrainer
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from p0.training.utils import default_device, seed_everything


def _trainer_config(config: BCConfig, *, overfit: bool) -> dict[str, Any]:
    return {
        "batch_decisions": config.batch_decisions,
        "max_chunk_size": config.max_chunk_size,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "value_coef": config.value_coef,
        "epochs": 200 if overfit else config.epochs,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "amp": config.amp,
        "overfit": overfit,
    }


def _provenance(
    config: BCConfig,
    *,
    dataset_hash: str,
    split_manifest: Path,
    overfit: bool,
) -> dict[str, Any]:
    return {
        "dataset_hash": dataset_hash,
        "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "trainer_config": _trainer_config(config, overfit=overfit),
        "gamma": config.gamma,
        "value_target_semantics": "discounted_terminal_outcome.v1",
    }


def _validation_is_failed(metrics: BCEvaluationMetrics) -> bool:
    return (
        metrics.non_finite_values > 0
        or metrics.illegal_predictions > 0
        or not all(
            math.isfinite(value)
            for value in (
                metrics.overall_nll,
                metrics.exact_nll,
                metrics.partial_nll,
                metrics.exact_joint_accuracy,
                metrics.value_loss,
            )
        )
    )


def _flatten_metrics(
    prefix: str,
    values: Mapping[str, object],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(key, value))
        elif isinstance(value, (int, float)):
            flattened[key] = float(value)
    return flattened


def _append_metrics(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _load_identities(
    shard_manifest_path: Path,
    split_manifest_path: Path,
) -> tuple[Any, Any]:
    shard_value = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    shard_manifest = load_shard_manifest(shard_value)
    split_manifest = load_split_manifest(split_manifest_path)
    if split_manifest.runtime_contract_sha256 != shard_manifest.runtime_contract_sha256:
        raise ValueError("BC shard and split manifests reference different runtime contracts")
    if split_manifest.dataset_hash != shard_manifest.dataset_hash:
        raise ValueError("BC shard and split manifests reference different datasets")
    return shard_manifest, split_manifest


def train_bc(
    config: BCConfig,
    *,
    overfit: bool = False,
    device: torch.device | str | None = None,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Train one epoch at a time, validate, and checkpoint completed epochs.

    Arguments:
        config: Behaviour-cloning dataset, optimizer, and output configuration.
        overfit: Use the bounded overfit mode intended for data-pipeline checks.
        device: Device override; the runtime default is used when omitted.
        cancel_requested: Callback polled between training units.

    Returns:
        The final training and validation metrics plus selected checkpoint state.
    """
    shard_manifest, _ = _load_identities(config.shard_manifest, config.split_manifest)
    train_dataset = LazyReplayDataset(
        config.shard_manifest,
        split="train",
        split_manifest=config.split_manifest,
    )
    validation_dataset = LazyReplayDataset(
        config.shard_manifest,
        split="validation",
        split_manifest=config.split_manifest,
    )
    selected_device = default_device() if device is None else torch.device(device)
    seed_everything(config.seed)
    store = CheckpointStore()
    if config.resume_checkpoint is None:
        policy = build_policy(ModelConfig.baseline(), default_runtime_resources())
    else:
        policy = store.load_policy(
            config.resume_checkpoint,
            selected_device,
            expected_metadata={
                "gamma": config.gamma,
                "value_target_semantics": "discounted_terminal_outcome.v1",
            },
        )

    dataset_output = config.output_dir / shard_manifest.dataset_hash
    dataset_output.mkdir(parents=True, exist_ok=True)
    latest_path = dataset_output / "bc_latest_training.pt"
    best_path = dataset_output / "bc_best_policy.pt"
    metrics_path = dataset_output / "metrics.jsonl"
    provenance = _provenance(
        config,
        dataset_hash=shard_manifest.dataset_hash,
        split_manifest=config.split_manifest,
        overfit=overfit,
    )
    trainer = BCTrainer(
        policy,
        train_dataset,
        config,
        device=selected_device,
        checkpoint_store=store,
        provenance=provenance,
        cancel_requested=cancel_requested,
    )
    if config.resume_checkpoint is not None:
        completed_epoch = trainer.load_checkpoint(config.resume_checkpoint)
        selection_state = trainer.load_selection_state(config.resume_checkpoint)
        raw_best = selection_state.get("best_validation_nll")
        raw_epoch = selection_state.get("selected_epoch")
        best_validation_nll = (
            float(raw_best) if isinstance(raw_best, (int, float)) else float("inf")
        )
        best_selected_epoch = int(raw_epoch) if isinstance(raw_epoch, int) else 0
    else:
        completed_epoch = 0
        selection_state = {}
        best_validation_nll = float("inf")
        best_selected_epoch = 0
    policy = compile_policy(policy, enable=selected_device.type == "cuda")
    initial_training = trainer.evaluate(train_dataset)
    if _validation_is_failed(initial_training):
        raise RuntimeError("Initial BC training evaluation contains invalid predictions or values")
    initial_nll = initial_training.overall_nll
    final_training = initial_training
    final_validation: BCEvaluationMetrics | None = None
    max_epochs = 200 if overfit else config.epochs
    writer = SummaryWriter(log_dir=str(dataset_output / "tensorboard"))
    cancelled = False
    try:
        for epoch in range(completed_epoch + 1, max_epochs + 1):
            if cancel_requested():
                cancelled = True
                break
            try:
                training = trainer.train_epoch()
            except BCCancelled:
                cancelled = True
                break
            validation = trainer.evaluate(validation_dataset)
            if _validation_is_failed(validation):
                raise RuntimeError(f"BC validation failed at epoch {epoch}")
            if overfit:
                final_training = trainer.evaluate(train_dataset)
                if _validation_is_failed(final_training):
                    raise RuntimeError(f"BC training evaluation failed at epoch {epoch}")
            else:
                final_training = BCEvaluationMetrics(
                    overall_nll=float(training["loss"]),
                    exact_nll=float(training["exact_nll"]),
                    partial_nll=float(training["partial_nll"]),
                    exact_joint_accuracy=0.0,
                    decisions=int(training["decisions"]),
                    labeled_decisions=int(training["labeled_decisions"]),
                    unknown_decisions=int(training["decisions"])
                    - int(training["labeled_decisions"]),
                    exact_decisions=int(training["exact_decisions"]),
                    partial_decisions=int(training["partial_decisions"]),
                    illegal_predictions=0,
                    non_finite_values=0,
                    by_decision_type={},
                    confidence_buckets={},
                    candidate_set_sizes={},
                    value_loss=float(training["value_loss"]),
                    value_decisions=int(training["value_decisions"]),
                )
            final_validation = validation
            if validation.overall_nll < best_validation_nll:
                best_validation_nll = validation.overall_nll
                best_selected_epoch = epoch
                store.save_policy(
                    best_path,
                    trainer.policy,
                    metadata={**provenance, "selected_epoch": epoch},
                )
            trainer.save_checkpoint(
                latest_path,
                epoch=epoch,
                selection_state={
                    "best_validation_nll": best_validation_nll,
                    "selected_epoch": best_selected_epoch,
                    "best_artifact": str(best_path),
                },
            )
            record = {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "epoch": epoch,
                "dataset_hash": shard_manifest.dataset_hash,
                "train_update": training,
                "training": final_training.to_dict(),
                "validation": validation.to_dict(),
            }
            _append_metrics(metrics_path, record)
            for name, value in _flatten_metrics("train", training).items():
                writer.add_scalar(name, value, epoch)
            for name, value in _flatten_metrics("validation", validation.to_dict()).items():
                writer.add_scalar(name, value, epoch)
            writer.flush()
            completed_epoch = epoch
            if (
                overfit
                and final_training.overall_nll <= initial_nll * 0.2
                and final_training.exact_joint_accuracy >= 0.9
            ):
                break
    finally:
        writer.close()
    overfit_passed = not overfit or (
        final_training.overall_nll <= initial_nll * 0.2
        and final_training.exact_joint_accuracy >= 0.9
    )
    result = {
        "dataset_hash": shard_manifest.dataset_hash,
        "runtime_hash": shard_manifest.runtime_contract_sha256,
        "completed_epoch": completed_epoch,
        "cancelled": cancelled,
        "overfit_passed": overfit_passed,
        "initial_training": initial_training.to_dict(),
        "final_training": final_training.to_dict(),
        "final_validation": (None if final_validation is None else final_validation.to_dict()),
        "latest_training_checkpoint": str(latest_path.resolve()),
        "best_policy_checkpoint": str(best_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
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
    shard_manifest, _ = _load_identities(config.shard_manifest, config.split_manifest)
    selected_device = default_device() if device is None else torch.device(device)
    store = CheckpointStore()
    policy = store.load_policy(checkpoint, selected_device)
    dataset = LazyReplayDataset(
        config.shard_manifest,
        split=split,
        split_manifest=config.split_manifest,
    )
    trainer = BCTrainer(policy, dataset, config, device=selected_device, checkpoint_store=store)
    metrics = trainer.evaluate()
    if _validation_is_failed(metrics):
        raise RuntimeError("BC evaluation contains invalid predictions or non-finite values")
    return {
        "dataset_hash": shard_manifest.dataset_hash,
        "runtime_hash": shard_manifest.runtime_contract_sha256,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "metrics": metrics.to_dict(),
    }
