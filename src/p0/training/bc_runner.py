"""Behavior cloning training and evaluation runner."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.resources import default_runtime_resources
from p0.persistence import atomic_json_save
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest
from p0.training.bc import BCCancelled, BCEvaluationMetrics, BCTrainer
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from p0.training.files import training_run
from p0.training.utils import default_device, seed_everything

LOGGER = logging.getLogger(__name__)


def _validation_is_failed(metrics: BCEvaluationMetrics) -> bool:
    return (
        metrics.non_finite_values > 0
        or metrics.labeled_count == 0
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


def _reported_path(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return str(path.resolve())


def open_bc_dataset(dataset_dir: Path) -> LazyReplayDataset:
    """
    Open one published dataset together with the splits stored beside it.

    dataset_dir is either one dataset folder, or the build-shards output folder, in
    which case latest.json names the most recently published dataset.
    """
    if not (dataset_dir / "manifest.json").is_file():
        latest = dataset_dir / "latest.json"
        if not latest.is_file():
            raise ValueError(
                f"No dataset found in {dataset_dir}; build one with 'p0-replays build-shards'"
            )
        dataset_dir = dataset_dir / orjson.loads(latest.read_bytes())["dataset_id"]

    return LazyReplayDataset(
        dataset_dir / "manifest.json", split_manifest=dataset_dir / "splits.json"
    )


def train_bc(
    config: BCConfig,
    *,
    device: torch.device | str | None = None,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """
    Train one epoch at a time, validate, and checkpoint completed epochs.

    Arguments:
        config: Behavior-cloning dataset, optimizer, and output configuration.
        device: Device override; the runtime default is used when omitted.
        cancel_requested: Callback polled between training units.

    Returns:
        The final training and validation metrics plus selected checkpoint state.
    """
    store = CheckpointStore()
    dataset = open_bc_dataset(config.dataset_dir)
    shard_manifest, split_manifest = dataset.manifest, dataset.split_manifest
    assert split_manifest is not None
    accepted_series = frozenset(dataset.accepted_series_ids())
    for split in ("train", "validation"):
        if not _has_accepted_series(split_manifest, accepted_series, split):
            raise ValueError(f"BC {split} split has no accepted series")
    train_dataset = dataset.for_split("train")
    validation_dataset = dataset.for_split("validation")
    dataset_output = config.output_dir
    latest_path = dataset_output / "bc_latest_training.pt"
    best_path = dataset_output / "bc_best_policy.pt"
    metadata = {
        "dataset_id": shard_manifest.dataset_id,
        "split_id": split_manifest.split_id,
        "gamma": config.gamma,
    }
    with training_run(
        store,
        latest_path,
        dataset_output,
        trainer_kind="bc",
        source_path=config.resume_checkpoint,
        resume=config.resume_checkpoint is not None,
    ) as files:
        selected_device = default_device() if device is None else torch.device(device)
        seed_everything(config.seed)
        policy = (
            store.load_policy(files.source, selected_device, gamma=config.gamma)
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
            source_metadata = store.load_metadata(files.source)
            for name in ("dataset_id", "split_id"):
                if source_metadata.get(name) != metadata[name]:
                    LOGGER.warning(
                        "BC resume checkpoint was trained with a different %s; "
                        "its best validation score is not comparable with this run",
                        name,
                    )

            completed_epoch = store.load_training(
                files.source,
                trainer.policy,
                trainer_kind="bc",
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
            )

            selection_state = dict(source_metadata.get("selection_state", {}))
            if not best_path.is_file():
                LOGGER.warning(
                    "No best policy exists at %s; the next epoch starts a new selection",
                    best_path,
                )
                selection_state = {}

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
        if _validation_is_failed(initial_training):
            raise RuntimeError(
                "Initial BC training evaluation contains invalid predictions or values"
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
            if _validation_is_failed(validation):
                raise RuntimeError(f"BC validation failed at epoch {epoch}")
            last_training_update = training
            final_validation = validation
            if validation.overall_nll < selection_state.get("best_validation_nll", float("inf")):
                store.save_policy(
                    best_path,
                    trainer.policy,
                    metadata={**metadata, "trainer_kind": "bc", "selected_epoch": epoch},
                )
                selection_state = {
                    "best_validation_nll": validation.overall_nll,
                    "selected_epoch": epoch,
                }
            validation_values = validation.to_dict()
            record = {
                "epoch": epoch,
                "dataset_id": shard_manifest.dataset_id,
                "training": training,
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
        result = {
            "dataset_id": shard_manifest.dataset_id,
            "completed_epoch": completed_epoch,
            "cancelled": cancelled,
            "initial_training": initial_training.to_dict(),
            "final_training": ({} if last_training_update is None else last_training_update),
            "final_validation": (None if final_validation is None else final_validation.to_dict()),
            "latest_training_checkpoint": _reported_path(latest_artifact),
            "best_policy_checkpoint": _reported_path(best_path),
            "metrics_path": _reported_path(files.metrics_path),
        }
        atomic_json_save(dataset_output / "bc-result.json", result)
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
    policy = store.load_policy(checkpoint, selected_device, gamma=config.gamma)
    dataset = open_bc_dataset(config.dataset_dir)
    shard_manifest, split_manifest = dataset.manifest, dataset.split_manifest
    assert split_manifest is not None
    accepted_series = frozenset(dataset.accepted_series_ids())
    dataset = dataset.for_split(split)
    if not _has_accepted_series(split_manifest, accepted_series, split):
        raise ValueError(f"BC {split} split has no accepted series")
    trainer = BCTrainer(policy, dataset, config, device=selected_device)
    metrics = trainer.evaluate()
    if _validation_is_failed(metrics):
        raise RuntimeError("BC evaluation contains invalid predictions or non-finite values")
    return {
        "dataset_id": shard_manifest.dataset_id,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "gamma": config.gamma,
        "metrics": metrics.to_dict(),
    }
