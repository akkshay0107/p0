"""Behavior cloning training and evaluation runner."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.utils.tensorboard import SummaryWriter

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.resources import default_runtime_resources
from p0.persistence import atomic_json_save
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest, load_split_manifest
from p0.replays.shards import load_shard_manifest
from p0.training.bc import BCCancelled, BCEvaluationMetrics, BCTrainer
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig
from p0.training.utils import default_device, seed_everything


class _SelectedPolicy(NamedTuple):
    validation_nll: float
    epoch: int
    artifact: Path | None
    artifact_sha256: str | None


def _provenance(
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
        "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
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


def _write_board_metrics(
    writer: SummaryWriter,
    phase: str,
    values: Mapping[str, float | int],
    names: Iterable[str],
    epoch: int,
) -> None:
    for name in names:
        writer.add_scalar(f"{phase}/{name}", float(values[name]), epoch)


def _append_metrics(path: Path, value: Mapping[str, Any]) -> None:
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
    if split_manifest.global_contract_sha256 != shard_manifest.global_contract_sha256:
        raise ValueError("BC shard and split manifests reference different global contracts")
    if split_manifest.dataset_hash != shard_manifest.dataset_hash:
        raise ValueError("BC shard and split manifests reference different datasets")
    return shard_manifest, split_manifest


def _has_accepted_series(
    split_manifest: SeriesSplitManifest,
    accepted_series: frozenset[str],
    split: str,
) -> bool:
    return any(
        assigned_split == split and series_id in accepted_series
        for series_id, assigned_split in split_manifest.assignments.items()
    )


def _authenticate_shards(shard_manifest: Path) -> frozenset[str]:
    """Authenticate every shard once before constructing streaming split readers."""
    authenticated = LazyReplayDataset(shard_manifest, verify_hashes=True)
    return frozenset(authenticated.accepted_series_ids())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_selected_artifact(
    store: CheckpointStore,
    resume_checkpoint: Path,
    selection_state: Mapping[str, object],
    *,
    completed_epoch: int,
    expected_metadata: Mapping[str, object],
) -> _SelectedPolicy:
    raw_score = selection_state.get("best_validation_nll")
    raw_epoch = selection_state.get("selected_epoch")
    raw_artifact = selection_state.get("best_artifact")
    raw_digest = selection_state.get("best_artifact_sha256")
    if (
        not isinstance(raw_score, (int, float))
        or isinstance(raw_score, bool)
        or not math.isfinite(float(raw_score))
        or type(raw_epoch) is not int
        or raw_epoch <= 0
        or not isinstance(raw_artifact, str)
        or not raw_artifact
        or not isinstance(raw_digest, str)
    ):
        raise ValueError("BC resume checkpoint has incomplete selected-policy state")
    if raw_epoch > completed_epoch:
        raise ValueError("BC resume checkpoint has stale selected-policy state")

    artifact = (resume_checkpoint.parent / raw_artifact).resolve()
    if not artifact.is_file():
        raise ValueError(f"BC selected policy artifact is missing: {artifact}")
    digest = _file_sha256(artifact)
    if digest != raw_digest:
        raise ValueError("BC selected policy artifact does not match its saved digest")

    metadata = store.load_metadata(artifact)
    if metadata.get("selected_epoch") != raw_epoch:
        raise ValueError("BC selected policy artifact does not match its selected epoch")
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("BC selected policy artifact does not match its saved provenance")
    return _SelectedPolicy(float(raw_score), raw_epoch, artifact, digest)


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
    checkpoint_context = (
        store.reuse_artifact(config.resume_checkpoint)
        if config.resume_checkpoint is not None
        else nullcontext()
    )
    with checkpoint_context:
        return _train_bc_with_store(
            config,
            overfit=overfit,
            device=device,
            cancel_requested=cancel_requested,
            store=store,
        )


def _train_bc_with_store(
    config: BCConfig,
    *,
    overfit: bool,
    device: torch.device | str | None,
    cancel_requested: Callable[[], bool],
    store: CheckpointStore,
) -> dict[str, Any]:
    if config.resume_checkpoint is not None:
        store.preflight(config.resume_checkpoint)

    shard_manifest, split_manifest = _load_identities(config.shard_manifest, config.split_manifest)
    accepted_series = _authenticate_shards(config.shard_manifest)
    for split in ("train", "validation"):
        if not _has_accepted_series(split_manifest, accepted_series, split):
            raise ValueError(f"BC {split} split has no accepted series")
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
    latest_path = dataset_output / "bc_latest_training.pt"
    best_path = dataset_output / "bc_best_policy.pt"
    metrics_path = dataset_output / "metrics.jsonl"
    if dataset_output.exists() and any(dataset_output.iterdir()):
        resumes_this_output = (
            config.resume_checkpoint is not None
            and config.resume_checkpoint.resolve() == latest_path.resolve()
        )
        if not resumes_this_output:
            raise ValueError(
                f"BC output directory already contains an experiment: {dataset_output}"
            )
    dataset_output.mkdir(parents=True, exist_ok=True)
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
        resume_provenance = {
            key: value for key, value in provenance.items() if key != "epoch_budget"
        }
        completed_epoch = trainer.load_checkpoint(
            config.resume_checkpoint,
            expected_metadata=resume_provenance,
        )
        selection_state = trainer.load_selection_state(config.resume_checkpoint)
        selected = _load_selected_artifact(
            store,
            config.resume_checkpoint,
            selection_state,
            completed_epoch=completed_epoch,
            expected_metadata=resume_provenance,
        )
        latest_artifact: Path | None = config.resume_checkpoint.resolve()
    else:
        completed_epoch = 0
        selection_state = {}
        selected = _SelectedPolicy(float("inf"), 0, None, None)
        latest_artifact = None
    policy = compile_policy(
        policy,
        enable=config.enable_optim and selected_device.type == "cuda",
    )
    initial_training = trainer.evaluate(train_dataset)
    if _validation_is_failed(initial_training, require_policy_support=True):
        raise RuntimeError("Initial BC training evaluation contains invalid predictions or values")
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
    final_training_evaluation: BCEvaluationMetrics | None = initial_training if overfit else None
    last_training_update: dict[str, float | int] | None = None
    final_validation: BCEvaluationMetrics | None = None
    writer = SummaryWriter(log_dir=str(dataset_output / "tensorboard"))
    cancelled = False
    try:
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
            if validation.overall_nll < selected.validation_nll:
                store.save_policy(
                    best_path,
                    trainer.policy,
                    metadata={**provenance, "selected_epoch": epoch},
                )
                artifact = best_path.resolve()
                selected = _SelectedPolicy(
                    validation.overall_nll,
                    epoch,
                    artifact,
                    _file_sha256(artifact),
                )
            trainer.save_checkpoint(
                latest_path,
                epoch=epoch,
                selection_state={
                    "best_validation_nll": selected.validation_nll,
                    "selected_epoch": selected.epoch,
                    "best_artifact": str(selected.artifact),
                    "best_artifact_sha256": selected.artifact_sha256,
                    "overfit_initial_nll": initial_nll if overfit else None,
                },
            )
            latest_artifact = latest_path.resolve()
            validation_values = validation.to_dict()
            record = {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
            _append_metrics(metrics_path, record)
            _write_board_metrics(writer, "train", training, _BC_TRAIN_BOARD_METRICS, epoch)
            _write_board_metrics(
                writer,
                "validation",
                validation_values,
                validation_values,
                epoch,
            )
            writer.flush()
            completed_epoch = epoch
            if overfit and _overfit_succeeded(final_training_evaluation, initial_nll):
                break
    finally:
        writer.close()
    overfit_passed = _overfit_succeeded(final_training_evaluation, initial_nll) if overfit else None
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
        "best_policy_checkpoint": _reported_path(selected.artifact),
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
    shard_manifest, split_manifest = _load_identities(config.shard_manifest, config.split_manifest)
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
    accepted_series = _authenticate_shards(config.shard_manifest)
    dataset = LazyReplayDataset(
        config.shard_manifest,
        split=split,
        split_manifest=config.split_manifest,
    )
    if not _has_accepted_series(split_manifest, accepted_series, split):
        raise ValueError(f"BC {split} split has no accepted series")
    trainer = BCTrainer(policy, dataset, config, device=selected_device, checkpoint_store=store)
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
