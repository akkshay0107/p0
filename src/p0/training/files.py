"""Shared training output, metric history, and signal handling."""

from __future__ import annotations

import fcntl
import hashlib
import math
import signal
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from torch.utils.tensorboard import SummaryWriter

from p0.model.policy import PolicyNet
from p0.persistence import atomic_json_save, atomic_torch_save
from p0.training.checkpoint import CheckpointStore, LoadedCheckpoint


def code_sha256() -> str:
    """Identify the installed Python source, including uncommitted edits."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@contextmanager
def cancellation_signals() -> Iterator[Callable[[], bool]]:
    """Let both training commands finish at their next safe stopping point."""
    stop = threading.Event()
    previous = {
        name: signal.signal(name, lambda *_: stop.set()) for name in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield stop.is_set
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


@contextmanager
def training_run(
    store: CheckpointStore,
    checkpoint_path: Path,
    metrics_dir: Path,
    *,
    trainer_kind: str,
    settings: Mapping[str, Any],
    source_path: Path | None = None,
    resume: bool = False,
) -> Iterator[TrainingRun]:
    """Exclude concurrent writers and read the input before any output changes."""
    with ExitStack() as stack:
        for directory in sorted({checkpoint_path.parent.resolve(), metrics_dir.resolve()}):
            directory.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory / ".training.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"Training output is already in use: {directory}") from exc
        same_output = (
            resume
            and source_path is not None
            and source_path.resolve() == checkpoint_path.resolve()
        )
        source = store.read(source_path) if source_path is not None else None
        run = TrainingRun(
            store, checkpoint_path, metrics_dir, trainer_kind, settings, source, resume
        )
        occupied_metrics = any(
            path.name != ".training.lock"
            and not (same_output and path.resolve() == checkpoint_path.resolve())
            for path in metrics_dir.iterdir()
        )
        owns_metrics = False
        if resume and source is not None:
            old_run = source.artifact["provenance"].get("run", {})
            saved_directory = old_run.get("metrics_directory")
            owns_metrics = (
                saved_directory is not None
                and Path(saved_directory).resolve() == metrics_dir.resolve()
            )
            if occupied_metrics and not owns_metrics and old_run.get("id"):
                try:
                    metrics = orjson.loads((metrics_dir / "metrics.json").read_bytes())
                    owns_metrics = (
                        isinstance(metrics, dict) and metrics.get("run_id") == old_run["id"]
                    )
                except (OSError, orjson.JSONDecodeError):
                    # Unreadable display files cannot prove ownership of a moved directory.
                    pass
        if (checkpoint_path.exists() and not same_output) or (
            occupied_metrics and not owns_metrics
        ):
            raise ValueError(
                f"Training output already contains an experiment: {metrics_dir}; "
                "choose a fresh output directory"
            )
        del source
        try:
            yield run
        finally:
            run.close()


class TrainingRun:
    """Keep recoverable state in the checkpoint and rebuild display files from it."""

    def __init__(
        self,
        store: CheckpointStore,
        checkpoint_path: Path,
        metrics_dir: Path,
        trainer_kind: str,
        settings: Mapping[str, Any],
        source: LoadedCheckpoint | None,
        resume: bool,
    ) -> None:
        self.store = store
        self.checkpoint_path = checkpoint_path
        self.metrics_dir = metrics_dir
        self.trainer_kind = trainer_kind
        self.source = source
        old_metadata = source.artifact["provenance"] if source is not None else {}
        old_run = old_metadata.get("run", {})
        if resume:
            if (
                not isinstance(old_run, Mapping)
                or not old_run.get("id")
                or not isinstance(old_run.get("settings"), Mapping)
            ):
                raise ValueError("Checkpoint run metadata must contain a valid ID and settings")
            if old_run["settings"] != dict(settings):
                raise ValueError("Training settings do not match the checkpoint")
        training_state = source.artifact.get("training_state", {}) if source is not None else {}
        saved_state = training_state.get("run", {}) if isinstance(training_state, Mapping) else {}
        if resume and not isinstance(saved_state, Mapping):
            raise ValueError("Checkpoint run state must be a mapping")
        parent = None
        if source is not None:
            parent = {
                "sha256": source.sha256,
                "filename": source.path.name,
                "trainer_kind": old_metadata.get("trainer_kind"),
                "step": training_state.get("episode", 0),
            }
        origin = old_run.get("source")
        if origin is None and parent is not None:
            origin = {
                **parent,
                "metadata": {
                    key: old_metadata[key]
                    for key in (
                        "dataset_hash",
                        "split_manifest_sha256",
                        "trainer_config",
                        "selected_epoch",
                        "gamma",
                        "value_target_semantics",
                    )
                    if key in old_metadata
                },
            }
        self.metadata = {
            "id": old_run.get("id", str(uuid.uuid4())) if resume else str(uuid.uuid4()),
            "parent": parent,
            "source": origin,
            "settings": dict(settings),
            "code_sha256": code_sha256(),
            "metrics_directory": str(metrics_dir.resolve()),
        }
        self.state = dict(saved_state) if resume and source is not None else {}
        self.state.setdefault("metrics", [])
        self.writer: SummaryWriter | None = None

    def start(self, step: int) -> None:
        """Restore display files only after the runner has validated recovery state."""
        records = self.state["metrics"]
        if (
            not isinstance(records, list)
            or any(
                not isinstance(record, dict)
                or type(record.get("step")) is not int
                or not 0 < record["step"] <= step
                for record in records
            )
            or any(left["step"] >= right["step"] for left, right in zip(records, records[1:]))
        ):
            raise ValueError("Checkpoint metric history has invalid steps")
        for record in records:
            board = record.get("board", {})
            if not isinstance(board, Mapping):
                raise ValueError("Checkpoint board metrics must be a mapping")
            for phase, values in board.items():
                if (
                    not isinstance(phase, str)
                    or not isinstance(values, Mapping)
                    or any(
                        not isinstance(name, str)
                        or type(value) not in (int, float)
                        or not math.isfinite(value)
                        for name, value in values.items()
                    )
                ):
                    raise ValueError("Checkpoint board metrics must contain named finite scalars")
        self.writer = SummaryWriter(log_dir=str(self.metrics_dir / "tensorboard"), purge_step=0)
        for record in records:
            self.write_scalars(record["step"], record.get("board", {}))
        if step:
            self.write_outputs(step)
        # Optimizer state and the input model are no longer needed on CPU.
        self.source = None

    def record(
        self, step: int, values: Mapping[str, Any], board: Mapping[str, Mapping[str, float | int]]
    ) -> None:
        """Record one completed training step and send its scalars to TensorBoard."""
        self.state["metrics"].append(
            {
                **values,
                "step": step,
                "timestamp": datetime.now(UTC).isoformat(),
                "board": dict(board),
            }
        )
        self.write_scalars(step, board)

    def write_scalars(self, step: int, board: Mapping[str, Mapping[str, float | int]]) -> None:
        if self.writer is not None:
            for phase, metrics in board.items():
                for name, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{name}", value, step)

    def save(
        self, step: int, policy: PolicyNet, *, metadata: Mapping[str, Any], **services: Any
    ) -> None:
        """Commit recovery state first; the other files are disposable copies."""
        self.store.save_training(
            self.checkpoint_path,
            step,
            policy,
            metadata={**metadata, "run": self.metadata},
            trainer_kind=self.trainer_kind,
            run_state=self.state,
            **services,
        )
        self.write_outputs(step)

    def write_outputs(self, step: int) -> None:
        """Regenerate files that are convenient to inspect or load for inference."""
        atomic_json_save(
            self.metrics_dir / "metrics.json",
            {
                "run_id": self.metadata["id"],
                "completed_step": step,
                "metrics": self.state["metrics"],
            },
        )
        if "best_policy" in self.state:
            atomic_torch_save(
                self.checkpoint_path.parent / "bc_best_policy.pt", self.state["best_policy"]
            )
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
