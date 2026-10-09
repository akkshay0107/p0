"""Shared training output, metric files, and signal handling."""

from __future__ import annotations

import fcntl
import signal
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from p0.model.policy import PolicyNet
from p0.persistence import atomic_output
from p0.training.checkpoint import CheckpointStore, LoadedCheckpoint
from p0.training.magnet import Magnet

METRICS_FILENAME = "metrics.jsonl"


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

        # Only a run that resumes its own checkpoint may reuse existing output.
        resumes_in_place = (
            resume
            and source_path is not None
            and source_path.resolve() == checkpoint_path.resolve()
        )
        if not resumes_in_place and (
            checkpoint_path.exists() or (metrics_dir / METRICS_FILENAME).exists()
        ):
            raise ValueError(
                f"Training output already contains an experiment: {metrics_dir}; "
                "choose a fresh output directory"
            )

        source = store.read(source_path) if source_path is not None else None
        run = TrainingRun(store, checkpoint_path, metrics_dir, trainer_kind, source)
        del source
        try:
            yield run
        finally:
            run.close()


class TrainingRun:
    """Write one run's checkpoint, metric records, and TensorBoard scalars."""

    def __init__(
        self,
        store: CheckpointStore,
        checkpoint_path: Path,
        metrics_dir: Path,
        trainer_kind: str,
        source: LoadedCheckpoint | None,
    ) -> None:
        self.store = store
        self.checkpoint_path = checkpoint_path
        self.metrics_dir = metrics_dir
        self.metrics_path = metrics_dir / METRICS_FILENAME
        self.trainer_kind = trainer_kind
        self.source = source
        self.writer: SummaryWriter | None = None

    def start(self, step: int) -> None:
        """Drop metrics recorded after the resumed step, then open TensorBoard."""
        if self.metrics_path.exists():
            kept = []
            for line in self.metrics_path.read_bytes().splitlines():
                # A killed process can leave a partial last line.
                with suppress(orjson.JSONDecodeError):
                    if orjson.loads(line)["step"] <= step:
                        kept.append(line + b"\n")

            with atomic_output(self.metrics_path) as temporary:
                temporary.write_bytes(b"".join(kept))

        # TensorBoard hides events it already holds from the first step this run records.
        self.writer = SummaryWriter(
            log_dir=str(self.metrics_dir / "tensorboard"), purge_step=step + 1
        )
        # Optimizer state and the input model are no longer needed on CPU.
        self.source = None

    def record(
        self, step: int, values: Mapping[str, Any], board: Mapping[str, Mapping[str, float | int]]
    ) -> None:
        """Append one completed training step and send its scalars to TensorBoard."""
        record = {**values, "step": step, "timestamp": datetime.now(UTC).isoformat()}
        with self.metrics_path.open("ab") as stream:
            stream.write(orjson.dumps(record, option=orjson.OPT_SORT_KEYS) + b"\n")

        if self.writer is not None:
            for phase, metrics in board.items():
                for name, value in metrics.items():
                    self.writer.add_scalar(f"{phase}/{name}", value, step)
            self.writer.flush()

    def save(
        self,
        step: int,
        policy: PolicyNet,
        *,
        optimizer: Optimizer,
        scaler: GradScaler,
        magnet: Magnet | None = None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Write the training checkpoint for a completed step."""
        self.store.save_training(
            self.checkpoint_path,
            step,
            policy,
            trainer_kind=self.trainer_kind,
            optimizer=optimizer,
            scaler=scaler,
            magnet=magnet,
            metadata=metadata,
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
