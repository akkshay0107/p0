"""Deterministic lifecycle management for pinned local Showdown processes."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

from p0.paths import DEFAULT_PATHS


def build_showdown(showdown_root: Path = DEFAULT_PATHS.showdown_root) -> None:
    """Build the local Showdown Node server assets in the target directory."""
    try:
        subprocess.run(
            ["node", "build"],
            cwd=showdown_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "")[-4000:]
        raise RuntimeError(f"Showdown build failed in {showdown_root}: {stderr}") from exc


def allocate_loopback_ports(count: int) -> tuple[int, ...]:
    """Allocate loopback socket ports dynamically."""
    if count < 1:
        raise ValueError("At least one Showdown port is required")

    listeners: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listeners.append(listener)

        return tuple(int(listener.getsockname()[1]) for listener in listeners)
    finally:
        for listener in listeners:
            listener.close()


class ShowdownServer:
    """Own exactly one local Showdown subprocess with non-blocking file log output."""

    def __init__(
        self,
        port: int,
        *,
        showdown_root: Path = DEFAULT_PATHS.showdown_root,
        log_path: Path | None = None,
        startup_timeout: float = 30.0,
        stop_timeout: float = 5.0,
    ) -> None:
        self.port = port
        self.showdown_root = showdown_root
        self.log_path = log_path or (DEFAULT_PATHS.artifacts_root / "logs" / f"showdown_{port}.log")
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.process: subprocess.Popen[str] | None = None
        self._log_file: TextIO | None = None

    @property
    def websocket_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/showdown/websocket"

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError(f"Showdown server on port {self.port} is already started")

        command = [
            "node",
            "--max-old-space-size=1536",
            "pokemon-showdown",
            "start",
            "--no-security",
            "--skip-build",
            str(self.port),
        ]

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8")

        self.process = subprocess.Popen(
            command,
            cwd=self.showdown_root,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._raise_startup_failure(command)

            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)

        self.stop()
        raise RuntimeError(
            f"Showdown command {command!r} did not listen on port {self.port} "
            f"within {self.startup_timeout:g}s"
        )

    def _raise_startup_failure(self, command: list[str]) -> None:
        assert self.process is not None
        code = self.process.returncode
        self.process = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        stderr = (
            self.log_path.read_text(encoding="utf-8")[-4000:] if self.log_path.is_file() else ""
        )
        raise RuntimeError(
            f"Showdown command {command!r} exited with {code} on port {self.port}: {stderr}"
        )

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.stop_timeout)

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> ShowdownServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


@contextlib.contextmanager
def start_showdown_servers(
    count: int,
    *,
    showdown_root: Path = DEFAULT_PATHS.showdown_root,
    ports: Sequence[int] | None = None,
) -> Iterator[tuple[ShowdownServer, ...]]:
    selected_ports = allocate_loopback_ports(count) if ports is None else tuple(ports)
    if len(selected_ports) != count or len(set(selected_ports)) != count:
        raise ValueError("Showdown ports must be unique and match the requested server count")

    build_showdown(showdown_root)
    with contextlib.ExitStack() as stack:
        servers = tuple(
            stack.enter_context(ShowdownServer(port, showdown_root=showdown_root))
            for port in selected_ports
        )
        yield servers
