"""Deterministic lifecycle management for pinned local Showdown processes."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from p0.paths import DEFAULT_PATHS


def build_showdown(showdown_root: Path = DEFAULT_PATHS.showdown_root) -> None:
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
    """Own exactly one local Showdown subprocess."""

    def __init__(
        self,
        port: int,
        *,
        showdown_root: Path = DEFAULT_PATHS.showdown_root,
        startup_timeout: float = 30.0,
        stop_timeout: float = 5.0,
    ) -> None:
        self.port = port
        self.showdown_root = showdown_root
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.process: subprocess.Popen[str] | None = None

    @property
    def websocket_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/showdown/websocket"

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError(f"Showdown server on port {self.port} is already started")
        command = [
            "node",
            "pokemon-showdown",
            "start",
            "--no-security",
            "--skip-build",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.showdown_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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
        stderr = self.process.communicate(timeout=1)[1][-4000:]
        code = self.process.returncode
        self.process = None
        raise RuntimeError(
            f"Showdown command {command!r} exited with {code} on port {self.port}: {stderr}"
        )

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self.stop_timeout)

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
