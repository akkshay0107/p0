"""Start and stop pinned local Showdown processes."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

from poke_env.ps_client import ServerConfiguration

from p0.paths import DEFAULT_PATHS

LOOPBACK_HOST = "127.0.0.1"
# Local servers run with --no-security, so this public endpoint is only a placeholder
# for poke-env's required login URL.
AUTHENTICATION_URL = "https://play.pokemonshowdown.com/action.php?"


def local_server_configuration(port: int) -> ServerConfiguration:
    """Return the poke-env connection settings for a local Showdown server on port."""
    return ServerConfiguration(
        f"ws://{LOOPBACK_HOST}:{port}/showdown/websocket", AUTHENTICATION_URL
    )


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
    """Find distinct free ports on the loopback interface."""
    if count < 1:
        raise ValueError("At least one Showdown port is required")

    with contextlib.ExitStack() as stack:
        listeners = [stack.enter_context(socket.socket()) for _ in range(count)]
        for listener in listeners:
            listener.bind((LOOPBACK_HOST, 0))
        return tuple(int(listener.getsockname()[1]) for listener in listeners)


class ShowdownServer:
    """Manage one local Showdown process and its error log."""

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
        return local_server_configuration(self.port).websocket_url

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

        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.showdown_root,
                stdout=subprocess.DEVNULL,
                stderr=self._log_file,
                text=True,
            )
        except Exception:
            # Release the owned log before propagating any process-start failure.
            self._close_log()
            raise
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._raise_startup_failure(command)

            try:
                with socket.create_connection((LOOPBACK_HOST, self.port), timeout=0.2):
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

        self._close_log()

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

        self._close_log()

    def _close_log(self) -> None:
        log_file, self._log_file = self._log_file, None
        if log_file is not None:
            log_file.close()

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
    build_assets: bool = True,
    log_dir: Path | None = None,
) -> Iterator[tuple[ShowdownServer, ...]]:
    selected_ports = allocate_loopback_ports(count) if ports is None else tuple(ports)
    if len(selected_ports) != count or len(set(selected_ports)) != count:
        raise ValueError("Showdown ports must be unique and match the requested server count")

    if build_assets:
        build_showdown(showdown_root)
    with contextlib.ExitStack() as stack:
        servers = tuple(
            stack.enter_context(
                ShowdownServer(
                    port,
                    showdown_root=showdown_root,
                    log_path=log_dir / f"showdown_{port}.log" if log_dir is not None else None,
                )
            )
            for port in selected_ports
        )
        yield servers
