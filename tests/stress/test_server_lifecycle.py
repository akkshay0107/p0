from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from p0.runtime import showdown
from tests.stress._helpers import stress_repetitions


@pytest.mark.stress
def test_loopback_port_allocator_returns_distinct_reusable_ports() -> None:
    ports = showdown.allocate_loopback_ports(stress_repetitions(default=8))
    assert len(ports) == len(set(ports))
    sockets = [socket.socket() for _ in ports]
    try:
        for port, listener in zip(ports, sockets, strict=True):
            listener.bind(("127.0.0.1", port))
    finally:
        for listener in sockets:
            listener.close()


@pytest.mark.stress
def test_showdown_server_start_stop_owns_process_log_and_command(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            commands.append(command)
            self.kwargs = kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            del timeout
            return 0

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket,
        "create_connection",
        lambda *args, **kwargs: FakeConnection(),
    )
    server = showdown.ShowdownServer(
        9123,
        showdown_root=tmp_path,
        log_path=tmp_path / "nested" / "server.log",
        startup_timeout=1,
    )

    server.start()
    assert commands == [
        [
            "node",
            "--max-old-space-size=1536",
            "pokemon-showdown",
            "start",
            "--no-security",
            "--skip-build",
            "9123",
        ]
    ]
    assert server.websocket_url.endswith(":9123/showdown/websocket")
    assert server.process is not None
    server.stop()
    assert server.process is None
    assert server._log_file is None
    assert (tmp_path / "nested" / "server.log").is_file()


@pytest.mark.stress
def test_showdown_server_rejects_double_start_and_invalid_port_groups(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            del timeout
            return 0

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(showdown.socket, "create_connection", lambda *args, **kwargs: _Connection())
    server = showdown.ShowdownServer(9124, showdown_root=tmp_path, startup_timeout=1)
    server.start()
    with pytest.raises(RuntimeError, match="already started"):
        server.start()
    server.stop()

    with pytest.raises(ValueError, match="unique"):
        with showdown.start_showdown_servers(2, showdown_root=tmp_path, ports=(1, 1)):
            pass


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args


@pytest.mark.stress
def test_showdown_server_kills_process_when_graceful_stop_times_out(tmp_path: Path) -> None:
    class StuckProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("node", timeout)
            return 0

    process = StuckProcess()
    server = showdown.ShowdownServer(9125, showdown_root=tmp_path, stop_timeout=0.01)
    server.process = process
    server.stop()
    assert process.killed
