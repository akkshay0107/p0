"""Tests for local Showdown process and port management."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from p0.runtime import showdown


def _log_is_open(path: Path) -> bool:
    path_stat = path.stat()
    return any(
        entry.stat() == path_stat
        for entry in Path("/proc/self/fd").iterdir()
        if entry.exists() and entry.is_symlink()
    )


class TestShowdown:
    def test_showdown_server_group_rejects_duplicate_ports(self) -> None:
        """Reject duplicate ports before launching a process."""
        with pytest.raises(ValueError, match="unique"):
            with showdown.start_showdown_servers(2, ports=(1, 1)):
                pass

    def test_missing_assets_name_initialization_before_opening_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "showdown.log"
        server = showdown.ShowdownServer(
            8000,
            showdown_root=tmp_path / "missing",
            log_path=log_path,
        )

        with pytest.raises(FileNotFoundError, match="scripts/init-data.sh"):
            server.start()

        assert not log_path.exists()

    def test_start_timeout_closes_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "timeout.log"
        port = showdown.allocate_loopback_ports(1)[0]
        server = showdown.ShowdownServer(port, log_path=log_path, startup_timeout=0.0)

        with pytest.raises(RuntimeError, match="did not listen"):
            server.start()

        assert log_path.is_file()
        assert not _log_is_open(log_path)

    def test_loopback_port_allocator_returns_distinct_reusable_ports(self) -> None:
        ports = showdown.allocate_loopback_ports(8)
        assert len(ports) == len(set(ports))
        sockets = [socket.socket() for _ in ports]
        try:
            for port, listener in zip(ports, sockets, strict=True):
                listener.bind(("127.0.0.1", port))
        finally:
            for listener in sockets:
                listener.close()
