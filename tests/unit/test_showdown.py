"""Tests for local Showdown process and port management."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from p0.runtime import showdown


class TestShowdown:
    def test_start_closes_log_when_process_creation_fails(self, tmp_path: Path) -> None:
        server = showdown.ShowdownServer(
            8000,
            showdown_root=tmp_path / "missing",
            log_path=tmp_path / "showdown.log",
        )

        with pytest.raises(FileNotFoundError):
            server.start()

        assert server._log_file is None

    def test_start_closes_log_when_process_arguments_are_invalid(self, tmp_path: Path) -> None:
        server = showdown.ShowdownServer(
            8000,
            showdown_root=Path(f"{tmp_path}\0"),
            log_path=tmp_path / "invalid.log",
        )

        with pytest.raises(ValueError):
            server.start()

        assert server._log_file is None

    @pytest.mark.network
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
