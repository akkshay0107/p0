"""Tests for local Showdown process and port management."""

from __future__ import annotations

import socket

import pytest

from p0.runtime import showdown


class TestShowdown:
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
