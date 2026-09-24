"""Integration tests for the real local Showdown process lifecycle."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from p0.runtime.showdown import allocate_loopback_ports, start_showdown_servers


class TestShowdown:
    @pytest.mark.integration
    def test_showdown_server_context_terminates_process_and_releases_port(
        self, showdown_assets
    ) -> None:
        """Verify the public server context starts, stops, and releases its port."""
        port = allocate_loopback_ports(1)[0]

        with start_showdown_servers(1, ports=(port,), build_assets=False) as servers:
            server = servers[0]
            assert server.process is not None
            process_id = server.process.pid
            assert server.port == port
            assert server.websocket_url.endswith(f":{port}/showdown/websocket")
            assert server.log_path.is_file()
            assert Path(f"/proc/{process_id}").exists()

        assert server.process is None
        assert not Path(f"/proc/{process_id}").exists()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))

    @pytest.mark.integration
    def test_showdown_server_group_rejects_duplicate_ports(self, showdown_assets) -> None:
        """Verify invalid public group configuration fails before launching a process."""
        with pytest.raises(ValueError, match="unique"):
            with start_showdown_servers(2, ports=(1, 1), build_assets=False):
                pass
