"""Integration tests for the real local Showdown process lifecycle."""

from __future__ import annotations

import pytest

from p0.runtime.showdown import allocate_loopback_ports, start_showdown_servers


@pytest.mark.integration
def test_showdown_server_context_exposes_a_live_process(showdown_assets) -> None:
    """Verify the public server context starts and stops a real local process."""
    port = allocate_loopback_ports(1)[0]

    with start_showdown_servers(1, ports=(port,), build_assets=False) as servers:
        server = servers[0]
        assert server.process is not None
        assert server.port == port
        assert server.websocket_url.endswith(f":{port}/showdown/websocket")
        assert server.log_path.is_file()

    assert server.process is None


@pytest.mark.integration
def test_showdown_server_group_rejects_duplicate_ports(showdown_assets) -> None:
    """Verify invalid public group configuration fails before launching a process."""
    with pytest.raises(ValueError, match="unique"):
        with start_showdown_servers(2, ports=(1, 1), build_assets=False):
            pass
