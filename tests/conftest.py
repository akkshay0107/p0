import pytest
from poke_env import LocalhostServerConfiguration, ServerConfiguration

SHOWDOWN_TEST_PORT = 8120


@pytest.fixture(scope="function")
def showdown_server():
    """
    Starts a local Pokemon Showdown server on the dedicated test port.
    Ensures it is running before yielding, and cleans it up afterward.
    """
    from p0.paths import DEFAULT_PATHS
    from p0.runtime.showdown import start_showdown_servers

    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    server_configuration = ServerConfiguration(
        websocket_url=f"ws://localhost:{SHOWDOWN_TEST_PORT}/showdown/websocket",
        authentication_url=LocalhostServerConfiguration.authentication_url,
    )
    with start_showdown_servers(1, ports=(SHOWDOWN_TEST_PORT,)):
        yield server_configuration
