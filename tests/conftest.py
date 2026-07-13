import pytest


@pytest.fixture(scope="function")
def showdown_server():
    """
    Starts a local Pokemon Showdown server on port 8000.
    Ensures it is running before yielding, and cleans it up afterward.
    """
    from p0.paths import DEFAULT_PATHS
    from p0.runtime.showdown import start_showdown_servers

    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    with start_showdown_servers(1, ports=(8000,)):
        yield "localhost:8000"
