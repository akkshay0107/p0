import pytest


@pytest.fixture(scope="function")
def showdown_server():
    """
    Starts a local Pokemon Showdown server on port 8000.
    Ensures it is running before yielding, and cleans it up afterward.
    """
    from p0.showdown_server import SHOWDOWN_DIR, spawned_showdown

    if not SHOWDOWN_DIR.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    with spawned_showdown(port=8000):
        yield "localhost:8000"
