import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def showdown_server():
    """
    Starts a local Pokemon Showdown server on port 8000.
    Ensures it is running before yielding, and cleans it up afterward.
    """
    root_dir = Path(__file__).resolve().parent.parent
    showdown_dir = root_dir / "pokemon-showdown"

    if not showdown_dir.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    env = os.environ.copy()
    env["PORT"] = "8000"

    process = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=str(showdown_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    server_url = "http://localhost:8000"
    max_retries = 30
    ready = False

    for _ in range(max_retries):
        try:
            urllib.request.urlopen(server_url)
            ready = True
            break
        except Exception:
            time.sleep(0.5)

    if not ready:
        process.terminate()
        process.wait()
        pytest.fail("Showdown server failed to start on port 8000.")

    yield "localhost:8000"

    process.terminate()
    process.wait()
