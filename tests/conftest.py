from __future__ import annotations

import pytest

from p0.paths import DEFAULT_PATHS
from p0.runtime.showdown import build_showdown


@pytest.fixture(scope="session")
def showdown_assets() -> None:
    """Build the pinned Showdown assets once for the pytest session."""
    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    build_showdown(DEFAULT_PATHS.showdown_root)
