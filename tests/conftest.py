from __future__ import annotations

import pytest

from p0.paths import DEFAULT_PATHS


@pytest.fixture(scope="session")
def showdown_assets() -> None:
    """Require the assets produced by the resource initialization script."""
    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    if not (DEFAULT_PATHS.showdown_root / "dist/sim/index.js").is_file():
        pytest.fail("Showdown assets are missing. Run bash scripts/init-data.sh.")
