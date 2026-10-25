from __future__ import annotations

import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.data import GenData
from poke_env.ps_client.ps_client import PSClient

from p0.paths import DEFAULT_PATHS

# The poke-env classes that p0.runtime.poke_env_patches changes.
_PATCHED_CLASSES = (PSClient, DoubleBattle, Pokemon, Move)


@pytest.fixture(autouse=True)
def restore_poke_env():
    """Undo the poke-env patches a test installs, so no test depends on run order."""
    gen_data = GenData.from_gen(9)
    saved_data = gen_data.pokedex, gen_data.moves
    saved_attributes = [(owner, dict(vars(owner))) for owner in _PATCHED_CLASSES]
    yield
    for owner, attributes in saved_attributes:
        for name in tuple(vars(owner)):
            # A patch of an inherited method adds a name the class did not have.
            if name not in attributes:
                delattr(owner, name)
            elif vars(owner)[name] is not attributes[name]:
                setattr(owner, name, attributes[name])
    if gen_data.pokedex is not saved_data[0]:
        gen_data.pokedex, gen_data.moves = saved_data
        Move.should_be_stored.cache_clear()


@pytest.fixture(scope="session")
def showdown_assets() -> None:
    """Require the assets produced by the resource initialization script."""
    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    if not (DEFAULT_PATHS.showdown_root / "dist/sim/index.js").is_file():
        pytest.fail("Showdown assets are missing. Run bash scripts/init-data.sh.")
