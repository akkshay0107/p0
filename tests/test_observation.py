import asyncio
from pathlib import Path

import pytest
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from src.model.observation_builder import from_battle
from src.model.structured_observation import CATEGORICAL_WIDTH, NUMERICAL_WIDTH, SEQUENCE_LENGTH
from src.model.tokenizer import tokenizer
from src.team_picker import RandomTeamFromPool


@pytest.fixture(scope="module")
def battle_format():
    return "gen9championsvgc2026regma"


@pytest.fixture(scope="module")
def sample_team():
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    if not teams_dir.exists():
        pytest.skip("No teams directory found.")
    team_files = [
        path.read_text(encoding="utf-8")
        for path in teams_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    return RandomTeamFromPool(team_files)


def test_tokenizer_parsing():
    """Verify tokenizer correctly maps known entities."""

    class MockPokemon:
        def __init__(self):
            self.species = "charizard"
            self.base_species = "charizard"
            self.item = "charizarditey"
            self.ability = "blaze"
            self.moves = {}

    mock_mon = MockPokemon()
    assert tokenizer.species_id(mock_mon) > 0
    assert tokenizer.item_id(mock_mon) > 0
    assert tokenizer.ability_id(mock_mon) > 0

    from poke_env.battle.effect import Effect

    effects = {Effect.CONFUSION: 2, Effect.DISABLE: 1}
    vol_ids = tokenizer.volatile_ids(effects)
    assert len(vol_ids) == 6
    assert sum(1 for v in vol_ids if v > 0) == 2


@pytest.mark.asyncio
async def test_observation_builder_live(showdown_server, battle_format, sample_team):
    """
    Spins up two RandomPlayers against the live local Showdown server.
    Hooks into `teampreview` and `choose_move` to extract live DoubleBattle states,
    and runs `from_battle` to verify the resulting observation arrays.
    """
    captured_battles = []
    captured_errors = []

    class CapturePlayer(RandomPlayer):
        def teampreview(self, battle):
            try:
                obs = from_battle(battle, tokenizer, as_dict=True)
                captured_battles.append((battle.turn, battle.teampreview, obs))
            except Exception as e:
                captured_errors.append(f"teampreview: {e}")
            return super().teampreview(battle)

        def choose_move(self, battle):
            try:
                obs = from_battle(battle, tokenizer, as_dict=True)
                captured_battles.append((battle.turn, battle.teampreview, obs))
            except Exception as e:
                captured_errors.append(f"choose_move: {e}")
            return super().choose_move(battle)

    p1 = CapturePlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=sample_team,
        max_concurrent_battles=1,
    )
    p2 = RandomPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=sample_team,
        max_concurrent_battles=1,
    )

    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=15.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Battle timed out. Internal errors: {captured_errors}")
    except Exception as e:
        pytest.fail(f"Battle failed with exception: {e}. Internal errors: {captured_errors}")

    if captured_errors:
        pytest.fail(f"Errors occurred during observation building: {captured_errors}")

    assert len(captured_battles) > 0, "No battle states were captured."

    seen_teampreview = False
    seen_normal_turn = False

    for _, is_teampreview, obs in captured_battles:
        assert isinstance(obs, dict)
        assert "categorical" in obs
        assert "numerical" in obs

        cat = obs["categorical"]
        num = obs["numerical"]

        assert cat.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
        assert num.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)

        global_field_num = num[25]
        if is_teampreview:
            seen_teampreview = True
            # teampreview flag is at index 2 in Global Field numerical array
            assert global_field_num[2].item() == 1.0
        else:
            seen_normal_turn = True
            assert global_field_num[2].item() == 0.0

    assert seen_teampreview, "Did not capture a teampreview state."
    assert seen_normal_turn, "Did not capture a normal turn state."
