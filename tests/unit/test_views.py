"""Tests for transformed battle view wrappers."""

from __future__ import annotations

from poke_env.battle import Pokemon
from poke_env.teambuilder.teambuilder import TeambuilderPokemon

from p0.battle.views import (
    TransformedPokemonView,
)


def _pokemon(
    species: str,
    *,
    ability: str,
    item: str,
    moves: list[str],
) -> Pokemon:
    return Pokemon(
        gen=9,
        teambuilder=TeambuilderPokemon(
            species=species,
            ability=ability,
            item=item,
            moves=moves,
        ),
    )


class TestTransformedPokemonView:
    def test_transformed_pokemon_view_equality_and_hashing(self) -> None:
        """Verify equality symmetry, hashing, and comparison between transformed views and base."""
        base1 = _pokemon("ditto", ability="imposter", item="choicescarf", moves=["transform"])
        base2 = _pokemon("smeargle", ability="own tempo", item="choicescarf", moves=["transform"])
        target = _pokemon(
            "koraidon", ability="orichalcum pulse", item="rusted sword", moves=["collision course"]
        )

        t1 = TransformedPokemonView(base1, target)
        t2 = TransformedPokemonView(base1, target)
        t3 = TransformedPokemonView(base2, target)

        assert t1 == t2
        assert t2 == t1
        assert t1 == base1
        assert base1 == t1
        assert t1 != t3
        assert hash(t1) == hash(base1)
        assert hash(t1) == hash(t2)
