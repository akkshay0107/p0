"""Tests for runtime resources and token-store initialization."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
from poke_env.battle import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.teambuilder.teambuilder import TeambuilderPokemon

from p0.cli.build_vocab import build
from p0.format_config import (
    ACTION_CONTRACT,
    FORMAT,
    RuntimeManifest,
)
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.fused_token_encoder import (
    FusedTokenEncoder,
)
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.model.tokenizer import PokemonTokenizer, Resolution, tokenizer

ROOT = Path(__file__).resolve().parents[2]


class TestRuntimeResources:
    def test_active_contract_is_reg_m_b_and_manifest_matches_sources(self) -> None:
        """Verify default runtime_manifest matches Champions Regulation M-B battle formats and action contract."""
        manifest = RuntimeManifest.from_dict(
            json.loads((ROOT / "data/runtime_manifest.json").read_text())
        )
        assert FORMAT.battle_format == "gen9championsvgc2026regmb"
        assert FORMAT.bo3_format == "gen9championsvgc2026regmbbo3"
        assert manifest.battle_format == FORMAT.battle_format
        assert manifest.bo3_format == FORMAT.bo3_format
        assert manifest.action == ACTION_CONTRACT
        assert len(manifest.global_sha256) == 64

    def test_runtime_resources_reject_unrecorded_dex_content(self, tmp_path: Path) -> None:
        """Verify RuntimeResources rejects champions_dex.json modifications not recorded in runtime_manifest."""
        data = tmp_path / "data"
        data.mkdir()
        for name in ("runtime_manifest.json", "vocab.json", "champions_dex.json"):
            shutil.copy2(ROOT / "data" / name, data / name)
        dex_path = data / "champions_dex.json"
        dex = json.loads(dex_path.read_text())
        dex["moves"][0]["basePower"] = int(dex["moves"][0].get("basePower", 0)) + 1
        dex_path.write_text(json.dumps(dex), encoding="utf-8")

        with pytest.raises(ValueError, match="default global manifest"):
            RuntimeResources.from_manifest(data / "runtime_manifest.json")

    def test_every_legal_content_key_resolves(self) -> None:
        """Verify all legal species, items, abilities, and moves in champions_dex resolve to known tokenizer IDs."""
        vocab = json.loads((ROOT / "data/vocab.json").read_text())
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        tokenizer_instance = PokemonTokenizer(vocab)
        for table in ("species", "items", "abilities", "moves"):
            for key in dex["legality"][table]:
                assert tokenizer_instance.resolve(table, key)[1] == "known", (table, key)

    def test_item_and_ability_mechanics_reach_public_encoder_output(self) -> None:
        """Verify changing public item and ability IDs changes the encoded Pokémon token."""
        resources = default_runtime_resources()
        encoder = FusedTokenEncoder(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            resources=resources,
        )
        encoder.eval()
        first = StructuredObservation.empty_batch(1)
        second = first.clone()
        first.categorical[0, 0, 1] = resources.tokenizer.id_for("abilities", "intimidate")
        first.categorical[0, 0, 2] = resources.tokenizer.id_for("items", "sitrusberry")
        second.categorical[0, 0, 1] = resources.tokenizer.id_for("abilities", "defiant")
        second.categorical[0, 0, 2] = resources.tokenizer.id_for("items", "leftovers")
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)

        with torch.inference_mode():
            first_tokens, _ = encoder(first, action_mask)
            second_tokens, _ = encoder(second, action_mask)

        assert not torch.equal(first_tokens[:, 0], second_tokens[:, 0])

    def test_field_namespace_and_coverage_audit_are_present(self, tmp_path: Path) -> None:
        """Verify build_vocab produces clean coverage audits with 0 missing legal effects."""
        vocab = json.loads((ROOT / "data/vocab.json").read_text())
        report = build(
            ROOT / "data/champions_dex.json",
            tmp_path / "vocab.json",
            tmp_path / "manifest.json",
        )
        assert "trickroom" in vocab["fields"]
        assert report["missingLegalContent"] == {}
        assert report["unmappedLegalEffects"] == []

    def test_reg_mb_legality_inventory_uses_resolved_showdown_rules(self) -> None:
        """Verify Regulation M-B format legality whitelist includes legal items/moves and excludes banned content."""
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        assert "pikachu" in dex["legality"]["species"]
        assert "protect" in dex["legality"]["moves"]
        assert "ababo" not in dex["legality"]["species"]
        assert "berserkgene" not in dex["legality"]["items"]

    def test_generation_is_deterministic_and_nonlegal_effects_are_reported(
        self, tmp_path: Path
    ) -> None:
        """Verify build_vocab execution is bitwise deterministic and unsupported nonlegal effects are logged in coverage audit."""
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        dex["protocolEffects"].append("nonlegaltesteffect")
        dex_path = tmp_path / "dex.json"
        dex_path.write_text(json.dumps(dex), encoding="utf-8")
        outputs = []
        for suffix in ("a", "b"):
            vocab = tmp_path / f"vocab-{suffix}.json"
            manifest = tmp_path / f"manifest-{suffix}.json"
            coverage = tmp_path / f"coverage-{suffix}.json"
            build(dex_path, vocab, manifest, coverage)
            outputs.append((vocab.read_bytes(), manifest.read_bytes(), coverage.read_bytes()))
        assert outputs[0] == outputs[1]
        report = json.loads(outputs[0][2])
        assert "condition:nonlegaltesteffect" in report["unsupportedNonlegalEffects"]

    def test_unknown_legal_effect_namespace_fails_generation(self, tmp_path: Path) -> None:
        """Verify build_vocab fails with ValueError if an unknown effect namespace is introduced into legal effects."""
        dex = deepcopy(json.loads((ROOT / "data/champions_dex.json").read_text()))
        dex["legalProtocolEffects"]["unmapped_family"] = ["reachableeffect"]
        dex_path = tmp_path / "dex.json"
        dex_path.write_text(json.dumps(dex), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown legal protocol-effect namespace"):
            build(
                dex_path,
                tmp_path / "vocab.json",
                tmp_path / "manifest.json",
                tmp_path / "coverage.json",
            )

    def test_tokenizer_normalization_and_table_resolution(self) -> None:
        """Verify PokemonTokenizer normalization and resolution taxonomy: KNOWN, KNOWN_NONE, OOV, UNKNOWN."""
        assert PokemonTokenizer.normalize_id("Charizard-Mega-Y") == "charizardmegay"
        assert PokemonTokenizer.normalize_id("U-turn") == "uturn"
        assert PokemonTokenizer.normalize_id("Leech Seed") == "leechseed"
        assert PokemonTokenizer.normalize_id("  Thunderbolt  ") == "thunderbolt"
        assert PokemonTokenizer.normalize_id(None) == ""
        custom_vocab = {
            "custom_table": {
                "apple": 1,
                "banana": 2,
            }
        }
        tok = PokemonTokenizer(custom_vocab)
        assert tok.id_for("custom_table", "Apple") == 1
        assert tok.id_for("custom_table", "cherry") == 0
        assert tok.id_for("missing_table", "apple") == 0
        species = PokemonTokenizer({"species": {"pikachu": 1}})
        assert species.resolve("species", None) == (0, Resolution.KNOWN_NONE)
        assert species.resolve("species", "missingno") == (0, Resolution.OOV)
        assert species.resolve("species", "pikachu") == (1, Resolution.KNOWN)

    def test_tokenizer_normalization_cache_is_stable_across_repeated_protocol_ids(self) -> None:
        """Verify repeated normalization calls return stable canonical IDs."""
        values = (
            "Charizard-Mega-Y",
            "U-turn",
            "Leech Seed",
            "CHARIZARD-MEGA-Y",
            "Species-1",
            "Species-2",
        )
        expected = tuple(
            "".join(
                character.lower()
                for character in value
                if character.isascii() and character.isalnum()
            )
            for value in values
        )
        for _ in range(4):
            assert tuple(PokemonTokenizer.normalize_id(value) for value in values) == expected

    def test_tokenizer_domain_objects_and_missing_values(self) -> None:
        """Verify tokenizer extracts vocabulary IDs from poke-env domain objects (Pokemon, Move, Status, PokemonType, Nature)."""
        assert tokenizer.status_id(Status.BRN) == tokenizer.status[Status.BRN]
        assert tokenizer.status_id(Status.SLP) == tokenizer.status[Status.SLP]
        assert tokenizer.status_id(None) == 0
        assert tokenizer.status_id(cast(Status, "UNKNOWN_STATUS")) == 0

        p1 = Pokemon(gen=9, species="archaludon")
        assert tokenizer.species_id(p1) == tokenizer.vocab["species"]["archaludon"]

        class FallbackPokemon(Pokemon):
            @property
            def species(self) -> str:
                return ""

            @property
            def base_species(self) -> str:
                return "charizard"

        p2 = FallbackPokemon(gen=9, species="charizard")
        assert tokenizer.species_id(p2) == tokenizer.vocab["species"]["charizard"]
        assert tokenizer.species_id(None) == 0

        p3 = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(species="charizard", ability="intimidate"),
        )
        assert tokenizer.ability_id(p3) == tokenizer.vocab["abilities"]["intimidate"]
        assert tokenizer.ability_id(None) == 0

        p4 = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(species="charizard", item="choicescarf"),
        )
        assert tokenizer.item_id(p4) == tokenizer.vocab["items"]["choicescarf"]
        assert tokenizer.item_id(None) == 0

        assert tokenizer.type_id(PokemonType.FIRE) == tokenizer.vocab["types"]["fire"]
        assert tokenizer.type_id(PokemonType.WATER) == tokenizer.vocab["types"]["water"]
        assert tokenizer.type_id(None) == 0

        m1 = Move("closecombat", 9)
        assert tokenizer.move_id(m1) == tokenizer.vocab["moves"]["closecombat"]
        m_aquajet = Move("aquajet", 9)
        assert tokenizer.move_id(m_aquajet) == tokenizer.vocab["moves"]["aquajet"]
        assert tokenizer.move_id(None) == 0

        assert tokenizer.move_type_id(m1) == tokenizer.vocab["types"]["fighting"]
        assert tokenizer.move_type_id(None) == 0

        m2 = Move("thunderbolt", 9)
        assert tokenizer.move_category_id(m2) == 2

        m3 = Move("protect", 9)
        assert tokenizer.move_category_id(m3) == 3
        assert tokenizer.move_category_id(None) == 0
        assert tokenizer.nature_id(None) == 0

        p = Pokemon(gen=9, species="pikachu")
        assert tokenizer.nature_id(p) == 0

        # Neutral natures map to 0
        p = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(
                species="pikachu", nature="Serious", evs=[1, 0, 0, 0, 0, 0]
            ),
        )
        serious_id = tokenizer.nature_id(p)
        assert serious_id == 0
        bashful = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(
                species="pikachu", nature="Bashful", evs=[1, 0, 0, 0, 0, 0]
            ),
        )
        assert serious_id == tokenizer.nature_id(bashful)

        # Non-neutral natures map to positive integers > 0
        jolly = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(
                species="pikachu", nature="Jolly", evs=[1, 0, 0, 0, 0, 0]
            ),
        )
        jolly_id = tokenizer.nature_id(jolly)
        assert jolly_id > 0
        assert tokenizer.natures_list[jolly_id] == "jolly"

        adamant = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(
                species="pikachu", nature="Adamant", evs=[1, 0, 0, 0, 0, 0]
            ),
        )
        adamant_id = tokenizer.nature_id(adamant)
        assert adamant_id > 0
        assert tokenizer.natures_list[adamant_id] == "adamant"

        unknown = Pokemon(
            gen=9,
            teambuilder=TeambuilderPokemon(
                species="pikachu", nature="unknown_nature", evs=[1, 0, 0, 0, 0, 0]
            ),
        )
        assert tokenizer.nature_id(unknown) == 0

    def test_token_store_initialization(self) -> None:
        """Verify SeriesTokenStore initializes empty storage with specified dimensions."""
        store = SeriesTokenStore(d_model=64, max_games=2)
        assert store.d_model == 64
        assert store.max_games == 2

    def test_token_store_append_and_get(self) -> None:
        """Verify SeriesTokenStore appends tokens, pads remaining slots with zeros, and computes active boolean masks."""
        store = SeriesTokenStore(d_model=16, max_games=2)

        tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        store.append("series-A", tokens1)

        out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
        assert out_tokens.shape == (1, SERIES_SLOTS, 16)
        assert out_mask.shape == (1, SERIES_SLOTS)

        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
        assert torch.all(out_tokens[0, SERIES_TOKENS_PER_GAME:] == 0)
        assert torch.all(out_mask[0, :SERIES_TOKENS_PER_GAME])
        assert not torch.any(out_mask[0, SERIES_TOKENS_PER_GAME:])

        tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        store.append("series-A", tokens2)

        out_tokens, out_mask = store.get_tokens(["series-A"], device=torch.device("cpu"))
        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens1)
        assert torch.allclose(
            out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens2
        )
        assert torch.all(out_mask[0, : 2 * SERIES_TOKENS_PER_GAME])

    def test_token_store_max_games_truncation(self) -> None:
        """Verify SeriesTokenStore retains only the most recent max_games=2 entries using FIFO eviction."""
        store = SeriesTokenStore(d_model=16, max_games=2)

        tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
        tokens3 = torch.randn(SERIES_TOKENS_PER_GAME, 16)

        store.append("series-B", tokens1)
        store.append("series-B", tokens2)
        store.append("series-B", tokens3)

        out_tokens, out_mask = store.get_tokens(["series-B"], device=torch.device("cpu"))

        # Oldest tokens (tokens1) must be evicted; tokens2 and tokens3 retained
        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens2)
        assert torch.allclose(
            out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens3
        )

    def test_token_store_batching_and_missing(self) -> None:
        """Verify SeriesTokenStore returns zero-filled tensors and False masks for unknown series keys."""
        store = SeriesTokenStore(d_model=8)

        t1 = torch.randn(SERIES_TOKENS_PER_GAME, 8)
        store.append("s1", t1)

        out_tokens, out_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
        assert out_tokens.shape == (2, SERIES_SLOTS, 8)

        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], t1)
        assert out_mask[0, 0]

        assert torch.all(out_tokens[1] == 0)
        assert not torch.any(out_mask[1])

    def test_token_store_training_state_round_trip(self) -> None:
        """Verify SeriesTokenStore state dictionary roundtrips losslessly into a fresh instance."""
        store = SeriesTokenStore(d_model=8)
        first = torch.randn(SERIES_TOKENS_PER_GAME, 8)
        second = torch.randn(SERIES_TOKENS_PER_GAME, 8)
        store.append("series-1", first)
        store.append("series-1", second)

        restored = SeriesTokenStore(d_model=8)
        restored.restore_training_state(store.training_state())

        tokens, mask = restored.get_tokens(["series-1"], device=torch.device("cpu"))
        assert torch.all(mask[0, : 2 * SERIES_TOKENS_PER_GAME])
        assert torch.allclose(tokens[0, :SERIES_TOKENS_PER_GAME], first)
        assert torch.allclose(
            tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], second
        )

    def test_token_store_isolates_canonical_player_perspectives(self) -> None:
        """Verify SeriesTokenStore isolates storage between canonical player 0 and player 1 perspectives."""
        from p0.battle.series import SeriesPerspectiveKey

        store = SeriesTokenStore(d_model=8)
        first_player = SeriesPerspectiveKey("series-1", 0)
        second_player = SeriesPerspectiveKey("series-1", 1)
        tokens = torch.randn(SERIES_TOKENS_PER_GAME, 8)

        store.append(first_player, tokens)

        out_tokens, out_mask = store.get_tokens(
            [first_player, second_player],
            device=torch.device("cpu"),
        )
        assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens)
        assert torch.all(out_mask[0, :SERIES_TOKENS_PER_GAME])
        assert not torch.any(out_mask[1])

    def test_token_store_drop_and_clear(self) -> None:
        """Verify SeriesTokenStore drop() deletes specific series keys and clear() purges all stored tokens."""
        store = SeriesTokenStore(d_model=8)
        store.append("s1", torch.randn(SERIES_TOKENS_PER_GAME, 8))

        store.drop("s1")
        out_tokens, out_mask = store.get_tokens(["s1"], device=torch.device("cpu"))
        assert not torch.any(out_mask)

        store.append("s2", torch.randn(SERIES_TOKENS_PER_GAME, 8))
        store.clear()
        out_tokens, out_mask = store.get_tokens(["s2"], device=torch.device("cpu"))
        assert not torch.any(out_mask)
