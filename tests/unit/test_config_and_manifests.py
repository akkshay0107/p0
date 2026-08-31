from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
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
    RESOURCE_FEATURE_ABI,
    TENSOR_ABI,
    RuntimeManifest,
    active_global_contract,
    canonical_json_sha256,
    current_manifest,
    load_runtime_manifest,
    sha256_file,
    validate_artifact_runtime_contract,
)
from p0.model.architecture_contract import SERIES_SLOTS, SERIES_TOKENS_PER_GAME
from p0.model.config import ModelConfig
from p0.model.fused_token_encoder import (
    FusedTokenEncoder,
)
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.model.tokenizer import PokemonTokenizer, Resolution, tokenizer
from p0.paths import DEFAULT_PATHS
from p0.training.config import (
    BCConfig,
    BotConfig,
    GlobalConfig,
    TrainingConfig,
    load_config,
)


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_config_requires_file(tmp_path: Path) -> None:
    """Verify load_config raises FileNotFoundError when the specified config file does not exist."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_applies_partial_yaml_to_source_defaults(tmp_path: Path) -> None:
    """Verify that partial YAML overrides selectively update target configuration sections while retaining default values."""
    config = load_config(write_config(tmp_path, "training:\n  n_envs: 8\n  magnet_alpha: 0.05\n"))

    assert isinstance(config, GlobalConfig)
    assert config.training.n_envs == 8
    assert config.training.magnet_alpha == 0.05
    assert config.training.num_episodes == TrainingConfig().num_episodes
    assert config.training.magnet_refresh_interval == TrainingConfig().magnet_refresh_interval


@pytest.mark.parametrize(
    ("label", "contents", "message"),
    [
        (
            "unknown training field",
            "training:\n  unknown_value: 1\n",
            "unknown TrainingConfig field",
        ),
        (
            "magnet refresh exceeds episodes",
            "training:\n  num_episodes: 10\n  magnet_refresh_interval: 20\n",
            "magnet_refresh_interval",
        ),
        (
            "mismatched bot format",
            "bot:\n  battle_format: gen9anythinggoes\n",
            "battle_format",
        ),
        (
            "removed bo3 switch",
            "bo3: 1\n",
            "unknown root configuration section",
        ),
    ],
)
def test_load_config_rejects_invalid_contracts_with_specific_errors(
    tmp_path: Path, label: str, contents: str, message: str
) -> None:
    """Verify load_config detects and rejects schema violations with informative error messages."""
    with pytest.raises(ValueError, match=message):
        load_config(write_config(tmp_path, contents))


def test_bot_config_accepts_configured_bo3_format() -> None:
    """Verify live bot configuration exposes the checked-in Bo3 format."""
    config = BotConfig()

    assert config.battle_format == FORMAT.bo3_format


def test_bot_config_rejects_bo1_format() -> None:
    """Verify the live bot configuration rejects formats unsupported by RLPlayer."""
    with pytest.raises(ValueError, match="must be the Bo3 format"):
        BotConfig(battle_format=FORMAT.battle_format)


def test_bot_config_rejects_live_concurrency_above_one() -> None:
    """Live Bo3 series tracking is intentionally single-battle for now."""
    with pytest.raises(ValueError, match="fixed at 1"):
        BotConfig(max_concurrent_battles=2)


def test_config_is_immutable(tmp_path: Path) -> None:
    """Verify GlobalConfig dataclasses are frozen to prevent accidental in-place mutation during execution."""
    config = load_config(write_config(tmp_path, "{}\n"))

    with pytest.raises(FrozenInstanceError):
        setattr(config, "training", TrainingConfig())
    with pytest.raises(FrozenInstanceError):
        setattr(config.training, "n_envs", 1)


def test_paths_and_team_pool_paths_resolve_once_from_project_root(tmp_path: Path) -> None:
    """Verify relative paths configured in YAML resolve deterministically against the project root directory."""
    config = load_config(
        write_config(
            tmp_path,
            """
paths:
  data_root: relative-data
teams:
  all: team-pool
  reduced: reduced-pool
""",
        )
    )

    assert config.paths.repository_root.is_absolute()
    assert config.paths.data_root == (Path(__file__).parents[2] / "relative-data").resolve()
    assert config.teams.all == (Path(__file__).parents[2] / "teams" / "team-pool").resolve()
    assert config.teams.reduced == (Path(__file__).parents[2] / "teams" / "reduced-pool").resolve()


def test_model_config_is_checkpoint_local_and_validated() -> None:
    """Verify ModelConfig enforces divisibility requirements (d_model % nhead == 0)."""
    config = ModelConfig.baseline()
    assert config.d_model == 512
    assert config.dim_feedforward == 2048
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(63, 8, 1, 256)


def _resources(
    tmp_path: Path, *, extra_species: bool = False, base_power: int = 90
) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


def test_runtime_manifest_round_trips_one_readable_contract(tmp_path: Path) -> None:
    """Verify RuntimeManifest serializes and deserializes losslessly while preserving ABI invariants and contract hashes."""
    vocab, dex = _resources(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    restored = RuntimeManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert restored == manifest
    assert restored.tensor_abi == TENSOR_ABI
    assert restored.resource_feature_abi == RESOURCE_FEATURE_ABI
    assert restored.action == ACTION_CONTRACT
    assert restored.global_sha256 == manifest.global_sha256


def test_canonical_hash_ignores_object_order_but_not_required_semantics() -> None:
    """Verify canonical JSON SHA-256 is insensitive to dictionary key ordering but sensitive to value changes."""
    first = {"shape": [31, 10], "dtype": "int64"}
    reordered = {"dtype": "int64", "shape": [31, 10]}
    changed = {"dtype": "int64", "shape": [32, 10]}

    assert canonical_json_sha256(first) == canonical_json_sha256(reordered)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)
    with pytest.raises(ValueError, match="unsupported value"):
        canonical_json_sha256({"scale": 0.5})


def test_vocabulary_expansion_breaks_contract_but_dex_change_does_not(tmp_path: Path) -> None:
    """Verify that vocabulary alterations modify global contract hash, while minor dex stats updates only alter dex checksum."""
    vocab, dex = _resources(tmp_path, base_power=90)
    original = current_manifest(vocab_path=vocab, dex_path=dex)

    # Modifying dex move basePower changes dex sha256 without breaking tensor shapes or global contract hash
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": 80}]}))
    rebalanced = current_manifest(vocab_path=vocab, dex_path=dex)
    assert rebalanced.global_sha256 == original.global_sha256
    assert rebalanced.champions_dex_sha256 != original.champions_dex_sha256

    # Adding new species to vocabulary changes tensor vocab sizes and breaks global contract hash
    vocab, dex = _resources(tmp_path, extra_species=True, base_power=80)
    expanded = current_manifest(vocab_path=vocab, dex_path=dex)
    assert expanded.global_sha256 != original.global_sha256


def test_global_contract_freezes_payloads_and_bumps_only_the_required_identity() -> None:
    """Verify immutable subsystem payloads and version bumping mechanics on active global contract."""
    contract = active_global_contract()
    with pytest.raises(TypeError):
        contract.payload("actions", "major")["action_count"] = 50  # type: ignore[index]

    # Minor subsystem update increments minor version without altering global major contract SHA-256
    changed_minor = contract.with_subsystem_update(
        "resources",
        minor_payload={**contract.payload("resources", "minor"), "showdown_commit": "next"},
    )
    assert changed_minor.global_sha256 == contract.global_sha256
    assert (
        changed_minor.subsystem("resources").minor_version
        == contract.subsystem("resources").minor_version + 1
    )

    # Major subsystem update alters tensor ABI, bumps major version, and recomputes global SHA-256
    changed_major = contract.with_subsystem_update(
        "model",
        major_payload={**contract.payload("model", "major"), "tensor_abi": "next"},
    )
    assert changed_major.global_sha256 != contract.global_sha256
    assert (
        changed_major.subsystem("model").major_version
        == contract.subsystem("model").major_version + 1
    )

    # Major subsystem bump resets minor version counter to 0
    bumped = contract.with_subsystem_update(
        "resources",
        major_payload={**contract.payload("resources", "major"), "resource_feature_abi": "next"},
    )
    assert contract.subsystem("resources").minor_version > 0
    assert bumped.subsystem("resources").minor_version == 0


def test_global_contract_rejects_hash_valid_but_malformed_subsystem_payload() -> None:
    """Verify RuntimeManifest.create validates non-empty payloads for all registered subsystems."""
    contract = active_global_contract()
    contracts = {
        name: {
            "major": dict(contract.payload(name, "major")),
            "minor": dict(contract.payload(name, "minor")),
        }
        for name in contract.subsystems
    }
    contracts["actions"]["major"] = {}
    with pytest.raises(ValueError, match="actions major payload"):
        RuntimeManifest.create(contracts, contract.subsystems)


def test_artifact_validation_uses_only_the_active_global_contract() -> None:
    """Verify validate_artifact_runtime_contract accepts artifacts matching current global contract and rejects mismatches."""
    manifest = active_global_contract()
    artifact = {"global_contract_sha256": manifest.global_sha256}
    assert validate_artifact_runtime_contract(artifact) == manifest
    artifact["global_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        validate_artifact_runtime_contract(artifact)


def test_model_config_has_only_scaling_fields() -> None:
    """Verify ModelConfig accepts valid scaling architectures and rejects deprecated or incompatible parameters."""
    config = ModelConfig.baseline()
    assert config.dim_feedforward == 2048
    assert ModelConfig.from_dict(config.to_dict()) == config
    enabled = ModelConfig(
        d_model=64,
        nhead=4,
        reducer_layers=1,
        dim_feedforward=128,
    )
    assert ModelConfig.from_dict(enabled.to_dict()) == enabled
    stale = config.to_dict()
    stale["history_tokens"] = 8
    with pytest.raises(ValueError, match=r"unknown=.*history_tokens"):
        ModelConfig.from_dict(stale)
    with pytest.raises(ValueError, match="low-width event channel"):
        ModelConfig(d_model=96, nhead=3, reducer_layers=1, dim_feedforward=128)


def test_config_sections(tmp_path: Path) -> None:
    """Verify current config sections load correctly and disallow conflicting objective parameters."""
    config = load_config("config.example.yaml")
    assert config.bc.batch_decisions == 256
    assert config.bc.gamma == config.training.gamma
    assert config.bc.value_coef == config.training.value_coef
    assert config.teams.all == (ROOT / "teams" / "all").resolve()
    assert config.teams.reduced == (ROOT / "teams" / "reduced").resolve()
    assert config.evaluation.episodes_per_matchup == 20
    bad = tmp_path / "config.yaml"
    bad.write_text("bc:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown BCConfig field"):
        load_config(bad)

    duplicate_objective = tmp_path / "duplicate-objective.yaml"
    duplicate_objective.write_text(
        "training:\n  gamma: 0.95\nbc:\n  gamma: 0.9\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bc.gamma is derived from training"):
        load_config(duplicate_objective)

    with pytest.raises(ValueError, match="bc.gamma must match training.gamma"):
        GlobalConfig(training=TrainingConfig(gamma=0.95), bc=BCConfig(gamma=0.9))


def test_schema_modules_stay_pure() -> None:
    """Verify intermediate representation modules (p0.replays.schema, p0.battle.series) stay pure without importing torch or runtime."""
    code = (
        "import sys\n"
        "import p0.replays.schema, p0.battle.series\n"
        "assert 'torch' not in sys.modules, 'IR layer must stay torch-free'\n"
        "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
        "import p0.replays.shards\n"
        "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


ROOT = Path(__file__).resolve().parents[2]


def test_active_contract_is_reg_m_b_and_manifest_matches_sources() -> None:
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


def test_runtime_resources_reject_unrecorded_dex_content(tmp_path: Path) -> None:
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


def test_every_legal_content_key_resolves() -> None:
    """Verify all legal species, items, abilities, and moves in champions_dex resolve to known tokenizer IDs."""
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tokenizer_instance = PokemonTokenizer(vocab)
    for table in ("species", "items", "abilities", "moves"):
        for key in dex["legality"][table]:
            assert tokenizer_instance.resolve(table, key)[1] == "known", (table, key)


def test_item_and_ability_mechanics_reach_public_encoder_output() -> None:
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


def test_field_namespace_and_coverage_audit_are_present(tmp_path: Path) -> None:
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


def test_reg_mb_legality_inventory_uses_resolved_showdown_rules() -> None:
    """Verify Regulation M-B format legality whitelist includes legal items/moves and excludes banned content."""
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    assert "pikachu" in dex["legality"]["species"]
    assert "protect" in dex["legality"]["moves"]
    assert "ababo" not in dex["legality"]["species"]
    assert "berserkgene" not in dex["legality"]["items"]


def test_generation_is_deterministic_and_nonlegal_effects_are_reported(tmp_path: Path) -> None:
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


def test_unknown_legal_effect_namespace_fails_generation(tmp_path: Path) -> None:
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


def test_tokenizer_normalization_and_table_resolution() -> None:
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


def test_tokenizer_normalization_cache_is_stable_across_repeated_protocol_ids() -> None:
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
            character.lower() for character in value if character.isascii() and character.isalnum()
        )
        for value in values
    )
    for _ in range(4):
        assert tuple(PokemonTokenizer.normalize_id(value) for value in values) == expected


def test_tokenizer_domain_objects_and_missing_values() -> None:
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
        teambuilder=TeambuilderPokemon(species="pikachu", nature="Serious", evs=[1, 0, 0, 0, 0, 0]),
    )
    serious_id = tokenizer.nature_id(p)
    assert serious_id == 0
    bashful = Pokemon(
        gen=9,
        teambuilder=TeambuilderPokemon(species="pikachu", nature="Bashful", evs=[1, 0, 0, 0, 0, 0]),
    )
    assert serious_id == tokenizer.nature_id(bashful)

    # Non-neutral natures map to positive integers > 0
    jolly = Pokemon(
        gen=9,
        teambuilder=TeambuilderPokemon(species="pikachu", nature="Jolly", evs=[1, 0, 0, 0, 0, 0]),
    )
    jolly_id = tokenizer.nature_id(jolly)
    assert jolly_id > 0
    assert tokenizer.natures_list[jolly_id] == "jolly"

    adamant = Pokemon(
        gen=9,
        teambuilder=TeambuilderPokemon(species="pikachu", nature="Adamant", evs=[1, 0, 0, 0, 0, 0]),
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


def test_token_store_initialization() -> None:
    """Verify SeriesTokenStore initializes empty storage with specified dimensions."""
    store = SeriesTokenStore(d_model=64, max_games=2)
    assert store.d_model == 64
    assert store.max_games == 2


def test_token_store_append_and_get() -> None:
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


def test_token_store_max_games_truncation() -> None:
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


def test_token_store_batching_and_missing() -> None:
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


def test_token_store_training_state_round_trip() -> None:
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
    assert torch.allclose(tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], second)


def test_token_store_isolates_canonical_player_perspectives() -> None:
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


def test_token_store_drop_and_clear() -> None:
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


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


def test_runtime_manifest_digest_is_semantic_and_round_trips(tmp_path: Path) -> None:
    """Verify runtime manifest digest computation is invariant to key reordering in on-disk JSON."""
    vocab, dex = _runtime_files(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    reordered = json.loads(json.dumps(manifest.to_dict()))
    reordered["contracts"]["actions"]["major"] = {
        key: reordered["contracts"]["actions"]["major"][key]
        for key in reversed(tuple(reordered["contracts"]["actions"]["major"]))
    }
    assert RuntimeManifest.from_dict(reordered) == manifest
    path = tmp_path / "runtime_manifest.json"
    path.write_text(json.dumps(reordered), encoding="utf-8")
    assert load_runtime_manifest(path) == manifest


def test_tokenizer_aliases_and_resolution_keep_unknown_zero_distinct_from_known_none() -> None:
    """Verify tokenizer distinguishes between KNOWN, KNOWN_NONE (valid empty entity), OOV, and UNKNOWN."""
    tokenizer_instance = PokemonTokenizer(
        {
            "weathers": {"raindance": 4},
            "status": {"brn": 5},
            "moves": {"uturn": 7},
        }
    )
    assert tokenizer_instance.id_for("moves", "U-turn") == 7
    assert tokenizer_instance.effect_id_for("status", "status: brn") == 5
    assert tokenizer_instance.resolve("weathers", "rain") == (4, Resolution.KNOWN)
    assert tokenizer_instance.resolve("status", "burn") == (5, Resolution.KNOWN)
    assert tokenizer_instance.resolve("status", "not-a-status") == (0, Resolution.OOV)
    assert tokenizer_instance.resolve("status", None) == (0, Resolution.KNOWN_NONE)
    assert tokenizer_instance.resolve("missing", "rain") == (0, Resolution.UNKNOWN)


def test_enum_like_tables_lazy_cache_alias_and_missing_member_results() -> None:
    """Verify lazy alias dictionary caching on enum-like tables (weathers, status)."""
    tokenizer_instance = PokemonTokenizer({"weathers": {"raindance": 4}, "status": {"brn": 5}})
    assert tokenizer_instance.weathers["R-a-i-n"] == 4
    assert tokenizer_instance.weathers["rain"] == 4
    assert tokenizer_instance.weathers == {"rain": 4}
    assert tokenizer_instance.weathers["unknown-weather"] == 0
    assert "unknownweather" not in tokenizer_instance.weathers
    assert tokenizer_instance.status["burn"] == 5
    assert tokenizer_instance.status[Status.BRN] == 5
    assert tokenizer_instance.status["unknown-status"] == 0
    assert tokenizer_instance.status == {"burn": 5, "brn": 5}
    assert all(isinstance(key, str) for key in tokenizer_instance.status)


def test_active_contract_rejects_an_unrecorded_spread_table() -> None:
    """Verify active global contract checks spread_usage.json checksum."""
    contract = active_global_contract()
    assert contract.spread_usage_sha256 == sha256_file(
        DEFAULT_PATHS.data_root / "spread_usage.json"
    )
