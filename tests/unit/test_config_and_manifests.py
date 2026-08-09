from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
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

from p0.battle.series import SeriesPerspectiveKey
from p0.cli.build_vocab import build
from p0.cli.corpus import _variants_from_showdown
from p0.cli.corpus import main as corpus_main
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
    _load_mechanic_tag_tables,
    _load_move_statics,
    _load_species_statics,
)
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.token_store import SeriesTokenStore
from p0.model.tokenizer import PokemonTokenizer, Resolution, tokenizer
from p0.paths import DEFAULT_PATHS
from p0.replays.schema import (
    ActionEvidence,
    DecisionRecord,
    DecisionType,
    FetchIndexEntry,
    GameEndReason,
    GameRecord,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    ReplayDiagnostics,
    SeriesRecord,
)
from p0.replays.shards import (
    SHARD_TENSOR_SPECS,
    ShardIndexEntry,
    ShardManifest,
    load_shard_manifest,
    observation_field_specs,
)
from p0.teams.corpus import (
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)
from p0.teams.corpus_build import build_corpus
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FileTeamSource
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord
from p0.teams.validation import AdmissionResult
from p0.training.config import (
    BCConfig,
    CorpusConfig,
    GlobalConfig,
    TeamSourceConfig,
    TrainingConfig,
    load_config,
)
from p0.training.ppo_runner import _team_source


def write_config(tmp_path, contents: str):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_config_validation_and_parsing(tmp_path):
    def _test_load_config_requires_file(tmp_path):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(tmp_path / "missing.yaml")

    _test_load_config_requires_file(tmp_path)

    def _test_load_config_applies_partial_yaml_to_source_defaults(tmp_path):
        config = load_config(
            write_config(tmp_path, "training:\n  n_envs: 8\n  magnet_alpha: 0.05\n")
        )

        assert isinstance(config, GlobalConfig)
        assert config.training.n_envs == 8
        assert config.training.magnet_alpha == 0.05
        assert config.training.num_episodes == TrainingConfig().num_episodes
        assert config.training.magnet_refresh_interval == TrainingConfig().magnet_refresh_interval

    _test_load_config_applies_partial_yaml_to_source_defaults(tmp_path)

    def _test_load_config_rejects_invalid_contracts_with_specific_errors(tmp_path):
        cases = (
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
                "removed team-source kind",
                "environment:\n  agent_team_source:\n    kind: directory_magic\n",
                "unknown TeamSourceConfig field",
            ),
            (
                "mismatched bot format",
                "bot:\n  battle_format: gen9anythinggoes\n",
                "battle_format",
            ),
            (
                "invalid corpus sampling policy",
                "corpus:\n  sampling_policy: made_up\n",
                "sampling_policy",
            ),
            (
                "removed bo3 switch",
                "bo3: 1\n",
                "unknown root configuration section",
            ),
        )
        for label, contents, message in cases:
            try:
                load_config(write_config(tmp_path, contents))
            except ValueError as exc:
                assert re.search(message, str(exc)), f"{label}: unexpected error: {exc}"
            else:
                pytest.fail(f"{label}: expected ValueError")

    _test_load_config_rejects_invalid_contracts_with_specific_errors(tmp_path)


def test_config_is_immutable(tmp_path):
    config = load_config(write_config(tmp_path, "{}\n"))

    with pytest.raises(FrozenInstanceError):
        setattr(config, "training", TrainingConfig())
    with pytest.raises(FrozenInstanceError):
        setattr(config.training, "n_envs", 1)


def test_paths_and_team_source_paths_resolve_once_from_project_root(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            """
paths:
  data_root: relative-data
environment:
  agent_team_source:
    path: team-pool
""",
        )
    )

    assert config.paths.repository_root.is_absolute()
    assert config.paths.data_root == (Path(__file__).parents[2] / "relative-data").resolve()
    assert (
        config.environment.agent_team_source.path
        == (Path(__file__).parents[2] / "teams" / "team-pool").resolve()
    )


def test_model_config_is_checkpoint_local_and_validated():
    config = ModelConfig.baseline()
    assert config.d_model == 512
    assert config.dim_feedforward == 2048
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(63, 8, 1, 256)


def _resources(tmp_path, *, extra_species: bool = False, base_power: int = 90):
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


def _write_manifest(path, manifest: RuntimeManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")


def test_runtime_manifest_round_trips_one_readable_contract(tmp_path):
    vocab, dex = _resources(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    restored = RuntimeManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert restored == manifest
    assert restored.tensor_abi == TENSOR_ABI
    assert restored.resource_feature_abi == RESOURCE_FEATURE_ABI
    assert restored.action == ACTION_CONTRACT
    assert restored.global_sha256 == manifest.global_sha256


def test_canonical_hash_ignores_object_order_but_not_required_semantics():
    first = {"shape": [31, 10], "dtype": "int64"}
    reordered = {"dtype": "int64", "shape": [31, 10]}
    changed = {"dtype": "int64", "shape": [32, 10]}

    assert canonical_json_sha256(first) == canonical_json_sha256(reordered)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)
    with pytest.raises(ValueError, match="unsupported value"):
        canonical_json_sha256({"scale": 0.5})


def test_vocabulary_expansion_breaks_contract_but_dex_change_does_not(tmp_path):
    vocab, dex = _resources(tmp_path, base_power=90)
    original = current_manifest(vocab_path=vocab, dex_path=dex)

    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": 80}]}))
    rebalanced = current_manifest(vocab_path=vocab, dex_path=dex)
    assert rebalanced.global_sha256 == original.global_sha256
    assert rebalanced.champions_dex_sha256 != original.champions_dex_sha256

    vocab, dex = _resources(tmp_path, extra_species=True, base_power=80)
    expanded = current_manifest(vocab_path=vocab, dex_path=dex)
    assert expanded.global_sha256 != original.global_sha256


def test_global_contract_freezes_payloads_and_bumps_only_the_required_identity():
    contract = active_global_contract()
    with pytest.raises(TypeError):
        contract.payload("actions", "major")["action_count"] = 50  # type: ignore[index]

    changed_minor = contract.with_subsystem_update(
        "resources",
        minor_payload={**contract.payload("resources", "minor"), "showdown_commit": "next"},
    )
    assert changed_minor.global_sha256 == contract.global_sha256
    # Relative to whatever the shipped contract is at, so later bumps do not break this.
    assert (
        changed_minor.subsystem("resources").minor_version
        == contract.subsystem("resources").minor_version + 1
    )

    changed_major = contract.with_subsystem_update(
        "model",
        major_payload={**contract.payload("model", "major"), "tensor_abi": "next"},
    )
    assert changed_major.global_sha256 != contract.global_sha256
    assert (
        changed_major.subsystem("model").major_version
        == contract.subsystem("model").major_version + 1
    )

    # A major bump resets minor. Asserted on a subsystem whose minor is non-zero, so
    # the reset is actually observable rather than trivially already zero.
    bumped = contract.with_subsystem_update(
        "resources",
        major_payload={**contract.payload("resources", "major"), "resource_feature_abi": "next"},
    )
    assert contract.subsystem("resources").minor_version > 0
    assert bumped.subsystem("resources").minor_version == 0


def test_global_contract_rejects_hash_valid_but_malformed_subsystem_payload():
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


def test_artifact_validation_uses_only_the_active_global_contract():
    manifest = active_global_contract()
    artifact = {"global_contract_sha256": manifest.global_sha256}
    assert validate_artifact_runtime_contract(artifact) == manifest
    artifact["global_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        validate_artifact_runtime_contract(artifact)


ACTIVE_CONTRACT = load_runtime_manifest().global_sha256


def _evidence(kind: LabelKind) -> ActionEvidence:
    candidates = {
        LabelKind.EXACT: ((7, 1),),
        LabelKind.PARTIAL: ((7, 1), (8, 1)),
        LabelKind.UNKNOWN: (),
    }[kind]
    return ActionEvidence(
        label_kind=kind,
        candidates=candidates,
        confidence=0.5 if kind is not LabelKind.UNKNOWN else 0.0,
        mask_provenance=MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tags=("fixture",),
    )


def _game_record() -> GameRecord:
    decision = DecisionRecord(
        decision_index=0,
        player=0,
        decision_type=DecisionType.TURN,
        pre_line_index=1,
        post_line_index=3,
        evidence=_evidence(LabelKind.PARTIAL),
    )
    return GameRecord(
        game_id="g1",
        series_id="s1",
        game_number=1,
        protocol_lines=("|start", "|turn|1", "|move|p1a: A|Protect|p1a: A", "|win|alice"),
        ots_payloads=("p1 sheet", "p2 sheet"),
        winner=0,
        end_reason=GameEndReason.NORMAL,
        turns=1,
        decisions=(decision,),
        diagnostics=ReplayDiagnostics(counters={"oov_ids": 0}, parse_errors=()),
    )


def _series_record() -> SeriesRecord:
    return SeriesRecord(
        series_id="s1",
        format_id="gen9championsvgc2026regmbbo3",
        players=("alice", "bob"),
        game_replay_ids=("r1", "r2"),
        game_player_roles=((0, 1), (1, 0)),
        team_hashes=("a" * 64, "b" * 64),
        is_complete=True,
        score=(2, 0),
        grouping_method=GroupingMethod.PARENT_ROOM,
        grouping_confidence=1.0,
    )


def _shard_manifest() -> ShardManifest:
    entry = ShardIndexEntry(
        filename="shard-000.pt", sha256="c" * 64, decisions=10, games=2, series=1, byte_size=1024
    )
    return ShardManifest(
        global_contract_sha256=ACTIVE_CONTRACT,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"max_candidates": 256},
        raw_replays={"game-1": "f" * 64},
        source_series={"series-1": ("game-1",)},
        source_games=1,
        accepted_games=1,
        rejected_games=0,
        artifact_hashes={
            "shard-000.pt": "c" * 64,
        },
    )


def _corpus_entry(packed: str = "packed-team") -> CorpusEntry:
    import hashlib

    return CorpusEntry(
        canonical_hash=hashlib.sha256(packed.encode()).hexdigest(),
        packed=packed,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        split=CorpusSplit.TRAIN,
        usage_count=3,
    )


def _corpus_manifest(entries: tuple[CorpusEntry, ...]) -> TeamCorpusManifest:
    return TeamCorpusManifest(
        global_contract_sha256=ACTIVE_CONTRACT,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"sampling": "uniform_canonical"},
    )


def test_evidence_shapes() -> None:
    assert _evidence(LabelKind.EXACT).exact_action == (7, 1)
    with pytest.raises(ValueError, match="only defined for EXACT"):
        _evidence(LabelKind.PARTIAL).exact_action
    with pytest.raises(ValueError, match="exactly one candidate"):
        ActionEvidence(LabelKind.EXACT, (), 1.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="two or more"):
        ActionEvidence(LabelKind.PARTIAL, ((7, 1),), 0.5, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="no candidates"):
        ActionEvidence(LabelKind.UNKNOWN, ((7, 1),), 0.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="outside"):
        ActionEvidence(LabelKind.EXACT, ((49, 0),), 1.0, MaskProvenance.ORACLE_REQUEST)
    with pytest.raises(ValueError, match="Duplicate"):
        ActionEvidence(LabelKind.PARTIAL, ((7, 1), (7, 1)), 0.5, MaskProvenance.ORACLE_REQUEST)


def test_ir_round_trips() -> None:
    game = _game_record()
    assert GameRecord.from_dict(game.to_dict()) == game
    series = _series_record()
    assert SeriesRecord.from_dict(series.to_dict()) == series
    fetch = FetchIndexEntry(
        replay_id="r1",
        format_id="gen9championsvgc2026regmbbo3",
        source_url="https://replay.pokemonshowdown.com/r1",
        fetched_at="2026-07-17T00:00:00Z",
        http_status=200,
        content_sha256="d" * 64,
        byte_size=100,
    )
    assert FetchIndexEntry.from_dict(fetch.to_dict()) == fetch


def test_ir_rejects_bad_serializations() -> None:
    payload = _game_record().to_dict()
    payload["ir_schema"] = 2
    with pytest.raises(ValueError, match="ir_schema"):
        GameRecord.from_dict(payload)
    payload = _series_record().to_dict()
    del payload["score"]
    payload["bogus"] = 1
    with pytest.raises(ValueError, match=r"missing=\['score'\], unknown=\['bogus'\]"):
        SeriesRecord.from_dict(payload)


def test_ir_validates_construction() -> None:
    with pytest.raises(ValueError, match="ascending"):
        game = _game_record()
        GameRecord.from_dict({**game.to_dict(), "decisions": [game.decisions[0].to_dict()] * 2})
    with pytest.raises(ValueError, match="two wins"):
        SeriesRecord.from_dict({**_series_record().to_dict(), "score": [1, 0]})


def test_observation_specs_are_derived() -> None:
    from p0.model.structured_observation import StructuredObservation

    specs = observation_field_specs()
    assert [spec[0] for spec in specs] == [spec[0] for spec in StructuredObservation._FIELD_SPECS]
    for (name, shape, dtype), (_, base_shape, base_dtype) in zip(
        specs, StructuredObservation._FIELD_SPECS, strict=True
    ):
        assert shape == (-1, *base_shape) and dtype is base_dtype, name
    assert [spec[0] for spec in SHARD_TENSOR_SPECS] == [
        "action_mask",
        "mask_provenance",
        "label_kind",
        "label_confidence",
        "loss_mask",
        "decision_type",
        "exact_action",
        "candidate_values",
        "candidate_offsets",
        "game_offsets",
        "series_offsets",
        "outcome",
    ]


def test_shard_manifest_contract() -> None:
    manifest = _shard_manifest()
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.decisions == 10 and manifest.games == 2 and manifest.series == 1
    assert load_shard_manifest(manifest.to_dict()) == manifest
    with pytest.raises(ValueError, match="incompatible"):
        load_shard_manifest({**manifest.to_dict(), "global_contract_sha256": "0" * 64})
    with pytest.raises(ValueError, match="unknown"):
        load_shard_manifest({**manifest.to_dict(), "runtime_manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="artifact schema"):
        ShardManifest.from_dict({**manifest.to_dict(), "artifact_schema": "p0.replay_shard.v0"})
    with pytest.raises(ValueError, match="observation_schema_version"):
        ShardManifest.from_dict({**manifest.to_dict(), "observation_schema_version": 2})
    # Series context is continuous and rebuilt in process, so the retired
    # symbolic-summary field must not reappear in a manifest.
    assert "series_summary_schema_version" not in manifest.to_dict()
    with pytest.raises(ValueError, match="unknown"):
        ShardManifest.from_dict({**manifest.to_dict(), "series_summary_schema_version": 1})


def test_corpus_manifest_contract() -> None:
    entries = (_corpus_entry("team-a"), _corpus_entry("team-b"))
    manifest = _corpus_manifest(entries)
    assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
    assert load_corpus_manifest(manifest.to_dict()) == manifest
    assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
    with pytest.raises(ValueError, match="does not match the packed team"):
        CorpusEntry(
            canonical_hash="a" * 64,
            packed="team",
            packed_sha256="b" * 64,
            split=CorpusSplit.TRAIN,
            usage_count=1,
        )
    with pytest.raises(ValueError, match="does not match the entries"):
        TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "0" * 64})
    with pytest.raises(ValueError, match="Duplicate corpus entry"):
        _corpus_manifest((entries[0], entries[0]))
    with pytest.raises(ValueError, match="unknown"):
        CorpusEntry.from_dict({**entries[0].to_dict(), "archetype_tags": []})


def test_corpus_source_spec_validates() -> None:
    spec = CorpusSourceSpec(
        corpus_path="teams/corpus_manifest.json",
        corpus_hash="a" * 64,
        format_id="gen9championsvgc2026regmb",
        split=CorpusSplit.TRAIN,
    )
    assert spec.split is CorpusSplit.TRAIN
    with pytest.raises(ValueError, match="split"):
        CorpusSourceSpec(
            corpus_path="x",
            corpus_hash="a" * 64,
            format_id="f",
            split=CorpusSplit.UNSPECIFIED,
        )


def test_model_config_has_only_scaling_fields() -> None:
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


def test_reserved_config_sections(tmp_path) -> None:
    config = load_config("config.yaml.example")
    assert config.bc.batch_decisions == 256
    assert config.bc.gamma == config.training.gamma
    assert config.bc.value_coef == config.training.value_coef
    assert config.corpus.agent_split == "train"
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


def test_active_contract_is_reg_m_b_and_manifest_matches_sources():
    manifest = RuntimeManifest.from_dict(
        json.loads((ROOT / "data/runtime_manifest.json").read_text())
    )
    assert FORMAT.battle_format == "gen9championsvgc2026regmb"
    assert FORMAT.bo3_format == "gen9championsvgc2026regmbbo3"
    assert manifest.battle_format == FORMAT.battle_format
    assert manifest.bo3_format == FORMAT.bo3_format
    assert manifest.action == ACTION_CONTRACT
    assert len(manifest.global_sha256) == 64


def test_runtime_resources_reject_unrecorded_dex_content(tmp_path):
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


def test_every_legal_content_key_resolves():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tokenizer = PokemonTokenizer(vocab)
    for table in ("species", "items", "abilities", "moves"):
        for key in dex["legality"][table]:
            assert tokenizer.resolve(table, key)[1] == "known", (table, key)


def test_mechanics_tables_cover_the_vocab():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    resources = default_runtime_resources()
    assert _load_move_statics(resources).shape[0] == len(vocab["moves"]) + 1
    assert _load_species_statics(resources).shape[0] == len(vocab["species"]) + 1
    mechanic_tags = _load_mechanic_tag_tables(resources)
    assert mechanic_tags["items"].shape[0] == len(vocab["items"]) + 1
    assert mechanic_tags["abilities"].shape[0] == len(vocab["abilities"]) + 1


def test_item_and_ability_mechanics_are_wired_into_encoder():
    encoder = FusedTokenEncoder(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        resources=default_runtime_resources(),
    )
    assert encoder.item_mechanic_proj.in_features == encoder._item_mechanic_tags.shape[1]
    assert encoder.ability_mechanic_proj.in_features == encoder._ability_mechanic_tags.shape[1]
    assert encoder._item_mechanic_tags.count_nonzero() > 0
    assert encoder._ability_mechanic_tags.count_nonzero() > 0


def test_field_namespace_and_coverage_audit_are_present(tmp_path):
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    report = build(
        ROOT / "data/champions_dex.json",
        tmp_path / "vocab.json",
        tmp_path / "manifest.json",
    )
    assert "trickroom" in vocab["fields"]
    assert report["missingLegalContent"] == {}
    assert report["unmappedLegalEffects"] == []


def test_reg_mb_legality_inventory_uses_resolved_showdown_rules():
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    assert "pikachu" in dex["legality"]["species"]
    assert "protect" in dex["legality"]["moves"]
    assert "ababo" not in dex["legality"]["species"]
    assert "berserkgene" not in dex["legality"]["items"]


def test_representative_dump_matches_pinned_showdown_runtime():
    script = r"""
const path = require('node:path');
const {Dex} = require(path.resolve('pokemon-showdown/dist/sim/dex'));
const dex = Dex.mod('champions');
const move = dex.moves.get('protect');
const species = dex.species.get('charizardmegax');
const item = dex.items.get('lifeorb');
const ability = dex.abilities.get('intimidate');
console.log(JSON.stringify({
  move: {basePower: move.basePower, pp: move.pp, target: move.target},
  species: {baseSpecies: species.baseSpecies, requiredItem: species.requiredItem, isMega: species.isMega},
  itemTags: Object.keys(item).filter(key => key.startsWith('on')).sort(),
  abilityTags: Object.keys(ability).filter(key => key.startsWith('on')).sort(),
}));
"""
    oracle = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True))
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tables = {
        name: {entry["id"]: entry for entry in dex[name]}
        for name in ("moves", "species", "items", "abilities")
    }
    assert {key: tables["moves"]["protect"][key] for key in oracle["move"]} == oracle["move"]
    assert {key: tables["species"]["charizardmegax"][key] for key in oracle["species"]} == oracle[
        "species"
    ]
    assert tables["items"]["lifeorb"]["mechanicTags"] == oracle["itemTags"]
    assert tables["abilities"]["intimidate"]["mechanicTags"] == oracle["abilityTags"]


def test_generation_is_deterministic_and_nonlegal_effects_are_reported(tmp_path):
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


def test_unknown_legal_effect_namespace_fails_generation(tmp_path):
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


def test_tokenizer_normalization_and_table_resolution():
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


def test_tokenizer_domain_objects_and_missing_values():
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

    p3 = Pokemon(gen=9, species="charizard")
    p3._ability = "intimidate"
    assert tokenizer.ability_id(p3) == tokenizer.vocab["abilities"]["intimidate"]
    assert tokenizer.ability_id(None) == 0

    p4 = Pokemon(gen=9, species="charizard")
    p4._item = "choicescarf"
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
    assert tokenizer.nature_id(p) == 0  # no nature set yet

    p._nature = "Serious"
    serious_id = tokenizer.nature_id(p)
    assert serious_id == 0
    p._nature = "Bashful"
    assert serious_id == tokenizer.nature_id(p)

    p._nature = "Jolly"
    jolly_id = tokenizer.nature_id(p)
    assert jolly_id > 0
    assert tokenizer.natures_list[jolly_id] == "jolly"

    p._nature = "Adamant"
    adamant_id = tokenizer.nature_id(p)
    assert adamant_id > 0
    assert tokenizer.natures_list[adamant_id] == "adamant"

    p._nature = "unknown_nature"
    assert tokenizer.nature_id(p) == 0


def test_series_perspective_key_rejects_invalid_players():
    with pytest.raises(ValueError, match="canonical_player"):
        SeriesPerspectiveKey("series-1", 2)
    with pytest.raises(ValueError, match="canonical_player"):
        SeriesPerspectiveKey("series-1", True)


def test_token_store_initialization():
    store = SeriesTokenStore(d_model=64, max_games=2)
    assert store.d_model == 64
    assert store.max_games == 2
    assert store._store == {}


def test_token_store_append_and_get():
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


def test_token_store_max_games_truncation():
    store = SeriesTokenStore(d_model=16, max_games=2)

    tokens1 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    tokens2 = torch.randn(SERIES_TOKENS_PER_GAME, 16)
    tokens3 = torch.randn(SERIES_TOKENS_PER_GAME, 16)

    store.append("series-B", tokens1)
    store.append("series-B", tokens2)
    store.append("series-B", tokens3)

    out_tokens, out_mask = store.get_tokens(["series-B"], device=torch.device("cpu"))

    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], tokens2)
    assert torch.allclose(
        out_tokens[0, SERIES_TOKENS_PER_GAME : 2 * SERIES_TOKENS_PER_GAME], tokens3
    )


def test_token_store_batching_and_missing():
    store = SeriesTokenStore(d_model=8)

    t1 = torch.randn(SERIES_TOKENS_PER_GAME, 8)
    store.append("s1", t1)

    out_tokens, out_mask = store.get_tokens(["s1", "s2"], device=torch.device("cpu"))
    assert out_tokens.shape == (2, SERIES_SLOTS, 8)

    assert torch.allclose(out_tokens[0, :SERIES_TOKENS_PER_GAME], t1)
    assert out_mask[0, 0]

    assert torch.all(out_tokens[1] == 0)
    assert not torch.any(out_mask[1])


def test_token_store_training_state_round_trip():
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


def test_token_store_isolates_canonical_player_perspectives():
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


def test_token_store_drop_and_clear():
    store = SeriesTokenStore(d_model=8)
    store.append("s1", torch.randn(SERIES_TOKENS_PER_GAME, 8))

    store.drop("s1")
    out_tokens, out_mask = store.get_tokens(["s1"], device=torch.device("cpu"))
    assert not torch.any(out_mask)

    store.append("s2", torch.randn(SERIES_TOKENS_PER_GAME, 8))
    store.clear()
    out_tokens, out_mask = store.get_tokens(["s2"], device=torch.device("cpu"))
    assert not torch.any(out_mask)


def _mock_vocab() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
        },
        "abilities": {
            "static": 1,
            "blaze": 2,
        },
        "moves": {
            "fakeout": 1,
            "protect": 2,
            "thunderbolt": 3,
            "electroweb": 4,
            "heatwave": 5,
            "solarbeam": 6,
            "weatherball": 7,
        },
    }


def _mock_variant(species: str = "Pikachu") -> TeamRecord:
    if species == "Pikachu":
        members = tuple(
            TeamMember(
                species="Pikachu",
                item="Light Ball",
                ability="Static",
                moves=("Fake Out", "Protect", "Thunderbolt", "Electroweb"),
                nature="Jolly",
            )
            for _ in range(6)
        )
    else:
        members = tuple(
            TeamMember(
                species="Charizard",
                item="Charizardite Y",
                ability="Blaze",
                moves=("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
                nature="Modest",
            )
            for _ in range(6)
        )
    return TeamRecord(
        team=CanonicalTeam(members),
        spreads=tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=TeamMetadata(
            source_series=("test-series",),
            source_replays=("game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=5 if species == "Pikachu" else 3,
        ),
    )


def _mock_validator(
    variants: Sequence[TeamRecord], **kwargs: object
) -> tuple[AdmissionResult, ...]:
    return tuple(
        AdmissionResult(
            team_hash=variant.team.team_hash,
            valid=True,
            packed_team="]".join(
                f"{m.species}|{m.species}|{m.item}|{m.ability}|{','.join(m.moves)}|{m.nature}"
                for m in variant.team.members
            ),
            problems=(),
        )
        for variant in variants
    )


def test_team_source_composition_resolves_corpus(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    contract_hash = current_manifest().global_sha256
    v1 = _mock_variant("Pikachu")
    manifest, _ = build_corpus(
        (v1,),
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    source = _team_source(
        TeamSourceConfig(path=manifest_path),
        corpus_config=CorpusConfig(
            agent_split="train",
        ),
        is_agent=True,
    )
    assert isinstance(source, CorpusTeamSource)
    desc = source.describe()
    assert desc["kind"] == "corpus"
    assert desc["corpus_hash"] == manifest.corpus_hash
    assert desc["split"] == "TRAIN"
    assert desc["sampling"] == "uniform_canonical"


def test_team_source_composition_falls_back_to_file_source(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    team_text = "\n\n".join(
        f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
        for i in range(1, 7)
    )
    (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
    source = _team_source(TeamSourceConfig(path=pool_dir))
    assert isinstance(source, FileTeamSource)


def test_corpus_cli_build_and_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch current_manifest and PokemonTokenizer to use our mock deterministic setup
    monkeypatch.setattr("p0.cli.corpus.current_manifest", lambda: current_manifest())
    monkeypatch.setattr(
        "p0.cli.corpus.PokemonTokenizer.from_file", lambda: PokemonTokenizer(_mock_vocab())
    )
    monkeypatch.setattr("p0.cli.corpus.validate_many", _mock_validator)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    team_text_1 = "\n\n".join(
        "Pikachu @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
        for i in range(1, 7)
    )
    team_text_2 = "\n\n".join(
        "Charizard @ Charizardite Y\nAbility: Blaze\nModest Nature\n- Heat Wave\n- Solar Beam\n- Protect\n- Weather Ball"
        for i in range(1, 7)
    )
    (input_dir / "v1.txt").write_text(team_text_1, encoding="utf-8")
    (input_dir / "v2.txt").write_text(team_text_2, encoding="utf-8")

    output_manifest = tmp_path / "output" / "corpus_manifest.json"
    pool_dir = tmp_path / "pools"

    corpus_main(
        [
            "build",
            "--input",
            str(input_dir),
            "--output",
            str(output_manifest),
            "--pool-dir",
            str(pool_dir),
            "--format-id",
            FORMAT.battle_format,
        ]
    )

    assert output_manifest.is_file()
    assert (pool_dir / "all" / "corpus_manifest.json").is_file()

    captured = capsys.readouterr()
    audit_data = json.loads(captured.out.split("\n")[-2]) if captured.out.strip() else {}
    assert audit_data["admitted_count"] == 2
    assert audit_data["rejected_count"] == 0

    corpus_main(["audit", "--manifest", str(output_manifest)])
    audit_captured = capsys.readouterr()
    re_audit_data = (
        json.loads(audit_captured.out.split("\n")[-2]) if audit_captured.out.strip() else {}
    )
    assert re_audit_data["admitted_count"] == 2


def test_team_source_composition_resolves_directory_manifest(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    contract_hash = current_manifest().global_sha256
    v1 = _mock_variant("Pikachu")
    manifest, _ = build_corpus(
        (v1,),
        tokenizer=tokenizer,
        validator=_mock_validator,
        global_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    pool_dir = tmp_path / "pool_all"
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "corpus_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    source = _team_source(
        TeamSourceConfig(path=pool_dir),
        corpus_config=CorpusConfig(agent_split="train"),
        is_agent=True,
    )
    assert isinstance(source, CorpusTeamSource)


def test_variants_from_showdown_with_dex() -> None:
    mock_dex = {
        "species": [
            {
                "name": "Pikachu",
                "baseStats": {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90},
            }
        ],
        "moves": [
            {"name": "Thunderbolt", "category": "Special"},
            {"name": "Fake Out", "category": "Physical"},
        ],
    }
    showdown_text = "\n\n".join(
        "Pikachu @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Thunderbolt"
        for _ in range(6)
    )
    variants = _variants_from_showdown(showdown_text, dex=mock_dex)
    assert len(variants) == 1
    variant = variants[0]
    assert len(variant.spreads) == 6
    # Verify exact spreads were imputed rather than the default fallback
    assert any(spread != StatPoints(hp=2, spa=32, spe=32) for spread in variant.spreads)
    assert variant.spread_provenance == "imputed"


def _migrated_runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


def _migrated_shard_manifest(contract: str) -> ShardManifest:
    entry = ShardIndexEntry("shard-000.pt", "c" * 64, 10, 2, 1, 100)
    return ShardManifest(
        global_contract_sha256=contract,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"seed": 3},
        raw_replays={"game-1": "f" * 64, "game-2": "e" * 64},
        source_series={"series-1": ("game-1", "game-2")},
        source_games=2,
        accepted_games=2,
        rejected_games=0,
        artifact_hashes={"shard-000.pt": "c" * 64},
    )


def test_runtime_manifest_digest_is_semantic_and_round_trips(tmp_path: Path) -> None:
    vocab, dex = _migrated_runtime_files(tmp_path)
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


def test_shard_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    vocab, dex = _migrated_runtime_files(tmp_path)
    runtime = current_manifest(vocab_path=vocab, dex_path=dex)
    manifest = _migrated_shard_manifest(runtime.global_sha256)
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="default global manifest"):
        load_shard_manifest(
            {**manifest.to_dict(), "global_contract_sha256": "b" * 64}, manifest_path
        )
    with pytest.raises(ValueError, match="source_series"):
        ShardManifest.from_dict({**manifest.to_dict(), "source_series": {"series-1": ("game-1",)}})


def test_corpus_manifest_hash_is_order_independent_but_packed_content_bound() -> None:
    entries = tuple(
        CorpusEntry(
            canonical_hash=hashlib.sha256(f"canonical-{letter}".encode()).hexdigest(),
            packed=f"team-{letter}",
            packed_sha256=hashlib.sha256(f"team-{letter}".encode()).hexdigest(),
            split=CorpusSplit.TRAIN,
            usage_count=index + 1,
        )
        for index, letter in enumerate(("a", "b", "c"))
    )
    manifest = TeamCorpusManifest(
        global_contract_sha256="d" * 64,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"seed": 3},
    )
    assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
    assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
    with pytest.raises(ValueError, match="does not match"):
        TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "e" * 64})


def test_tokenizer_aliases_and_resolution_keep_unknown_zero_distinct_from_known_none() -> None:
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
    tokenizer_instance = PokemonTokenizer({"weathers": {"raindance": 4}, "status": {"brn": 5}})
    assert tokenizer_instance.weathers["rain"] == 4
    assert tokenizer_instance.weathers["rain"] == 4
    assert tokenizer_instance.weathers["unknown-weather"] == 0
    assert tokenizer_instance.status["burn"] == 5
    assert tokenizer_instance.status["unknown-status"] == 0


def test_spread_table_refresh_is_a_minor_contract_change(tmp_path):
    """Refreshing the usage month must not invalidate existing checkpoints."""
    vocab, dex = _resources(tmp_path, base_power=90)
    table = tmp_path / "spread_usage.json"
    table.write_text(json.dumps({"schema": 2, "spreads": {}}))
    original = current_manifest(vocab_path=vocab, dex_path=dex, spread_usage_path=table)

    table.write_text(json.dumps({"schema": 2, "spreads": {}, "month": "2026-08"}))
    refreshed = current_manifest(vocab_path=vocab, dex_path=dex, spread_usage_path=table)

    assert refreshed.spread_usage_sha256 != original.spread_usage_sha256
    # Only the minor identity moves, so a policy trained on the old priors still loads.
    assert refreshed.global_sha256 == original.global_sha256


def test_active_contract_rejects_an_unrecorded_spread_table() -> None:
    """A table edited without a contract bump must fail loudly, not sample stale priors."""
    contract = active_global_contract()
    assert contract.spread_usage_sha256 == sha256_file(
        DEFAULT_PATHS.data_root / "spread_usage.json"
    )
