"""Runtime formats and the model compatibility reference shared by cached tensors."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, NamedTuple

import orjson

from p0.battle.actions import ACT_SIZE
from p0.contracts import RuntimeContract, build_runtime_contract
from p0.paths import DEFAULT_PATHS


class FormatSpec(NamedTuple):
    battle_format: str
    bo3_format: str
    action_size: int


@lru_cache(maxsize=1)
def active_runtime_contract() -> RuntimeContract:
    return build_runtime_contract(DEFAULT_PATHS.data_root)


def validate_artifact_runtime_contract(artifact: Mapping[str, Any]) -> None:
    if artifact.get("runtime_major") != active_runtime_contract().major_sha256:
        raise ValueError("Artifact vocabulary or encoding is incompatible; reconstruct the data")


_source = orjson.loads((DEFAULT_PATHS.data_root / "champions_dex.json").read_bytes())["source"]
FORMAT = FormatSpec(_source["battleFormat"], _source["bo3Format"], ACT_SIZE)


def is_corpus_format_compatible(model_format_id: str, corpus_format_id: str) -> bool:
    return corpus_format_id == model_format_id or (
        model_format_id == FORMAT.bo3_format and corpus_format_id == FORMAT.battle_format
    )
