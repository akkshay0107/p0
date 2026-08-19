"""Construct team sources from canonical pool paths."""

from __future__ import annotations

from pathlib import Path

import orjson

from p0.teams.corpus import CorpusSourceSpec, CorpusSplit, load_corpus_manifest
from p0.teams.corpus_source import CorpusTeamPool, CorpusTeamSource
from p0.teams.source import FileTeamSource, TeamSource

CORPUS_MANIFEST_NAME = "corpus_manifest.json"


def corpus_manifest_path(path: str | Path) -> Path:
    """Return the manifest path represented by a team pool path."""
    resolved = Path(path)
    if resolved.is_dir():
        return resolved / CORPUS_MANIFEST_NAME
    if resolved.is_file() and resolved.name == CORPUS_MANIFEST_NAME:
        return resolved
    raise ValueError(f"Path is not a team pool directory or corpus manifest: {resolved}")


def build_team_source(
    path: str | Path,
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
    expected_format_id: str | None = None,
) -> TeamSource:
    """Build a corpus source when a pool manifest exists, otherwise a file source."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Team source path not found: {resolved}")

    if resolved.is_dir():
        manifest_path = resolved / CORPUS_MANIFEST_NAME
        if manifest_path.is_file():
            return _build_corpus_source(manifest_path, split, expected_format_id)
        return FileTeamSource(resolved)

    if resolved.is_file():
        if resolved.name == CORPUS_MANIFEST_NAME:
            return _build_corpus_source(resolved, split, expected_format_id)
        return FileTeamSource.from_files((resolved,))

    raise ValueError(f"Unsupported team source path: {resolved}")


def _build_corpus_source(
    manifest_path: Path,
    split: CorpusSplit,
    expected_format_id: str | None,
) -> CorpusTeamSource:
    try:
        manifest = load_corpus_manifest(orjson.loads(manifest_path.read_bytes()))
    except (OSError, UnicodeError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid corpus manifest: {manifest_path}") from exc

    if expected_format_id is not None and manifest.format_id != expected_format_id:
        raise ValueError(
            f"Corpus format mismatch: manifest={manifest.format_id!r}, "
            f"expected={expected_format_id!r}"
        )

    spec = CorpusSourceSpec(
        corpus_path=str(manifest_path),
        corpus_hash=manifest.corpus_hash,
        format_id=manifest.format_id,
        split=split,
    )
    pool = CorpusTeamPool(spec, manifest)
    return CorpusTeamSource(spec, pool=pool)
