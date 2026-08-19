"""Command-line composition roots for corpus construction and auditing."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.paths import DEFAULT_PATHS
from p0.teams.corpus import TeamCorpusManifest
from p0.teams.corpus_build import (
    audit_corpus,
    build_corpus,
    write_corpus_manifest,
)
from p0.teams.factory import corpus_manifest_path
from p0.teams.spread_usage import load_spread_table_file
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord, normalize_id
from p0.teams.validation import validate_many

# Fixed so repeated corpus builds from the same export are byte-identical.
CORPUS_SPREAD_SEED = 0

# Last resort when a member has neither a usage prior nor enough moves to fall back.
DEFAULT_SPREAD = StatPoints(hp=2, spa=32, spe=32)


def _variants_from_showdown(
    text: str, dex: Mapping[str, Any] | None = None
) -> tuple[TeamRecord, ...]:
    """Parse a showdown text export into canonical team records.

    Arguments:
        text: Showdown export containing one or more six-member teams.
        dex: Optional dex data used for stat-spread imputation.

    Returns:
        One deduplicated team record per canonical team hash.
    """
    from p0.teams.source import _PACKER

    lines = text.replace("\r\n", "\n").split("\n")
    lines = [line for line in lines if not line.strip().startswith("===")]
    clean_text = "\n".join(lines)

    blocks = [b.strip() for b in clean_text.split("\n\n") if b.strip()]
    if len(blocks) % 6 != 0:
        raise ValueError(f"Showdown text contains {len(blocks)} blocks, not a multiple of 6")

    team_strings = ["\n\n".join(blocks[i : i + 6]) for i in range(0, len(blocks), 6)]
    team_counts: dict[str, int] = {}
    for t in team_strings:
        team_counts[t] = team_counts.get(t, 0) + 1

    unique_teams = list(team_counts.keys())
    unique_text = "\n\n\n".join(unique_teams)
    members = _PACKER.parse_showdown_team(unique_text)

    if not members or len(members) != len(unique_teams) * 6:
        raise ValueError(
            f"Showdown text parsed into {len(members)} members, expected {len(unique_teams) * 6}"
        )

    # Only move categories are needed from the dex now: spreads come from the usage
    # table as Stat Points, so base stats are never converted into stats here.
    move_categories = (
        {
            normalize_id(str(entry.get("id", entry.get("name", "")))).casefold(): str(
                entry.get("category", "")
            )
            for entry in dex.get("moves", ())
            if isinstance(entry, Mapping)
        }
        if isinstance(dex, Mapping)
        else {}
    )

    canonical_dict: dict[str, tuple[CanonicalTeam, int]] = {}
    for i in range(0, len(members), 6):
        team_members = tuple(
            TeamMember(
                species=m.species or m.nickname or "",
                item=m.item or "",
                ability=m.ability or "",
                moves=tuple(m.moves),
                nature=m.nature or "serious",
                gender=m.gender or "",
                level=m.level if m.level is not None else 100,
            )
            for m in members[i : i + 6]
        )
        team = CanonicalTeam(team_members)
        usage = team_counts[unique_teams[i // 6]]

        if team.team_hash in canonical_dict:
            canonical_dict[team.team_hash] = (team, canonical_dict[team.team_hash][1] + usage)
        else:
            canonical_dict[team.team_hash] = (team, usage)

    # Corpus teams sample the usage priors rather than taking the argmax, so generated
    # teams carry the meta's real spread variety. Seeded so builds stay reproducible.
    rng = random.Random(CORPUS_SPREAD_SEED)
    table = load_spread_table_file()

    variants: list[TeamRecord] = []
    for team, usage in canonical_dict.values():
        spreads_list: list[StatPoints] = []
        for m in team.members:
            categories = tuple(
                move_categories.get(normalize_id(move).casefold(), "") for move in m.moves
            )
            estimate = table.sample(m.species, m.nature or "serious", categories, rng)
            if estimate is None:
                # Every team member must carry a playable spread, so unlike the
                # observation path this cannot resolve to UNKNOWN. Reached only when
                # an export lists fewer than two moves of any one category.
                logging.getLogger(__name__).warning(
                    "No spread prior or fallback for %s; using the default spread", m.species
                )
                spreads_list.append(DEFAULT_SPREAD)
                continue
            spreads_list.append(estimate.points)

        spreads = tuple(spreads_list)
        metadata = TeamMetadata(
            source_series=(),
            source_replays=(),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
            usage_count=usage,
        )
        variants.append(TeamRecord(team=team, spreads=spreads, metadata=metadata))

    return tuple(variants)


def _load_variants(path: Path, dex: Mapping[str, Any] | None = None) -> tuple[TeamRecord, ...]:
    """Load variants from a file or directory of showdown exports."""
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files: list[Path] = []
    if path.is_dir():
        files.extend(
            sorted(f for f in path.rglob("*") if f.is_file() and not f.name.startswith("."))
        )
    else:
        files.append(path)

    texts: list[str] = []
    for file_path in files:
        if file_path.suffix == ".txt" or not file_path.suffix:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)

    if not texts:
        return ()

    combined_text = "\n\n".join(texts)
    return _variants_from_showdown(combined_text, dex=dex)


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser for corpus operations."""
    parser = argparse.ArgumentParser(prog="p0-corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file or directory containing team definitions or export strings",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pool directory for the manifest; defaults to the input directory",
    )
    build_parser.add_argument(
        "--format-id",
        default=FORMAT.battle_format,
        help="Battle format ID",
    )
    build_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    build_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    build_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio",
    )

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Team pool directory or corpus manifest path",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Corpus CLI entrypoint."""
    args = _parser().parse_args(argv)

    if args.command == "build":
        dex_path = DEFAULT_PATHS.data_root / "champions_dex.json"
        dex = json.loads(dex_path.read_text(encoding="utf-8")) if dex_path.is_file() else None
        variants = _load_variants(args.input, dex=dex)
        tokenizer = PokemonTokenizer.from_file()

        manifest, audit = build_corpus(
            variants,
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256=current_manifest().global_sha256,
            format_id=args.format_id,
            ratio_train=args.train_ratio,
            ratio_val=args.val_ratio,
            ratio_test=args.test_ratio,
        )
        if not manifest.entries:
            raise ValueError(f"No admitted teams found in input path: {args.input}")
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = args.input if args.input.is_dir() else args.input.parent
        write_corpus_manifest(manifest, output_dir)
        print(json.dumps(audit, sort_keys=True))
        return

    if args.command == "audit":
        manifest_path = corpus_manifest_path(args.path)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = TeamCorpusManifest.from_dict(raw)

        audit = audit_corpus(manifest)
        print(json.dumps(audit, sort_keys=True))
        return


if __name__ == "__main__":
    main()
