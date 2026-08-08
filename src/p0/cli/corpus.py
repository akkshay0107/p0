"""Command-line composition roots for corpus construction and auditing."""

from __future__ import annotations

import argparse
import json
import logging
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
    populate_pool_directories,
)
from p0.teams.stat_points import (
    BaseStats,
    Role,
    StatPoints,
    select_candidate,
)
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord, normalize_id
from p0.teams.validation import validate_many


def _variants_from_showdown(
    text: str, dex: Mapping[str, Any] | None = None
) -> tuple[TeamRecord, ...]:
    """Parse a showdown text export into canonical team records.

    Arguments:
        text: Showdown export containing one or more six-member teams.
        dex: Optional dex data used for stat-spread imputation and role labels.

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

    species_by_id = (
        {
            normalize_id(str(entry.get("id", entry.get("name", "")))).casefold(): entry
            for entry in dex.get("species", ())
            if isinstance(entry, Mapping)
        }
        if isinstance(dex, Mapping)
        else {}
    )

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

    variants: list[TeamRecord] = []
    for team, usage in canonical_dict.values():
        spreads_list: list[StatPoints] = []
        roles_list: list[Role] = []
        for m in team.members:
            species_entry = species_by_id.get(normalize_id(m.species).casefold())
            base_mapping = (
                species_entry.get("baseStats") if isinstance(species_entry, Mapping) else None
            )
            if isinstance(base_mapping, Mapping):
                try:
                    moves = m.moves
                    categories = tuple(
                        move_categories.get(normalize_id(move).casefold(), "") for move in moves
                    )
                    base_stats = BaseStats.from_mapping(base_mapping)
                    candidate = select_candidate(
                        nature=m.nature or "serious",
                        moves=moves,
                        move_categories=categories,
                        item=m.item,
                        ability=m.ability,
                        base_stats=base_stats,
                        seed=0,
                    )
                    spreads_list.append(candidate.points)
                    roles_list.append(candidate.role)
                    continue
                except (TypeError, ValueError) as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to impute stats for %s: %s", m.species, exc
                    )
            spreads_list.append(StatPoints(hp=2, spa=32, spe=32))

        spreads = tuple(spreads_list)
        if roles_list:
            if any(role == Role.TRICK_ROOM for role in roles_list):
                archetype_tags = ("trick-room", "imputed")
            elif any(role == Role.SPEED_CONTROL for role in roles_list):
                archetype_tags = ("speed-control", "imputed")
            elif any(role in (Role.PHYSICAL, Role.SPECIAL, Role.MIXED) for role in roles_list):
                archetype_tags = ("offense", "imputed")
            else:
                archetype_tags = ("balance", "imputed")
        else:
            archetype_tags = ("balance",)

        metadata = TeamMetadata(
            source_series=(),
            source_replays=(),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
            usage_count=usage,
            archetype_tags=archetype_tags,
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
        "--output",
        type=Path,
        required=True,
        help="Output manifest file path",
    )
    build_parser.add_argument(
        "--pool-dir",
        type=Path,
        default=None,
        help="Optional directory to populate split pools",
    )
    build_parser.add_argument(
        "--reduced-limit",
        type=int,
        default=64,
        help="Number of top usage entries to retain in the reduced pool",
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
        "--manifest",
        type=Path,
        required=True,
        help="Corpus manifest file path",
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.pool_dir is not None:
            populate_pool_directories(manifest, args.pool_dir, reduced_limit=args.reduced_limit)
        print(json.dumps(audit, sort_keys=True))
        return

    if args.command == "audit":
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = TeamCorpusManifest.from_dict(raw)

        audit = audit_corpus(manifest)
        print(json.dumps(audit, sort_keys=True))
        return


if __name__ == "__main__":
    main()
