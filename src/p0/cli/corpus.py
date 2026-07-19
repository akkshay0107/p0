"""Command-line composition roots for corpus construction and auditing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.paths import DEFAULT_PATHS
from p0.teams.corpus import TeamCorpusManifest
from p0.teams.corpus_build import (
    CorpusBuilder,
    SplitPolicy,
    audit_corpus,
    populate_pool_directories,
)
from p0.teams.stat_points import (
    BaseStats,
    ImputationInput,
    Role,
    StatPoints,
    select_candidate,
)
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamVariant, normalize_id
from p0.teams.validation import validate_many


def _variants_from_showdown(
    text: str, dex: Mapping[str, Any] | None = None
) -> tuple[TeamVariant, ...]:
    from p0.teams.source import _PACKER

    members = _PACKER.parse_showdown_team(text)
    if not members or len(members) % 6 != 0:
        raise ValueError(
            f"Showdown text contains {len(members)} members (expected a positive multiple of 6)"
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

    variants: list[TeamVariant] = []
    for i in range(0, len(members), 6):
        team_members = tuple(
            TeamMember(
                species=m.species or m.nickname or "",
                item=m.item or "",
                ability=m.ability or "",
                moves=tuple(m.moves),
                nature=m.nature or "",
                gender=m.gender or "",
                level=m.level if m.level is not None else 100,
            )
            for m in members[i : i + 6]
        )
        team = CanonicalTeam(team_members)
        spreads_list: list[StatPoints] = []
        roles_list: list[Role] = []
        for m in team_members:
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
                    imp_input = ImputationInput(
                        species=m.species,
                        nature=m.nature or "serious",
                        item=m.item,
                        ability=m.ability,
                        moves=moves,
                        move_categories=categories,
                        base_stats=BaseStats.from_mapping(base_mapping),
                        level=m.level,
                    )
                    candidate = select_candidate(imp_input, seed=0)
                    spreads_list.append(candidate.points)
                    roles_list.append(candidate.role)
                    continue
                except (TypeError, ValueError):
                    pass
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
            source_series=("cli-import",),
            source_replays=(),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
            usage_count=1,
            archetype_tags=archetype_tags,
        )
        variants.append(TeamVariant(team=team, spreads=spreads, metadata=metadata))
    return tuple(variants)


def _load_variants(path: Path, dex: Mapping[str, Any] | None = None) -> tuple[TeamVariant, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files: list[Path] = []
    if path.is_dir():
        files.extend(
            sorted(f for f in path.rglob("*") if f.is_file() and not f.name.startswith("."))
        )
    else:
        files.append(path)

    variants: list[TeamVariant] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            for line in file_path.read_text(encoding="utf-8").splitlines():
                line_str = line.strip()
                if line_str:
                    variants.append(TeamVariant.from_dict(json.loads(line_str)))
        elif file_path.suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "team" in item and "spreads" in item:
                        variants.append(TeamVariant.from_dict(item))
            elif isinstance(data, dict) and "team" in data and "spreads" in data:
                variants.append(TeamVariant.from_dict(data))
        else:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                variants.extend(_variants_from_showdown(text, dex=dex))
    return tuple(variants)


def _parser() -> argparse.ArgumentParser:
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
    args = _parser().parse_args(argv)
    if args.command == "build":
        dex_path = DEFAULT_PATHS.data_root / "champions_dex.json"
        dex = json.loads(dex_path.read_text(encoding="utf-8")) if dex_path.is_file() else None
        variants = _load_variants(args.input, dex=dex)
        tokenizer = PokemonTokenizer.from_file()
        split_policy = SplitPolicy(
            ratio_train=args.train_ratio,
            ratio_val=args.val_ratio,
            ratio_test=args.test_ratio,
        )
        builder = CorpusBuilder(
            tokenizer=tokenizer,
            validator=validate_many,
            runtime_contract_sha256=current_manifest().runtime_contract_sha256,
            format_id=args.format_id,
            split_policy=split_policy,
        )
        manifest, audit = builder.build(variants)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.pool_dir is not None:
            populate_pool_directories(manifest, args.pool_dir, reduced_limit=args.reduced_limit)
        print(json.dumps(audit.to_dict(), sort_keys=True))
        return

    if args.command == "audit":
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = TeamCorpusManifest.from_dict(raw)
        audit = audit_corpus(manifest)
        print(json.dumps(audit.to_dict(), sort_keys=True))
        return


if __name__ == "__main__":
    main()
