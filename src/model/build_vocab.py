import json
import re
from pathlib import Path


def normalize_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def infer_mega_form(species: str, item: str) -> str | None:
    """
    Determine the Showdown form ID for Mega or Primal transformations.
    Handles X/Y suffixes (e.g., Charizardite Y -> charizardmegay) and
    Primal orbs.

    Args:
        species: The base species name.
        item: The held item name.

    Returns:
        The form ID string if a transformation is inferred, else None.
    """
    species_id = normalize_id(species)
    item_id = normalize_id(item)

    if item_id in {"redorb", "blueorb"}:
        return f"{species_id}primal"

    if "ite" in item_id:
        suffix = ""
        if item_id.endswith("x"):
            suffix = "x"
        elif item_id.endswith("y"):
            suffix = "y"

        # heuristic: stone contains species name or vice versa
        base_item = item_id.replace("ite", "")
        if suffix:
            base_item = base_item[:-1]

        if base_item in species_id or species_id in base_item:
            return f"{species_id}mega{suffix}"

        # fallback for weird stones (e.g. Mewtwonite -> mewtwo)
        if species_id in item_id:
            return f"{species_id}mega{suffix}"

    return None


def extract_species(raw: str) -> str:
    """
    Extract the species name from a Showdown-formatted line.
    Handles "Nickname (Species)" and "Species (M/F)" patterns.

    Args:
        raw: The raw species/nickname line.

    Returns:
        The extracted species name.
    """
    raw = raw.strip()
    m = re.search(r"\(([^)]+)\)", raw)
    if m:
        # ignore gender markers
        if m.group(1).lower() in {"m", "f"}:
            return raw.split("(")[0].strip()
        return m.group(1).strip()
    return raw


def main():
    script_dir = Path(__file__).resolve().parent.parent
    teams_dir = script_dir.parent / "teams"
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    species_set = set()
    items_set = set()
    abilities_set = set()
    moves_set = set()

    # hard coded (cant infer directly from teamlist) sets below
    volatiles_set = {
        "encore",
        "disable",
        "leechseed",
        "confusion",
        "throatchop",
    }
    status_set = {"paralysis", "poison", "burn", "sleep", "freeze", "toxic"}
    side_conditions_set = {"tailwind", "auroraveil"}
    weathers_set = {"rain", "sun", "sand", "snow"}
    trickroom_set = {"trickroom"}
    # terrain_set needed for later once terrain setters get added to team list

    moves_set.update({"struggle", "recharge"})

    for txt_file in teams_dir.glob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            content = f.read()

        sets = [s.strip() for s in content.split("\n\n") if s.strip()]

        for s in sets:
            lines = [line.strip() for line in s.split("\n") if line.strip()]
            if not lines:
                continue

            header = lines[0]
            parts = header.split("@")
            species_raw = parts[0].strip()
            species_name = extract_species(species_raw)
            species_id = normalize_id(species_name)
            species_set.add(species_id)

            item_id = ""
            if len(parts) > 1:
                item = parts[1].strip()
                item_id = normalize_id(item)
                items_set.add(item_id)
                mega_form = infer_mega_form(species_name, item)
                if mega_form:
                    species_set.add(normalize_id(mega_form))

            ability_id = ""
            current_moves = []

            for line in lines[1:]:
                if line.startswith("Ability: "):
                    ability_id = normalize_id(line[len("Ability: ") :].strip())
                    abilities_set.add(ability_id)
                elif line.startswith("- "):
                    move = normalize_id(line[2:].strip())
                    moves_set.add(move)
                    current_moves.append(move)

    vocab = {
        "species": {name: idx + 1 for idx, name in enumerate(sorted(species_set))},
        "items": {name: idx + 1 for idx, name in enumerate(sorted(items_set))},
        "abilities": {name: idx + 1 for idx, name in enumerate(sorted(abilities_set))},
        "moves": {name: idx + 1 for idx, name in enumerate(sorted(moves_set))},
        "volatiles": {name: idx + 1 for idx, name in enumerate(sorted(volatiles_set))},
        "status": {name: idx + 1 for idx, name in enumerate(sorted(status_set))},
        "side_conditions": {name: idx + 1 for idx, name in enumerate(sorted(side_conditions_set))},
        "weathers": {name: idx + 1 for idx, name in enumerate(sorted(weathers_set))},
        "trickroom": {name: idx + 1 for idx, name in enumerate(sorted(trickroom_set))},
        "categories": {
            "physical": 1,
            "special": 2,
            "status": 3,
        },
        "types": {
            name: idx + 1
            for idx, name in enumerate(
                [
                    "normal",
                    "fire",
                    "water",
                    "electric",
                    "grass",
                    "ice",
                    "fighting",
                    "poison",
                    "ground",
                    "flying",
                    "psychic",
                    "bug",
                    "rock",
                    "ghost",
                    "dragon",
                    "dark",
                    "steel",
                    "fairy",
                ]
            )
        },
    }

    out_path = data_dir / "vocab.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"Vocabulary written to {out_path}")
    print(
        f"Species: {len(vocab['species'])}, "
        f"Items: {len(vocab['items'])}, "
        f"Abilities: {len(vocab['abilities'])}, "
        f"Moves: {len(vocab['moves'])}"
    )


if __name__ == "__main__":
    main()
