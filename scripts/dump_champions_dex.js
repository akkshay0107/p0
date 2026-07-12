#!/usr/bin/env node

/**
 * Emit the authoritative Champions content used by the pinned Showdown checkout.
 *
 * The output deliberately contains normalized, JSON-safe fields rather than the
 * simulator's runtime objects.  Run from the repository root:
 *
 *   node scripts/dump_champions_dex.js
 */

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const showdownRoot = path.resolve(__dirname, "..", "pokemon-showdown");
const { Dex } = require(path.join(showdownRoot, "dist", "sim", "dex"));
const { TeamValidator } = require(
  path.join(showdownRoot, "dist", "sim", "team-validator"),
);
const { toID } = Dex;

const SHOWDOWN_COMMIT = "8282e63102fa824fd2f7472778ec09793ceb7cac";
const FORMAT_IDS = [
  "gen9championsvgc2026regmb",
  "gen9championsvgc2026regmbbo3",
];

function sortedObject(value) {
  if (Array.isArray(value)) return value.map(sortedObject);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, sortedObject(value[key])]),
  );
}

function pick(value, fields) {
  const result = {};
  for (const field of fields) {
    if (value[field] !== undefined) result[field] = value[field];
  }
  return sortedObject(result);
}

function normalizedSpecies(species) {
  return pick(species, [
    "id",
    "name",
    "num",
    "gen",
    "isNonstandard",
    "tier",
    "doublesTier",
    "baseSpecies",
    "forme",
    "baseForme",
    "formeOrder",
    "otherFormes",
    "battleOnly",
    "changesFrom",
    "requiredItem",
    "requiredItems",
    "abilities",
    "types",
    "addedType",
    "baseStats",
    "bst",
    "weightkg",
    "heightm",
    "gender",
    "genderRatio",
    "isMega",
    "isPrimal",
  ]);
}

function normalizedMove(move) {
  return pick(move, [
    "id",
    "name",
    "num",
    "gen",
    "isNonstandard",
    "type",
    "target",
    "basePower",
    "accuracy",
    "priority",
    "category",
    "pp",
    "noPPBoosts",
    "flags",
    "spreadHit",
    "selfSwitch",
    "volatileStatus",
    "status",
    "secondary",
    "secondaries",
  ]);
}

function normalizedItem(item) {
  const result = pick(item, [
    "id",
    "name",
    "num",
    "gen",
    "isNonstandard",
    "desc",
    "shortDesc",
    "fling",
    "megaStone",
    "megaEvolves",
    "onPlate",
    "onDrive",
  ]);
  result.mechanicTags = Object.keys(item)
    .filter((key) => key.startsWith("on"))
    .sort();
  return result;
}

function normalizedAbility(ability) {
  const result = pick(ability, [
    "id",
    "name",
    "num",
    "gen",
    "isNonstandard",
    "desc",
    "shortDesc",
  ]);
  result.mechanicTags = Object.keys(ability)
    .filter((key) => key.startsWith("on"))
    .sort();
  return result;
}

function normalizedNature(nature) {
  return pick(nature, ["name", "id", "plus", "minus"]);
}

function useful(entries, normalizer) {
  return entries
    .filter((entry) => entry.exists !== false)
    .map(normalizer)
    .filter((entry) => entry.id || entry.name)
    .sort((a, b) =>
      String(a.id || a.name).localeCompare(String(b.id || b.name)),
    );
}

const dex = Dex.mod("champions");
const validator = new TeamValidator(FORMAT_IDS[0]);
const formats = Object.fromEntries(
  FORMAT_IDS.map((id) => {
    const format = Dex.formats.get(id);
    const metadata = pick(format, [
      "id",
      "name",
      "mod",
      "gameType",
      "ruleset",
      "banlist",
      "restricted",
      "bestOfDefault",
    ]);
    metadata.resolvedRules = [
      ...Dex.formats.getRuleTable(format).keys(),
    ].sort();
    return [id, metadata];
  }),
);

const dummySet = {
  name: "Coverage audit",
  species: "Pikachu",
  ability: "Static",
  item: "",
  moves: ["Protect"],
  nature: "Serious",
};
const passes = (problem) => !problem;
const legalSpecies = dex.species
  .all()
  .filter(
    (species) =>
      species.exists !== false &&
      passes(
        validator.checkSpecies(
          { ...dummySet, name: species.name, species: species.name },
          species,
          species,
          {},
        ),
      ),
  );
const allowedMoves = dex.moves
  .all()
  .filter(
    (move) =>
      move.exists !== false && passes(validator.checkMove(dummySet, move, {})),
  );
const legalItems = dex.items
  .all()
  .filter(
    (item) =>
      item.exists !== false && passes(validator.checkItem(dummySet, item, {})),
  );
const allowedAbilities = dex.abilities
  .all()
  .filter(
    (ability) =>
      ability.exists !== false &&
      passes(validator.checkAbility(dummySet, ability, {})),
  );
const legalNatures = dex.natures
  .all()
  .filter(
    (nature) =>
      nature.exists !== false &&
      passes(validator.checkNature(dummySet, nature, {})),
  );
const reachableMoveIds = new Set(["struggle", "recharge"]);
const reachableAbilityIds = new Set();
for (const species of legalSpecies) {
  let movePool;
  try {
    movePool = dex.species.getMovePool(species.id);
  } catch {
    movePool = dex.species.getMovePool(toID(species.baseSpecies));
  }
  for (const move of movePool) reachableMoveIds.add(move);
  for (const ability of Object.values(species.abilities || {}))
    reachableAbilityIds.add(toID(ability));
}
const legalMoves = allowedMoves.filter((move) => reachableMoveIds.has(move.id));
const legalAbilities = allowedAbilities.filter((ability) =>
  reachableAbilityIds.has(ability.id),
);
const legality = {
  species: legalSpecies.map((entry) => entry.id).sort(),
  moves: legalMoves.map((entry) => entry.id).sort(),
  items: legalItems.map((entry) => entry.id).sort(),
  abilities: legalAbilities.map((entry) => entry.id).sort(),
  natures: legalNatures.map((entry) => entry.id).sort(),
};

const transformations = dex.species
  .all()
  .filter(
    (species) =>
      species.isMega ||
      species.isPrimal ||
      species.battleOnly ||
      species.changesFrom,
  )
  .map((species) =>
    pick(species, [
      "id",
      "baseSpecies",
      "battleOnly",
      "changesFrom",
      "requiredItem",
      "requiredItems",
      "isMega",
      "isPrimal",
    ]),
  )
  .sort((a, b) => a.id.localeCompare(b.id));

const protocolEffectIds = [
  ...new Set([
    ...Object.keys(dex.data.Conditions),
    ...dex.moves
      .all()
      .filter((move) => move.condition)
      .map((move) => move.id),
    ...dex.items
      .all()
      .filter((item) => item.condition)
      .map((item) => item.id),
    ...dex.abilities
      .all()
      .filter((ability) => ability.condition)
      .map((ability) => ability.id),
  ]),
].sort();

const effectKeys = {
  volatileStatus: "effect",
  status: "status",
  sideCondition: "side_condition",
  weather: "weather",
  terrain: "field",
  pseudoWeather: "field",
};
const legalProtocolEffects = Object.fromEntries(
  ["effect", "status", "side_condition", "weather", "field"].map((family) => [
    family,
    new Set(),
  ]),
);
function collectEffectRefs(value, seen = new Set()) {
  if (
    !value ||
    (typeof value !== "object" && typeof value !== "function") ||
    seen.has(value)
  )
    return;
  seen.add(value);
  for (const [key, child] of Object.entries(value)) {
    const family = effectKeys[key];
    if (family && typeof child === "string")
      legalProtocolEffects[family].add(toID(child));
    if (child && typeof child === "object") collectEffectRefs(child, seen);
  }
  for (const fn of Object.values(value).filter(
    (child) => typeof child === "function",
  )) {
    const source = Function.prototype.toString.call(fn);
    const patterns = [
      ["effect", /addVolatile\(['"]([^'"]+)/g],
      ["side_condition", /addSideCondition\(['"]([^'"]+)/g],
      ["weather", /setWeather\(['"]([^'"]+)/g],
      ["field", /(?:setTerrain|addPseudoWeather)\(['"]([^'"]+)/g],
      ["status", /(?:setStatus|trySetStatus)\(['"]([^'"]+)/g],
    ];
    for (const [family, pattern] of patterns) {
      for (const match of source.matchAll(pattern))
        legalProtocolEffects[family].add(toID(match[1]));
    }
  }
}
for (const entry of [...legalMoves, ...legalItems, ...legalAbilities])
  collectEffectRefs(entry);
const serializedLegalEffects = Object.fromEntries(
  Object.entries(legalProtocolEffects).map(([family, values]) => [
    family,
    [...values].sort(),
  ]),
);

const output = {
  schemaVersion: 2,
  source: {
    repository: "https://github.com/smogon/pokemon-showdown",
    commit: SHOWDOWN_COMMIT,
    mod: "champions",
    formats,
  },
  species: useful(dex.species.all(), normalizedSpecies),
  moves: useful(dex.moves.all(), normalizedMove),
  items: useful(dex.items.all(), normalizedItem),
  abilities: useful(dex.abilities.all(), normalizedAbility),
  natures: useful(dex.natures.all(), normalizedNature),
  legality,
  transformations,
  protocolEffects: protocolEffectIds,
  legalProtocolEffects: serializedLegalEffects,
};

const serialized = `${JSON.stringify(sortedObject(output), null, 2)}\n`;
const outputPath = path.resolve(__dirname, "..", "data", "champions_dex.json");
fs.writeFileSync(outputPath, serialized, "utf8");

console.log(
  JSON.stringify(
    {
      output: path.relative(process.cwd(), outputPath),
      sha256: crypto.createHash("sha256").update(serialized).digest("hex"),
      counts: Object.fromEntries(
        Object.entries(output)
          .filter(([, value]) => Array.isArray(value))
          .map(([key, value]) => [key, value.length]),
      ),
    },
    null,
    2,
  ),
);
