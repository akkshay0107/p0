#!/usr/bin/env node

/**
 * Emit the authoritative Champions content used by the pinned Showdown checkout.
 *
 * The output deliberately contains normalized, JSON-safe fields rather than the
 * simulator's runtime objects.  Run from the repository root:
 *
 *   node scripts/dump_champions_dex.js
 */

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const showdownRoot = path.resolve(__dirname, '..', 'pokemon-showdown');
const {Dex} = require(path.join(showdownRoot, 'dist', 'sim', 'dex'));
const {toID} = Dex;

const SHOWDOWN_COMMIT = '8282e63102fa824fd2f7472778ec09793ceb7cac';
const FORMAT_IDS = [
  'gen9championsvgc2026regmb',
  'gen9championsvgc2026regmbbo3',
];

function sortedObject(value) {
  if (Array.isArray(value)) return value.map(sortedObject);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.keys(value).sort().map(key => [key, sortedObject(value[key])]));
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
    'id', 'name', 'num', 'gen', 'isNonstandard', 'tier', 'doublesTier',
    'baseSpecies', 'forme', 'baseForme', 'formeOrder', 'otherFormes',
    'battleOnly', 'changesFrom', 'requiredItem', 'requiredItems',
    'abilities', 'types', 'addedType', 'baseStats', 'bst', 'weightkg',
    'heightm', 'gender', 'genderRatio', 'isMega', 'isPrimal',
  ]);
}

function normalizedMove(move) {
  return pick(move, [
    'id', 'name', 'num', 'gen', 'isNonstandard', 'type', 'target', 'basePower',
    'accuracy', 'priority', 'category', 'pp', 'noPPBoosts', 'flags',
    'spreadHit', 'selfSwitch', 'volatileStatus', 'status', 'secondary',
    'secondaries',
  ]);
}

function normalizedItem(item) {
  return pick(item, [
    'id', 'name', 'num', 'gen', 'isNonstandard', 'desc', 'shortDesc',
    'fling', 'megaStone', 'megaEvolves', 'onPlate', 'onDrive',
  ]);
}

function normalizedAbility(ability) {
  return pick(ability, [
    'id', 'name', 'num', 'gen', 'isNonstandard', 'desc', 'shortDesc',
  ]);
}

function normalizedNature(nature) {
  return pick(nature, ['name', 'id', 'plus', 'minus']);
}

function useful(entries, normalizer) {
  return entries
    .filter(entry => entry.exists !== false)
    .map(normalizer)
    .filter(entry => entry.id || entry.name)
    .sort((a, b) => String(a.id || a.name).localeCompare(String(b.id || b.name)));
}

const dex = Dex.mod('champions');
const formats = Object.fromEntries(FORMAT_IDS.map(id => {
  const format = Dex.formats.get(id);
  return [id, pick(format, ['id', 'name', 'mod', 'gameType', 'ruleset', 'banlist', 'restricted'])];
}));

const output = {
  schemaVersion: 1,
  source: {
    repository: 'https://github.com/smogon/pokemon-showdown',
    commit: SHOWDOWN_COMMIT,
    mod: 'champions',
    formats,
  },
  species: useful(dex.species.all(), normalizedSpecies),
  moves: useful(dex.moves.all(), normalizedMove),
  items: useful(dex.items.all(), normalizedItem),
  abilities: useful(dex.abilities.all(), normalizedAbility),
  natures: useful(dex.natures.all(), normalizedNature),
};

const serialized = `${JSON.stringify(sortedObject(output), null, 2)}\n`;
const outputPath = path.resolve(__dirname, '..', 'data', 'champions_dex.json');
fs.writeFileSync(outputPath, serialized, 'utf8');

console.log(JSON.stringify({
  output: path.relative(process.cwd(), outputPath),
  sha256: crypto.createHash('sha256').update(serialized).digest('hex'),
  counts: Object.fromEntries(Object.entries(output).filter(([, value]) => Array.isArray(value)).map(([key, value]) => [key, value.length])),
}, null, 2));
