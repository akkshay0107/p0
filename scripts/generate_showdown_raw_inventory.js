/* Build the raw Showdown protocol emission inventory from the pinned TS tree. */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const ts = require('../pokemon-showdown/node_modules/typescript');

const ROOT = path.resolve(__dirname, '..', 'pokemon-showdown');
const EXPECTED = '8282e63102fa824fd2f7472778ec09793ceb7cac';
const DIRECTORIES = ['config/formats.ts', 'data/mods/champions', 'sim', 'data'];
const DATA_FILES = new Set(['moves.ts', 'abilities.ts', 'items.ts', 'conditions.ts']);
const CALLS = new Set(['add', 'addMove', 'addSplit', 'attrLastMove', 'retargetLastMove']);

function filesUnder(relative) {
  const full = path.join(ROOT, relative);
  if (fs.statSync(full).isFile()) return [full];
  const result = [];
  for (const entry of fs.readdirSync(full, {withFileTypes: true})) {
    const child = path.join(full, entry.name);
    if (entry.isDirectory()) result.push(...filesUnder(path.relative(ROOT, child)));
    else if (entry.name.endsWith('.ts')) result.push(child);
  }
  return result;
}

function sourceFiles() {
  const all = [];
  for (const item of DIRECTORIES) {
    if (item === 'data') {
      for (const name of DATA_FILES) {
        const file = path.join(ROOT, 'data', name);
        if (fs.existsSync(file)) all.push(file);
      }
    }
    else all.push(...filesUnder(item));
  }
  const room = path.join(ROOT, 'server/room-battle.ts');
  if (fs.existsSync(room)) all.push(room);
  return [...new Set(all)].sort();
}

function enclosingName(node) {
  for (let current = node.parent; current; current = current.parent) {
    if (ts.isMethodDeclaration(current) || ts.isFunctionDeclaration(current)) {
      return current.name ? current.name.getText() : '<anonymous>';
    }
  }
  return '<module>';
}

function enclosingKey(node) {
  for (let current = node.parent; current; current = current.parent) {
    if (ts.isPropertyAssignment(current) || ts.isMethodDeclaration(current)) {
      return current.name ? current.name.getText() : null;
    }
  }
  return null;
}

function dataOwner(node, source) {
  const keys = [];
  for (let current = node.parent; current; current = current.parent) {
    if ((ts.isPropertyAssignment(current) || ts.isMethodDeclaration(current)) && current.name) {
      keys.push(current.name.getText(source).replace(/^['"]|['"]$/g, ''));
    }
  }
  return keys.length ? keys[keys.length - 1] : null;
}

function scan(file) {
  const text = fs.readFileSync(file, 'utf8');
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const entries = [];
  function visit(node) {
    if (ts.isCallExpression(node)) {
      const callee = node.expression;
      const name = ts.isIdentifier(callee) ? callee.text
        : ts.isPropertyAccessExpression(callee) ? callee.name.text : '';
      const receiver = ts.isPropertyAccessExpression(callee) ? callee.expression.getText(source) : null;
      if (CALLS.has(name) && (receiver === null || receiver === 'this' || receiver.endsWith('.battle') || receiver.endsWith('.room') || receiver === 'room' || receiver === 'battle')) {
        const args = node.arguments.map(arg => arg.getText(source));
        const first = node.arguments[0];
        const literal = first && ts.isStringLiteralLike(first) ? first.text : null;
        const wireTag = literal && literal.startsWith('|') ? literal.split('|')[1] : literal;
        entries.push({
          path: path.relative(ROOT, file).replaceAll(path.sep, '/'),
          line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
          call: name,
          tag: wireTag,
          dynamic_tag_expression: literal === null && first ? first.getText(source) : null,
          arguments: args,
          enclosing: enclosingName(node),
          enclosing_key: enclosingKey(node),
          data_owner: file.includes('/data/') ? dataOwner(node, source) : null,
          receiver,
          expression: node.getText(source),
        });
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  return {path: path.relative(ROOT, file).replaceAll(path.sep, '/'), sha256: crypto.createHash('sha256').update(text).digest('hex'), entries};
}

function volatileTable() {
  const file = path.join(ROOT, 'data/conditions.ts');
  const text = fs.readFileSync(file, 'utf8');
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const {Dex} = require('../pokemon-showdown/dist/sim/dex');
  const dex = Dex.mod('champions');
  const rows = [];
  const sourceLines = new Map();
  function visit(node) {
    if (ts.isPropertyAssignment(node) && node.name && node.initializer &&
        (ts.isObjectLiteralExpression(node.initializer) || ts.isObjectLiteralExpression(node.initializer))) {
      const id = node.name.getText(source).replace(/^['"]|['"]$/g, '');
      const initializer = node.initializer.getText(source);
      sourceLines.set(id, source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  const keyLocation = id => {
    for (const filename of ['moves.ts', 'items.ts', 'abilities.ts', 'conditions.ts']) {
      const candidate = path.join(ROOT, 'data', filename);
      const lines = fs.readFileSync(candidate, 'utf8').split('\n');
      const index = lines.findIndex(line => new RegExp(`^\\s+${id}:\\s*[<{]`).test(line));
      if (index >= 0) return {path: `data/${filename}`, line: index + 1};
    }
    return {path: 'data/conditions.ts', line: null};
  };
  const dexCatalog = JSON.parse(fs.readFileSync(path.join(ROOT, '..', 'data/champions_dex.json'), 'utf8'));
  const legalEffects = new Set(dexCatalog.legalProtocolEffects.effect || []);
  const legalOwners = new Map(['moves', 'items', 'abilities'].map(kind => [kind,
    new Set((dexCatalog.legality?.[kind] || []).map(entry => typeof entry === 'string' ? entry : entry.id))]));
  const effectIds = new Set([...Object.keys(dex.data.Conditions), ...legalEffects]);
  const owners = [];
  for (const [kind, api] of [['moves', dex.moves], ['items', dex.items], ['abilities', dex.abilities]]) {
    for (const entry of api.all()) {
      if (entry.condition) {
        const id = entry.id;
        effectIds.add(id);
        owners.push({id, kind, owner: entry.id});
      }
    }
  }
  for (const id of effectIds) {
    const condition = dex.conditions.get(id);
    const owner = owners.find(item => item.id === condition.id);
    rows.push({id: condition.id, exists: condition.exists === true,
      noCopy: condition.exists === true && condition.noCopy === true,
      onCopy: condition.exists === true && typeof condition.onCopy === 'function',
      reachable: legalEffects.has(condition.id) || Boolean(owner && legalOwners.get(owner.kind).has(owner.owner)),
      owner: owner || null,
      source: owner ? keyLocation(owner.owner) : keyLocation(id)});
  }
  for (const id of ['typechange', 'typeadd']) {
    const condition = dex.conditions.get(id);
    rows.push({id, exists: condition.exists === true, noCopy: false, onCopy: false,
      source: {path: 'data/conditions.ts', line: null}});
  }
  return rows;
}

function main() {
  const gitdir = path.resolve(ROOT, fs.readFileSync(path.join(ROOT, '.git'), 'utf8').trim().replace('gitdir: ', ''));
  const head = fs.readFileSync(path.join(gitdir, 'HEAD'), 'utf8').trim();
  const revision = head.startsWith('ref: ')
    ? fs.readFileSync(path.join(gitdir, head.slice(5)), 'utf8').trim()
    : head;
  if (revision !== EXPECTED) throw new Error(`Showdown revision drift: ${revision}`);
  const files = sourceFiles().map(scan);
  const entries = files.flatMap(file => file.entries);
  const impossible = new Set(['debug', '-candynamax', '-center', '-terastallize', '-zpower', '-primal', '-burst', '-swapsideconditions', '-combine', '-waiting', '-notarget', '-nothing', '-eat']);
  const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, '..', 'data/champions_dex.json'), 'utf8'));
  const legal = new Map(['moves', 'items', 'abilities'].map(kind => [kind, new Set(catalog.legality?.[kind] || [])]));
  const dex = require('../pokemon-showdown/dist/sim/dex').Dex.mod('champions');
  const dynamicActivationEffects = [];
  // Battle#checkMoveMakesContact emits the active move's fullname when the
  // source asks for the announcement. Expand that dynamic site from the
  // legal move catalog, restricted to moves that can actually make contact.
  for (const id of legal.get('moves')) {
    const move = dex.moves.get(id);
    if (move.exists && move.flags?.contact) dynamicActivationEffects.push(move.name);
  }
  // Conditions#partiallytrapped emits the source effect's move name.
  for (const id of legal.get('moves')) {
    const move = dex.moves.get(id);
    if (move.exists && (move.volatileStatus === 'partiallytrapped' || move.status === 'partiallytrapped')) {
      dynamicActivationEffects.push(`move: ${move.name}`);
    }
  }
  for (const entry of entries) {
    const ownerKind = ['moves', 'items', 'abilities'].find(kind => entry.path === `data/${kind}.ts`);
    const illegalOwner = ownerKind && entry.data_owner && !legal.get(ownerKind).has(entry.data_owner);
    const inactiveFormat = entry.path === 'config/formats.ts' && !['gen9championsvgc2026regmb', 'gen9championsvgc2026regmbbo3'].includes(entry.data_owner);
    const dynamicTags = entry.dynamic_tag_expression === "isDrag ? 'drag' : 'switch'" ? ['drag', 'switch']
      : entry.dynamic_tag_expression === "this.battle.gen >= 5 ? '-fail' : '-notarget'" ? ['-fail', '-notarget']
      : entry.dynamic_tag_expression?.startsWith('teampreview') ? ['teampreview']
      : entry.path.endsWith('/rulesets.ts') && entry.dynamic_tag_expression === '${buf}' ? ['clearpoke', 'poke', 'rule']
      : entry.dynamic_tag_expression === 'msg' ? ['-boost', '-unboost', '-setboost']
      : [];
    entry.resolved_tags = dynamicTags;
    const nonProtocolCall = entry.call === 'attrLastMove' || entry.call === 'retargetLastMove';
    entry.reachability = nonProtocolCall
      ? 'excluded'
      : entry.tag === null
      ? (dynamicTags.length ? 'reachable-resolved' : 'excluded')
      : impossible.has(entry.tag) || illegalOwner || inactiveFormat ? 'excluded' : 'reachable-potential';
    entry.reachability_reason = entry.tag === null
      ? (dynamicTags.length ? 'dynamic source expression exhaustively expanded from its finite call-site alternatives' : nonProtocolCall ? 'animation or target mutation, not a protocol emission' : 'server or format-only dynamic text outside the replay wire contract')
      : impossible.has(entry.tag) ? 'impossible top-level tag under supported format contract' : illegalOwner ? `owner ${entry.data_owner} is absent from legal ${ownerKind} catalog` : inactiveFormat ? 'format block is outside the two supported formats' : 'source emission retained for exact format and effect review';
  }
  const output = {schema: 1, showdown_commit: revision, generated_by: 'TypeScript compiler AST', files, entries,
    reachability_counts: entries.reduce((counts, entry) => { counts[entry.reachability] = (counts[entry.reachability] || 0) + 1; return counts; }, {}),
    review_status: 'raw_inventory_unreviewed_dynamic_sites_explicit',
    activation_effects: [...new Set(['trickroom', 'protect', 'mummy', 'symbiosis', ...dynamicActivationEffects, ...entries.filter(entry => entry.tag === '-activate' && entry.reachability !== 'excluded').map(entry => entry.arguments[2] || entry.arguments[1]).filter(arg => arg && /^['"]/.test(arg)).map(arg => arg.replace(/^['"]|['"]$/g, '').replace(/^\[(move|ability|item)\]\s*/i, '').replace(/^(move|ability|item):\s*/i, '').toLowerCase().replace(/\s+/g, ''))])],
    volatile_conditions: volatileTable()};
  const destination = path.join(ROOT, '..', 'src/p0/replays/reconstruction/showdown_raw_emission_inventory.json');
  fs.writeFileSync(destination, JSON.stringify(output, null, 2) + '\n');
}
main();
