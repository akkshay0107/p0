/* Build the raw Showdown protocol emission inventory from the pinned TS tree. */
'use strict';

const fs = require('fs');
const path = require('path');
const ts = require('../pokemon-showdown/node_modules/typescript');

const ROOT = path.resolve(__dirname, '..', 'pokemon-showdown');
const DATA_ROOT = path.resolve(ROOT, '..', 'data');
const DATA_FILES = new Set(['moves.ts', 'abilities.ts', 'items.ts', 'conditions.ts', 'rulesets.ts']);
const catalog = JSON.parse(fs.readFileSync(path.join(DATA_ROOT, 'champions_dex.json'), 'utf8'));
const FORMATS = [catalog.source.battleFormat, catalog.source.bo3Format];
const DIRECTORIES = ['config/formats.ts', `data/mods/${catalog.source.mod}`, 'sim', 'data'];
const CALLS = new Set(['add', 'addMove', 'addSplit', 'attrLastMove', 'retargetLastMove']);
const {Dex} = require('../pokemon-showdown/dist/sim/dex');
const ACTIVE_RULES = new Set(FORMATS.flatMap(id => {
  const format = Dex.formats.get(id);
  return [...Dex.formats.getRuleTable(format).keys()];
}));

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

function templatePrefix(node) {
  if (ts.isTemplateExpression(node)) return node.head.text;
  if (ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
  return null;
}

function localStringLiteral(node, name) {
  for (let current = node.parent; current; current = current.parent) {
    if (!ts.isMethodDeclaration(current) && !ts.isFunctionDeclaration(current)) continue;
    if (!current.body) return null;
    const values = [];
    function visit(child) {
      if (ts.isVariableDeclaration(child) && child.name.getText() === name &&
          child.initializer && ts.isStringLiteralLike(child.initializer)) values.push(child.initializer.text);
      ts.forEachChild(child, visit);
    }
    visit(current.body);
    return values.length === 1 ? values[0] : null;
  }
  return null;
}

function guardedByInactiveRule(node, activeRules) {
  for (let current = node.parent; current; current = current.parent) {
    if (!ts.isIfStatement(current)) continue;
    const condition = current.expression;
    if (!ts.isCallExpression(condition) ||
        condition.expression.getText() !== 'this.ruleTable.has' ||
        condition.arguments.length !== 1) continue;
    const rule = condition.arguments[0];
    if (ts.isStringLiteralLike(rule) && !activeRules.has(rule.text)) return true;
  }
  return false;
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
        const split = name === 'addSplit' ? node.arguments[1] : null;
        const splitFirst = split && ts.isArrayLiteralExpression(split) ? split.elements[0] : null;
        const splitText = splitFirst && ts.isStringLiteralLike(splitFirst) ? splitFirst.text
          : splitFirst && ts.isIdentifier(splitFirst) ? localStringLiteral(node, splitFirst.text) : null;
        const splitTag = splitText?.split('|', 1)[0] || null;
        const prefix = first ? templatePrefix(first) : null;
        const inactiveRuleGuard = guardedByInactiveRule(node, ACTIVE_RULES);
        entries.push({
          path: path.relative(ROOT, file).replaceAll(path.sep, '/'),
          line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1,
          call: name,
          tag: wireTag,
          ...(splitTag ? {split_tag: splitTag} : {}),
          ...(prefix !== null ? {template_prefix: prefix} : {}),
          dynamic_tag_expression: literal === null && first ? first.getText(source) : null,
          ...(inactiveRuleGuard ? {inactive_rule_guard: true} : {}),
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
  return {path: path.relative(ROOT, file).replaceAll(path.sep, '/'), entries};
}

function volatileTable() {
  const dex = Dex.mod(catalog.source.mod);
  const rows = [];
  const keyLocation = id => {
    for (const filename of ['moves.ts', 'items.ts', 'abilities.ts', 'conditions.ts']) {
      const candidate = path.join(ROOT, 'data', filename);
      const lines = fs.readFileSync(candidate, 'utf8').split('\n');
      const index = lines.findIndex(line => new RegExp(`^\\s+${id}:\\s*[<{]`).test(line));
      if (index >= 0) return {path: `data/${filename}`, line: index + 1};
    }
    return {path: 'data/conditions.ts', line: null};
  };
  const legalEffects = new Set(catalog.legalProtocolEffects.effect || []);
  const legalOwners = new Map(['moves', 'items', 'abilities'].map(kind => [kind,
    new Set((catalog.legality?.[kind] || []).map(entry => typeof entry === 'string' ? entry : entry.id))]));
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
  const files = sourceFiles().map(scan);
  const entries = files.flatMap(file => file.entries);
  const impossible = new Set(['debug', '-candynamax', '-center', '-terastallize', '-zpower', '-primal', '-burst', '-swapsideconditions', '-combine', '-waiting', '-notarget', '-nothing', '-eat']);
  const legal = new Map(['moves', 'items', 'abilities'].map(kind => [kind, new Set(catalog.legality?.[kind] || [])]));
  const dex = Dex.mod(catalog.source.mod);
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
    const inactiveFormat = entry.path === 'config/formats.ts' && !FORMATS.includes(entry.data_owner);
    const inactiveRuleset = entry.path.endsWith('/rulesets.ts') && !ACTIVE_RULES.has(entry.data_owner);
    const serverOnly = entry.path.startsWith('server/') || entry.path === 'sim/battle-stream.ts';
    const genericSplit = entry.path === 'sim/battle.ts' &&
      (entry.enclosing === 'addSplit' || entry.enclosing === 'add');
    const templateTag = entry.template_prefix?.replace(/^\|/, '').split('|', 1)[0];
    const dynamicTags = entry.dynamic_tag_expression === "isDrag ? 'drag' : 'switch'" ? ['drag', 'switch']
      : entry.dynamic_tag_expression === "this.battle.gen >= 5 ? '-fail' : '-notarget'" ? ['-fail', '-notarget']
      : entry.split_tag ? [entry.split_tag]
      : templateTag ? [templateTag]
      : entry.path.endsWith('/rulesets.ts') && entry.dynamic_tag_expression === '${buf}' ? ['clearpoke', 'poke', 'rule']
      : entry.dynamic_tag_expression === 'msg' ? ['-boost', '-unboost', '-setboost']
      : [];
    entry.resolved_tags = dynamicTags;
    const nonProtocolCall = entry.call === 'attrLastMove' || entry.call === 'retargetLastMove';
    const excluded = nonProtocolCall || illegalOwner || inactiveFormat || inactiveRuleset ||
      entry.inactive_rule_guard || serverOnly || genericSplit || impossible.has(entry.tag) ||
      (entry.split_tag && impossible.has(entry.split_tag));
    entry.reachability = excluded
      ? 'excluded'
      : entry.tag === null
      ? (dynamicTags.length ? 'reachable-resolved' : 'unresolved')
      : 'reachable-potential';
    entry.reachability_reason = nonProtocolCall ? 'animation or target mutation, not a protocol emission'
      : illegalOwner ? `owner ${entry.data_owner} is absent from legal ${ownerKind} catalog`
      : inactiveFormat ? 'format block is outside the two supported formats'
      : inactiveRuleset ? `rule ${entry.data_owner} is absent from both supported rule tables`
      : entry.inactive_rule_guard ? 'guarded by a rule absent from both supported rule tables'
      : serverOnly ? 'server display or forwarding code, outside simulator replay output'
      : genericSplit ? 'generic addSplit forwarding helper, covered at its call sites'
      : impossible.has(entry.tag) || (entry.split_tag && impossible.has(entry.split_tag)) ? 'impossible top-level tag under supported format contract'
      : entry.reachability === 'unresolved' ? 'dynamic source expression needs classification'
      : dynamicTags.length ? 'source expression expanded from structural call-site values'
      : 'source emission retained for exact format and effect review';
  }
  const output = {schema: 1, showdown_commit: catalog.source.commit, formats: FORMATS, generated_by: 'TypeScript compiler AST', files, entries,
    reachability_counts: entries.reduce((counts, entry) => { counts[entry.reachability] = (counts[entry.reachability] || 0) + 1; return counts; }, {}),
    review_status: entries.some(entry => entry.reachability === 'unresolved')
      ? 'raw_inventory_unresolved_sites' : 'raw_inventory_reachable_sites_classified',
    activation_effects: [...new Set(['trickroom', 'protect', 'mummy', 'symbiosis', ...dynamicActivationEffects, ...entries.filter(entry => entry.tag === '-activate' && entry.reachability !== 'excluded').map(entry => entry.arguments[2] || entry.arguments[1]).filter(arg => arg && /^['"]/.test(arg)).map(arg => arg.replace(/^['"]|['"]$/g, '').replace(/^\[(move|ability|item)\]\s*/i, '').replace(/^(move|ability|item):\s*/i, '').toLowerCase().replace(/\s+/g, ''))])],
    volatile_conditions: volatileTable()};
  const destination = path.join(DATA_ROOT, 'showdown_raw_emission_inventory.json');
  fs.writeFileSync(destination, JSON.stringify(output, null, 2) + '\n');
}
main();
