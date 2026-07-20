'use strict';

const fs = require('fs');
const readline = require('readline');
const {Dex, Teams, TeamValidator} = require('../pokemon-showdown/dist/sim');

Dex.includeFormats();
const validators = new Map();

function getValidator(format) {
  let validator = validators.get(format);
  if (!validator) {
    validator = TeamValidator.get(format);
    validators.set(format, validator);
  }
  return validator;
}

function validateItem(item) {
  try {
    const validator = getValidator(item.format);
    const problems = validator.validateTeam(item.team);
    return {
      valid: !problems,
      problems: problems || [],
      packedTeam: problems ? null : Teams.pack(item.team),
    };
  } catch (err) {
    return {
      valid: false,
      problems: [String(err.message || err)],
      packedTeam: null,
    };
  }
}

function processBatch(batch) {
  if (!Array.isArray(batch)) {
    throw new Error('Expected JSON array of team validation payloads');
  }
  return batch.map(validateItem);
}

const isPersistent = process.argv.includes('--persistent');

if (isPersistent) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      const payload = JSON.parse(trimmed);
      if (payload.command === 'stop') {
        rl.close();
        process.exit(0);
      }
      const results = processBatch(payload.batch || []);
      process.stdout.write(JSON.stringify({status: 'ok', results}) + '\n');
    } catch (err) {
      process.stdout.write(JSON.stringify({
        status: 'error',
        message: String(err.message || err),
        results: [],
      }) + '\n');
    }
  });
} else {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const results = processBatch(input);
  process.stdout.write(JSON.stringify(results));
}
