'use strict';

const fs = require('fs');
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

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
if (!Array.isArray(input)) {
  throw new Error('Expected JSON array of team validation payloads');
}
process.stdout.write(JSON.stringify(input.map(validateItem)));
