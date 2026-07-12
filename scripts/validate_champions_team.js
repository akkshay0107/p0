'use strict';

const fs = require('fs');
const {Dex, Teams, TeamValidator} = require('../pokemon-showdown/dist/sim');

Dex.includeFormats();
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const validator = TeamValidator.get(input.format);
const problems = validator.validateTeam(input.team);
process.stdout.write(JSON.stringify({
  valid: !problems,
  problems: problems || [],
  packedTeam: problems ? null : Teams.pack(input.team),
}));
