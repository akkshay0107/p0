/* Emit a deterministic trace from the pinned local Showdown engine. */
const path = require("node:path");
const {BattleStream, Teams} = require(path.join(__dirname, "..", "pokemon-showdown", "dist", "sim"));

// Python verifies this revision before running the check.
const commit = "8282e63102fa824fd2f7472778ec09793ceb7cac";

const species = ["Pikachu", "Eevee", "Raichu", "Jolteon", "Vaporeon", "Flareon"];
const team = Teams.pack(species.map((name) => ({name, species: name, moves: ["protect", "tackle"]})));
const stream = new BattleStream();
const protocol = [];
let forceWinSent = false;
let teamChosen = {p1: false, p2: false};
let activeChosen = {p1: false, p2: false};

function write(command) { stream.write(command); }
function handleChunk(chunk) {
  const chunkLines = chunk.split("\n");
  const type = chunkLines.shift();
  const data = chunkLines.join("\n");
  if (type === "update") protocol.push(...data.split("\n").filter((line) => line.startsWith("|")));
  if (type !== "sideupdate") return;
  const side = data.split("\n", 1)[0];
  const requestLine = data.split("\n").find((line) => line.startsWith("|request|"));
  if (!requestLine) return;
  const request = JSON.parse(requestLine.slice("|request|".length));
  if (request.teamPreview && !teamChosen[side]) {
    teamChosen[side] = true;
    write(`>${side} team 1234`);
  } else if (request.active && !activeChosen[side]) {
    activeChosen[side] = true;
    write(`>${side} move 1, move 1`);
    if (activeChosen.p1 && activeChosen.p2 && !forceWinSent) {
      forceWinSent = true;
      setTimeout(() => write(">forcewin p1"), 25);
    }
  }
}

(async () => {
  for await (const chunk of stream) {
    handleChunk(chunk);
    if (forceWinSent && protocol.some((line) => line.startsWith("|win|"))) break;
  }
  process.stdout.write(JSON.stringify({commit, protocol, teamMembers: species}) + "\n");
})().catch((error) => { process.stderr.write(`${error.stack || error}\n`); process.exitCode = 1; });

write(`>start ${JSON.stringify({formatid: "gen9championsvgc2026regmb"})}
>player p1 ${JSON.stringify({name: "Alice", team})}
>player p2 ${JSON.stringify({name: "Bob", team})}`);

// Keep the process alive while BattleStream sends its updates.
setTimeout(() => {
  if (!teamChosen.p1) write(">p1 team 1234");
  if (!teamChosen.p2) write(">p2 team 1234");
}, 10);
setTimeout(() => {
  if (!forceWinSent) write(">forcewin p1");
}, 250);
