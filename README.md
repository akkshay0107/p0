# p0: Reinforcement Learning for Pokémon VGC

`p0` trains Pokémon VGC agents through self-play on Pokémon Showdown. It supports
Champions best-of-three (Bo3) battles with open team sheets (OTS), optional
behaviour cloning (BC) from public replays, and PPO training from scratch or a BC policy.
You can evaluate checkpoints against baseline bots or run them as Showdown players.

This is a research project. GPU validation is deferred, and the current model's
playing strength and training throughput have not been measured.

## Getting started

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Python 3.13 or newer, and Node.js with npm. The pinned Showdown version requires
Node.js 16 or newer. Training uses POSIX file locks; use Linux or macOS.

```bash
git clone https://github.com/akkshay0107/p0.git
cd p0
git submodule update --init --recursive
uv python install 3.13
uv sync --extra cpu
npm --prefix pokemon-showdown install
cp config.example.yaml config.yaml
```

For a CUDA machine, replace `uv sync --extra cpu` with `uv sync --extra cuda`.
The two extras are mutually exclusive. Run the commands below from the repository root.

### Train from scratch

Add your own six-Pokémon Showdown team exports (`.txt`) to `teams/all/`.
Teams must be legal for Champions VGC Regulation M-B. Team files and generated
corpus manifests are local inputs and are not included in the repository.

Review [config.example.yaml](config.example.yaml) and adjust your `config.yaml`,
particularly `training.n_envs` and the training budget. Then build the pool and train:

```bash
uv run p0-corpus build --input teams/all
uv run p0-train
```

Training starts from random weights and manages its own Showdown servers. By default,
it saves to `artifacts/checkpoints/ppo_checkpoint.pt`, with metrics and TensorBoard
logs under `artifacts/runs/ppo_training/`. Use empty output locations for a new run.

For a smaller agent pool, supply exports in `teams/reduced/`, build it with
`p0-corpus build --input teams/reduced`, and run `p0-train --agent-team-source reduced`.
The opponent still samples from `teams/all/`; the reduced pool is optional.

Set `paths.resume_checkpoint` in `config.yaml` to resume a training checkpoint,
or `paths.initial_policy_checkpoint` to start a new run from policy weights.
These settings are mutually exclusive. Resume keeps the saved settings and PPO budget;
use policy initialization for a new schedule. PPO drops unfinished games when saving.

## Other workflows

### Pretrain from replays

BC is optional. Collect complete linked Bo3 OTS series, compile them, and split
by series so games from one series do not cross train/validation/test boundaries:

```bash
uv run p0-replays scrape --cache-dir artifacts/replays --limit-games 50
uv run p0-replays build-shards --cache-dir artifacts/replays --output-dir artifacts/shards
```

`build-shards` prints a `manifest_path`. Replace the example path below with that
actual path; there is no fixed default. `create-splits` writes `splits.json` beside it.

```bash
shard_manifest="/absolute/path/from/build-shards/manifest.json"
uv run p0-replays create-splits --shard-manifest "$shard_manifest"
uv run p0-bc train --config config.yaml \
  --shard-manifest "$shard_manifest" \
  --split-manifest "${shard_manifest%/*}/splits.json"
```

The best BC policy is saved to `artifacts/checkpoints/bc/bc_best_policy.pt` by default.
Set `paths.initial_policy_checkpoint` to that file before starting PPO.
To use config instead of CLI overrides, replace `bc.shard_manifest` and
`bc.split_manifest` in `config.yaml`; the example values are placeholders.

After changing replay reconstruction, remove the old dataset build directory before
rebuilding shards, recreate the splits, and update their paths. Existing builds are
reused, and the dataset hash does not include reconstruction code changes.

### Evaluate or play a checkpoint

Evaluate against a baseline on a server managed by p0:

```bash
uv run p0-eval --checkpoint artifacts/checkpoints/ppo_checkpoint.pt --opponent random
```

The default report is `artifacts/eval/evaluation_report.json`. Use `--help` for other
baselines, checkpoint opponents, and evaluation settings.

`p0-play` connects to an existing server. For local play, start Showdown in a separate
terminal from `pokemon-showdown/`:

```bash
npm run build
node pokemon-showdown start --no-security 8000
```

Then, from the p0 repository root:

```bash
uv run p0-play --checkpoint artifacts/checkpoints/ppo_checkpoint.pt --username MyBot --team-pool all
```

Challenge `MyBot` in `gen9championsvgc2026regmbbo3`, the supported live-play format.
The `--no-security` server setup above is for local use. See `p0-play --help` for
remote server settings, repeatable team files, and challenge limits.

## Code and development

The policy encodes the board and recent battle events, reads a 48-decision history
and prior-game summaries, and selects the two actions in sequence with legality masks.
PPO runs the policy on both sides and regularizes it toward a frozen copy that is
refreshed periodically.

- [Configuration](config.example.yaml): training, paths, team pools, BC, and evaluation.
- [Model](src/p0/model/): [model dimensions](src/p0/model/config.py) and
  [tensor layout constants](src/p0/model/architecture_contract.py).
- [Runtime contract](data/runtime_manifest.json): hashes used to check artifact compatibility.
- [Command-line entry points](src/p0/cli/): run each command with `--help` for options.

From the repository root:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 P0_STRESS_GAME_COUNT=50 uv run pytest -q -m heavy
git diff --check
uv build
```

The default and heavy suites must pass before a commit. Integration tests need Node.js,
the Showdown submodule and dependencies, and loopback sockets. The default run excludes
heavy, stress, network, and GPU tests. Keep the heavy replay workload at 50 games or
fewer per run. GPU checks are deferred; CPU passes do not validate CUDA execution.

## About

This was initially a club project in Machine Learning @ Purdue. The original goal at the time was to create an MCTS based agent to play VGC (back when Reg H 2.0 was active in S/V). Unfortunately, building an MCTS agent required building a simulator from scratch due to how the official showdown server is implemented, which was too tedious a task.

We then pivoted to a model free approach, using small language models (TinyBERT) as the "context layer" for the policy to make decisions on top of. A lot of model free attempts at creating professional level RL bots (OpenAI Five, AlphaStar) relied on having a lot of expert data to bootstrap their model's behaviour after which self play RL was employed. We wanted to test if a model free approach could actually reach professional level play without having the prior expert data (like AlphaZero). The decision to try it out on a smaller minigame of teams was made due to the fact that getting a top level bot that plays pokemon generally would be out of budget for us as a student group. On training the model (v1) for ~11.5M steps of gameplay (thank you Concrete Engine for sponsoring this!), the results were quite average and I would estimate it to be ~1200 elo in Bo1 formats. I suspect that this is probably because TinyBERT was not good at providing the context needed for choosing moves but I cannot be sure.

With the release of Pokemon Champions, I thought it might be a good idea to revisit this by building the entire stack (vocab, tokenizer, encoder) from scratch similar to how VGC-Bench and Metamon built it. The current main branch tracks this new model and training loop that attempts to fix some of the issues from v1.

I also plan on hopefully releasing a larger article detailing the rationale behind a lot of the choices made in v1 and v2, explaining the failures, and current architecture. In the meanwhile, if you want to know more, feel free to reach out and contact me.

## References

This project was heavily inspired by the work of the devs behind the following repos, and in several cases, components of their source code were adapted or utilized as foundations for this engine. I am deeply grateful to them.

- [poke-env](https://github.com/hsahovic/poke-env)
- [Pokemon Showdown](https://github.com/smogon/pokemon-showdown)
- [VGC Bench](https://github.com/cameronangliss/VGC-Bench)
- [Metamon](https://github.com/UT-Austin-RPL/metamon)
- [Foul Play](https://github.com/pmariglia/foul-play)

## Contributing

Contributions are very welcome! If you are interested in this project, have any feedback or queries, want to help implement new features, or can provide resources to train larger models, please reach out or open an issue on GitHub.

For bugs and questions, [open an issue](https://github.com/akkshay0107/p0/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Also see [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) for the license of the included submodule.
