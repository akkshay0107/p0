# p0: Reinforcement Learning for Pokémon VGC

`p0` is a self-play reinforcement learning engine for Pokemon VGC. The current codebase is being refactored toward roster-independent Champions play and public Bo3 replay pretraining. NOTE: everything below is mostly stale (when I was training with a smaller vocab and a fixed set of teams). I will be updating it after I build out more features.

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Modules](#modules)
- [Workflow Guide](#workflow-guide)
  - [1. Setup & Installation](#1-setup--installation)
  - [2. PPO Training Loop](#2-ppo-training-loop)
  - [3. Local Play](#3-local-play)
- [Utility Scripts](#utility-scripts)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## About

This was initially a club project in Machine Learning @ Purdue. The original goal at the time was to create an MCTS based agent to play VGC (back when Reg H 2.0 was active in S/V). Unfortunately, building an MCTS agent required building a simulator from scratch due to how the official showdown server is implemented, which was too tedious a task.

We then pivoted to a model free approach, using small language models (TinyBERT) as the "context layer" for the policy to make decisions on top of. A lot of model free attempts at creating professional level RL bots (OpenAI Five, AlphaStar) relied on having a lot of expert data to bootstrap their model's behaviour after which self play RL was employed. We wanted to test if a model free approach could actually reach professional level play without having the prior expert data (like AlphaZero). The decision to try it out on a smaller minigame of teams was made due to the fact that getting a top level bot that plays pokemon generally would be out of budget for us as a student group. On training the model (v1) for ~11.5M steps of gameplay (thank you Concrete Engine for sponsoring this!), the results were quite average and I would estimate it to be ~1200 elo in Bo1 formats. I suspect that this is probably because TinyBERT was not good at providing the context needed for choosing moves but I cannot be sure.

With the release of Pokemon Champions, I thought it might be a good idea to revisit this by building the entire stack (vocab, tokenizer, encoder) from scratch similar to how VGC-Bench and Metamon built it. The current main branch tracks this new model and training loop that attempts to fix some of the issues from v1.

I also plan on hopefully releasing a larger article detailing the rationale behind a lot of the choices made in v1 and v2, explaining the failures, and current architecture. In the meanwhile, if you want to know more, feel free to reach out and contact me.

---

## Features

- **Custom Tokenizer & Observation Builder**: Converts Pokemon VGC game states (species, items, moves, abilities, status conditions, active/bench volatiles, side conditions) into tokens mapped from a pre-built game vocabulary. The categorical tokens and the remaining numerical features from the battle are packed into a structured observation used downstream.
- **Token Fusion Encoder**: Combines categorical embeds and numerical values per Pokemon, routing them through a single-layer encoder. Each Pokemon, global-field, and side-owner row is fused directly; event rows are encoded at low width and pooled into eight fixed event tokens. The custom implementation of a SwiGLU variant was built for fun to try something new. Other tokens for the global field status or the side conditions on each side are also fused at this layer.
- **Autoregressive Policy Pointer Head**: Uses a pointer-attention network to select actions. The first head predicts action `a1` for the first active Pokemon. This selection is embedded and passed as context to the second head to predict action `a2` for the second active Pokemon. Sequential masking prevents invalid choices (such as duplicate switch targets or multiple mega evolutions in a single turn).
- **Inbuilt Team Preview Handling**: The same policy used for battling can also be used for team picking at the team preview stage. The input is differentiated through a team preview flag in the observation.
- **League-Style Training (PFSP & Anchored Pool)**: Trains against past snapshots of itself and a shadow model (updated through EMA) with PFSP sampling. Anchors policies that have diverse play styles through an approximate test. Unlike AlphaStar, I do not train exploiter agents (since I would like my minimal step budget to go towards training the main agent; this might be considered with a larger budget).
- **Fixed Memory-Window Training**: Builds immutable per-decision local summaries, gathers a causal 48-decision history window, and reduces it with two fixed prior-game slots and full attention over a 75-position layout. BC and PPO batch complete games or bounded target windows without recurrent state APIs. Also uses DAPO style clip-higher (used to prevent entropy collapse in RLVR settings, found it interesting to try since v1 did have entropy collapse issues).
- **Vectorized Environments with Threaded Showdown Instances**: Runs parallel Node.js Pokémon Showdown server instances managed by a vectorized thread pool. It batches battle states for GPU inference.
- **Mixed Precision (FP16) & CUDA Graph Compilation**: Optional but speeds up training by around 1.7x on the few short runs I have done on a T4.

---

## Modules

### 1. `src/p0/` (Source)

- **`model/`**: Defines the tokenizer, structured observations, encoder, and actor-critic policy.
- **`train/`**: Contains PPO rollout, optimization, vector-environment, and league code.

### 2. `bench/` (Benchmarks)

Scripts to benchmark system and model performance, including inference and encoder throughput.

### 3. `tests/` (Tests)

## Workflow Guide

Steps 2 and 3 below are optional if you would like to start off self play from a purely random initialized bot. [uv](https://docs.astral.sh/uv/) is needed to setup and run the project.

### 1. Setup & Installation

First, clone the repository and initialize the git submodules (required for Pokémon Showdown):

```bash
git clone https://github.com/akkshay0107/p0.git
cd p0
git submodule update --init --recursive
```

Install the Python dependencies using uv from the p0 dir. By default, this installs the CUDA-enabled version of PyTorch:

```bash
uv python install 3.13
uv sync --extra cuda
```

_(If you are limited to CPU-only, use `uv sync --extra cpu` instead)._

Next, install the Node.js dependencies required by the local Pokémon Showdown server:

```bash
cd pokemon-showdown && npm install && cd ..
```

### 2. PPO Training Loop

The legacy heuristic bootstrap has been removed. Teams are organized into `teams/all/` for broad sampling and `teams/reduced/` for focused practice. Copy `config.yaml.example` to the ignored, machine-local `config.yaml`, then set `environment.agent_team_source.path` and `environment.opponent_team_source.path` independently. Relative paths are resolved under `paths.teams_root`.

Launch the main reinforcement learning loop. The script automatically manages the background Showdown servers and begins league-based self-play.

```bash
uv run p0-train
```

_Note: Training metrics (Win Rate, KL Divergence, Explained Variance, Entropy Loss, etc.) are exported to TensorBoard. You can view them by running `tensorboard --logdir ./artifacts/runs/ppo_training/`._

### 3. Local Play

You would have to move the trained model to a specific location and have the infra and client (which are slightly outdated since they were meant for v1) setup in order to play against the model locally with the usual showdown interface. Unfortunately, this part is slightly flaky since I haven't worked on it recently. See [p0-infra](https://github.com/akkshay0107/p0-infra) for more details.

---

## Utility Scripts

- **`cleanup.sh`**: Deletes all generated artifacts (such as TensorBoard runs, locally saved replays, checkpoints, and `.log` files) to start fresh.
- **`export_training.py`**: Exports the entire training state - current PPO weights, opponent pool backups, and the active `config.yaml` snapshot into a `tar.gz` archive. I use it for moving stuff between remote servers while training.

The former `.ppoconfig` format is no longer accepted; migrate its flat keys into the nested sections shown in `config.yaml.example`.

### Runtime compatibility

The reducer-depth benchmark measures all unified depth and pass-embedding variants using the project baseline model dimensions by default, on the project default device. Device and model dimensions have optional overrides. Timing, batch, depth, dtype, and seed inputs have practical defaults; optional BC validation requires a compatible checkpoint and tensor artifact.

For a default-sized run:

    uv run python bench/benchmark_reducer_depth.py --dtype float32 --seed 7 --warmup 2 --iterations 5 --repeats 5 --batch-size 2 --time-steps 4 --deep-core-repeats 3

`data/runtime_manifest.json` contains one human-readable, load-breaking runtime contract.
Checkpoints reference its `runtime_contract_sha256`. Vocabulary, action-layout, tensor-ABI,
or resource-feature-ABI changes require a new checkpoint or an explicit transfer tool.
Dex balance/learnset changes and Showdown revisions are recorded as mechanics provenance;
they do not prevent an existing policy from loading and continuing training. Old checkpoint
dictionaries containing `runtime_manifest_sha256` are intentionally unsupported.

## Development verification

Run the standard checkpoint gates from the repository root:

```bash
uv run ruff check src tests
uv run pyright
uv run pytest -q
uv build
```

The installed command-line interfaces are `p0-train`, `p0-play`, `p0-build-vocab`, and `p0-export-training`.

---

## References

This project was heavily inspired by the work of the devs behind the following repos, and in several cases, components of their source code were adapted or utilized as foundations for this engine. I am deeply grateful to them.

- [poke-env](https://github.com/hsahovic/poke-env)
- [Pokemon Showdown](https://github.com/smogon/pokemon-showdown)
- [VGC Bench](https://github.com/cameronangliss/VGC-Bench)
- [Metamon](https://github.com/UT-Austin-RPL/metamon)
- [Foul Play](https://github.com/pmariglia/foul-play)

---

## Contributing

Contributions are very welcome! If you are interested in this project, have any feedback or queries, want to help implement new features, or can provide resources to train larger models, please reach out.

Please refer to [TODO.md](TODO.md) for the roadmap of features I plan on implementing.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Also see [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) for the license of the included submodule.
