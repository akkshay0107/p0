# p0: Reinforcement Learning for Pokémon Champions

## The Idea

`p0` is a self-play reinforcement learning agent designed to play Pokémon VGC (Champions Regulation M-A) at a highly competent level without human data. Instead of learning completely from scratch, the system uses a hybrid approach. It first bootstraps a neural network policy via Behavioural Cloning from rule-based heuristics. From there, it transitions into Proximal Policy Optimization (PPO), learning through self-play and a league-based format against historical snapshots of itself to master team combinations, positioning, and prediction.

---

## Module Breakdown

The project is structured into three main modules:

### 1. `src/` (Source)

The core implementation of the environment, model, heuristic bots, and the training pipeline.

- **`model/`**: Defines the neural network architecture. Contains the custom tokenizer, structured observation builders, the fused SwiGLU token encoder, and the dual actor-critic policy networks.
- **`train/`**: Houses the PPO training loop, rollout buffers, vectorized environment management (spinning up local Node.js Showdown servers), behavioural cloning scripts, and the opponent pool (league) system.
- **`heuristic/`**: Contains rule-based baseline agents and scripts used for generating initial high-quality replays for Behavioural Cloning.

### 2. `bench/` (Benchmarks)

Scripts to benchmark system and model performance. This includes measuring inference times (action prediction and encoder throughput) as well as evaluating win rates for heuristic bots and policy models.

### 3. `tests/` (Tests)

A comprehensive unit testing suite. Ensures the correctness of observation parsing, neural network forward/backward passes, strict sequence masking, and tokenization logic.

---

## Model Architecture Choices

The model is designed specifically to handle open team sheets (and trained on open team sheet Bo1 games). It has fallbacks for unknown items, but I wouldn't recommend trying it out outside this specific case.

- **Fused Token Encoder (Backbone)**: Categorical variable embeddings (species, moves, items, types, abilities, volatile status) are projected and summed together per slot alongside learned token-type, slot, and side embeddings. Numerical states are concatenated and normalized. A SwiGLU Transformer layer fuses these components into distinct tokens.
- **Shared Trunk, Split Heads**: The Actor and Critic share the `FusedTokenEncoder` backbone for computational efficiency but branch off into separate paths.
- **Actor (Stateful)**: The actor path uses a recurrent memory design. Memory/History tokens are passed between turns (managed by a `CLSReducer`) to provide temporal context across a match.
- **Autoregressive Doubles Policy**: Because two actions must be selected in VGC Doubles, the actor predicts them autoregressively. It first predicts the action for Pokémon 1 ($P(a_1|z)$). This selected action is embedded and fed sequentially to predict the action for Pokémon 2 ($P(a_2|z, a_1)$). Sequential masking is applied to prevent illegal combinations (e.g., both Pokémon switching into the same slot, or attempting to mega-evolve twice).
- **Critic (Stateless)**: Provides a value estimate from the observation. There is optional code to scale down critic gradients into the shared features.

---

## Training Loop

The script automatically manages building, spinning up, and tearing down local `pokemon-showdown` servers on multiple ports to gather rollouts in parallel. The training loop runs using Proximal Policy Optimization (PPO) over a vectorized threaded environment (`ThreadVecEnv`).

- **League-Based Self-Play**: To prevent policy collapse and catastrophic forgetting, the system uses an `OpponentPool`. Trajectories are gathered by playing not just the current version of the agent, but also by playing against past checkpoints (snapshots) randomly sampled with FPSP from the pool.
- **Variable-Length BPTT**: The loop handles variable-length episodes and supports BPTT over minibatches.

---

## Workflow Guide

Steps 2 and 3 below are optional if you would like to start off self play from a purely random intialized bot.

### 1. Setup & Installation

First, clone the repository and initialize the git submodules (required for Pokémon Showdown):

```bash
git submodule update --init --recursive
```

Install the Python dependencies using `uv`. By default, this installs the CUDA-enabled version of PyTorch:

```bash
uv python install 3.13
uv sync --extra cuda
```

_(If you are limited to CPU-only, use `uv sync --extra cpu` instead)._

Next, install the Node.js dependencies required by the local Pokémon Showdown server:

```bash
cd pokemon-showdown && npm install && cd ..
```

### 2. Replay Generation

Generate offline replays by pitting heuristic rule-based bots against each other. This creates the foundational dataset for the model to learn the basics. This runs entirely on the CPU and takes a while (~20 mins with n = 5000 battles) to generate replays. It ends up generating more than the number asked due to recording both sides in mirror bot matches.

```bash
uv run python p0/src/heuristic/replay_gen.py -n 2500
```

### 3. Behavioural Cloning Bootstrapping

Train the initial neural network policy using supervised learning to predict the actions taken by the heuristics in the generated replays. This bootstraps the initial `OpponentPool` and gives the model a competent starting point before RL begins.

```bash
uv run python p0/src/train/seed_pool.py
```

### 4. PPO Training Loop

Launch the main reinforcement learning loop. The script automatically manages the background Showdown servers and begins league-based self-play.

```bash
uv run python p0/src/train/train_loop.py
```

_Note: Training metrics (Win Rate, KL Divergence, Explained Variance, Entropy Loss, etc.) are exported to TensorBoard. You can view them by running `tensorboard --logdir p0/runs/ppo_training/`._

---

## Utility Scripts

- **`cleanup.sh`**: Deletes all generated artifacts (such as TensorBoard runs, locally saved replays, checkpoints, and `.log` files) to start fresh.
- **`export_training.py`**: Exports the entire training state - current PPO weights, opponent pool backups, and `.ppoconfig` into a `tar.gz` archive. I use it for moving stuff between remote servers.
- **`reset_pool.sh`**: Clears out the current `OpponentPool` (snapshots inside `checkpoints/pool/`) while retaining the foundational heuristic seed models. If you want to reset the training phase without redoing replay gen / behaviour cloning.
