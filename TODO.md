# TODO

This file is meant to be a sort of roadmap on features I plan on implementing later in this repo in the order of implementation.

## Short Term

Things I plan on completing before I stop actively working on this project (1-2 months).

- [ ] Port pokemon-showdown and poke-env to the latest version for Reg M-B support.
- [ ] Update team lists (still sticking to a small ring of 7-8 teams) and heuristics for the new teams.
- [ ] Complete a training run with the leftover credits.
- [ ] Git LFS for replays / BC seeds (??)
- [ ] Complete p0-client and p0-infra to deploy the minigame online.

## Long Term

Would love to do these, but I wouldn't be able to train the expanded model with the budget I have so I have grouped these tasks for later.

- [ ] Clean up the poke-env monkey patching / logging fixes.
- [ ] Update the vocab, observation builder and model to support the entirety of VGC (and natively learn and play Bo3).
- [ ] Create a pipeline for cloning behaviour from public showdown replays as a replacement for the heuristic bootstrapping.
- [ ] Maybe consider a policy guided MCTS approach (have to build an engine from scratch) to utilise all the time given per turn. (??)
- [ ] Run a much larger training job to try to reach parity in performance with the top players.

If you are interested in this project, want to contribute, or could help we with training a larger model, please reach out. The long term goal would be to have a model that is able to adapt to different regulation sets as they come out with minimal retraining. Ideally, I would like for it to be at the level where people should be able to use this for practice to try new lines / teams.
