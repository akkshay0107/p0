# TODO

This file is meant to be a sort of roadmap on features I plan on implementing later in this repo in the order of implementation.

- [x] Clean up the poke-env monkey patching / logging fixes.
- [x] Update the vocab, observation builder and model to support the entirety of VGC.
- [ ] Add operational Bo3 orchestration around the retained series encoder.
- [ ] Create a pipeline for cloning behaviour from public Showdown replays.
- [ ] Maybe consider a policy guided MCTS approach (have to build an engine from scratch) to utilise all the time given per turn. (??)
- [ ] Run a much larger training job to try to reach parity in performance with the top players (??).

If you are interested in this project, want to contribute, or could help we with training a larger model, please reach out. The long term goal would be to have a model that is able to adapt to different regulation sets as they come out with minimal retraining. Ideally, I would like for it to be at the level where people should be able to use this for practice to try new lines / teams.
