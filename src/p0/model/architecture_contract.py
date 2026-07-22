"""Fixed tensor contracts for the memory-channel model generation."""

from __future__ import annotations

OBSERVATION_ENTITY_COUNT = 15
POKEMON_COUNT = 12
OWNER_COUNT = 3

RAW_EVENT_COUNT = 64
POOLED_EVENT_COUNT = 8
EVENT_RAW_WIDTH = 128

HISTORY_WINDOW = 48
SERIES_TOKENS_PER_GAME = 4
MAX_PRIOR_GAMES = 2
SERIES_SLOTS = MAX_PRIOR_GAMES * SERIES_TOKENS_PER_GAME
CURRENT_TOKEN_COUNT = OBSERVATION_ENTITY_COUNT + 1 + POOLED_EVENT_COUNT
CURRENT_REDUCER_TOKEN_COUNT = CURRENT_TOKEN_COUNT + 1
REDUCER_MAX_LENGTH = SERIES_SLOTS + HISTORY_WINDOW + CURRENT_REDUCER_TOKEN_COUNT

# The pointer head has a semantic sentinel for self-targeting. It is not a
# sequence position and therefore cannot drift when observation rows change.
SELF_TARGET_SENTINEL = -1

TENSOR_ABI = "champions-memory-channel-v2"
OBSERVATION_SCHEMA_VERSION = 4
CHECKPOINT_ARTIFACT_SCHEMA = "p0.policy_checkpoint.v3"
