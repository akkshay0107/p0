"""Fixed tensor contracts for the memory-channel model generation."""

from __future__ import annotations

from p0.format_config import active_global_contract

_MODEL_CONTRACT = active_global_contract().payload("model", "major")

OBSERVATION_ENTITY_COUNT = _MODEL_CONTRACT["observation_entity_count"]
POKEMON_COUNT = _MODEL_CONTRACT["pokemon_count"]
OWNER_COUNT = _MODEL_CONTRACT["owner_count"]

RAW_EVENT_COUNT = _MODEL_CONTRACT["raw_event_count"]
POOLED_EVENT_COUNT = _MODEL_CONTRACT["pooled_event_count"]
EVENT_RAW_WIDTH = _MODEL_CONTRACT["event_raw_width"]

HISTORY_WINDOW = _MODEL_CONTRACT["history_window"]
SERIES_TOKENS_PER_GAME = _MODEL_CONTRACT["series_tokens_per_game"]
MAX_PRIOR_GAMES = _MODEL_CONTRACT["max_prior_games"]
SERIES_SLOTS = MAX_PRIOR_GAMES * SERIES_TOKENS_PER_GAME
CURRENT_TOKEN_COUNT = OBSERVATION_ENTITY_COUNT + 1 + POOLED_EVENT_COUNT
CURRENT_REDUCER_TOKEN_COUNT = CURRENT_TOKEN_COUNT + 1
REDUCER_MAX_LENGTH = SERIES_SLOTS + HISTORY_WINDOW + CURRENT_REDUCER_TOKEN_COUNT

# The pointer head has a semantic sentinel for self-targeting. It is not a
# sequence position and therefore cannot drift when observation rows change.
SELF_TARGET_SENTINEL = _MODEL_CONTRACT["self_target_sentinel"]

OBSERVATION_SCHEMA_VERSION = _MODEL_CONTRACT["observation_schema_version"]
CHECKPOINT_ARTIFACT_SCHEMA = active_global_contract().payload("checkpoints", "major")[
    "artifact_schema"
]
