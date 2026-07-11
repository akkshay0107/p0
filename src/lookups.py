"""Shared shape and action-space constants."""

from src.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH

# Structured observation: CLS + 12 interleaved Pokemon super/numeric pairs + 6 field tokens.
OBS_DIM = (SEQUENCE_LENGTH, NUMERICAL_WIDTH)

# Action space:
# 0 pass, 1-6 switches, 7-26 regular move-target actions, 27-46 Mega move-target actions.
# 47 mega struggle/recharge, 48 struggle/recharge
ACT_SIZE = 49
