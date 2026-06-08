"""Shared shape and action-space constants."""

# Structured observation: CLS + 12 interleaved Pokemon super/numeric pairs + 6 field tokens.
OBS_DIM = (31, 50)

# Action space:
# 0 pass, 1-6 switches, 7-26 regular move-target actions, 27-46 Mega move-target actions.
ACT_SIZE = 47
