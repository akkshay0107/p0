"""Validated team domain objects with lazy runtime-source exports.

The pure team schema and Stat Point modules are used by offline replay code.
Importing the package must not eagerly import poke-env just to access them.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p0.teams.source import FileTeamSource, FixedTeamSource, TeamSource, ValidatedTeam

__all__ = ["FileTeamSource", "FixedTeamSource", "TeamSource", "ValidatedTeam"]


def __getattr__(name: str):
    if name in __all__:
        from p0.teams.source import FileTeamSource, FixedTeamSource, TeamSource, ValidatedTeam

        return {
            "FileTeamSource": FileTeamSource,
            "FixedTeamSource": FixedTeamSource,
            "TeamSource": TeamSource,
            "ValidatedTeam": ValidatedTeam,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
