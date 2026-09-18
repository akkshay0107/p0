"""
Validated team domain objects with lazy runtime-source exports.

The pure team schema and Stat Point modules are used by offline replay code.
Importing the package must not eagerly import poke-env just to access them.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from p0.teams.corpus_build import build_corpus
    from p0.teams.source import (
        CorpusTeamSource,
        FileTeamSource,
        FixedTeamSource,
        TeamSource,
        ValidatedTeam,
        build_team_source,
    )

__all__ = [
    "build_corpus",
    "build_team_source",
    "CorpusTeamSource",
    "FileTeamSource",
    "FixedTeamSource",
    "TeamSource",
    "ValidatedTeam",
]


def __getattr__(name: str) -> object:
    if name == "build_corpus":
        from p0.teams.corpus_build import build_corpus

        return build_corpus

    if name in (
        "build_team_source",
        "CorpusTeamSource",
        "FileTeamSource",
        "FixedTeamSource",
        "TeamSource",
        "ValidatedTeam",
    ):
        import p0.teams.source as source_module

        return getattr(source_module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
