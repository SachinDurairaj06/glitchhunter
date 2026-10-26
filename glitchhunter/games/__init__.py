"""Built-in test games. Each one renders into a git repository with planted, known bugs."""

from __future__ import annotations

from .base import BugSpec, CommitSpec, GameSpec


def get_game(name: str) -> GameSpec:
    if name == "crypt":
        from .crypt import SPEC
    elif name == "ironworks":
        from .ironworks import SPEC
    elif name == "hangar":
        from .hangar import SPEC
    else:
        raise KeyError(f"unknown game {name!r}; choose from {GAME_NAMES}")
    return SPEC


GAME_NAMES = ("crypt", "ironworks", "hangar")

__all__ = ["BugSpec", "CommitSpec", "GameSpec", "GAME_NAMES", "get_game"]
