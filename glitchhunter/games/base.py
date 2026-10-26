"""What a test game is made of: a template, a commit history, planted bugs, textures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CommitSpec:
    subject: str
    features: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    drop_variants: tuple[str, ...] = ()  # undo a variant (e.g. the commit that fixes a broken build)
    tag: str | None = None
    branch: str | None = None  # commit on this side branch (created from main on first use)
    merge: str | None = None  # merge this side branch into main (a real two-parent merge commit)
    scenarios: tuple[str, ...] = ()  # scenarios this commit adds to glitchhunter.json


@dataclass(frozen=True)
class BugSpec:
    name: str
    title: str
    detector: str
    scenarios: tuple[str, ...]
    flaky: bool = False
    trigger: str | None = None  # feature/variant that plants it, when it is not the bug's own name
    viewport: tuple[int, int] | None = None  # only visible at this screen size


@dataclass
class GameSpec:
    name: str
    title: str
    template_dir: Path
    history: tuple[CommitSpec, ...]
    bugs: dict[str, BugSpec]
    textures: dict[str, Callable[[], np.ndarray]]
    assets_by_feature: dict[str, tuple[str, ...]]
    manifest: dict  # glitchhunter.json minus "scenarios"
    scenarios: dict[str, dict]  # scenario name -> entry of the manifest's "scenarios" list
    vendor: tuple[tuple[str, str], ...]  # (path under games/vendor, path in the game repo)
    author: tuple[str, str] = ("Game Dev", "dev@game.invalid")
    clean_subjects: tuple[str, ...] = ()  # benign commits used for the false-alarm check
    base_scenarios: tuple[str, ...] = ()  # scenarios in the first commit's glitchhunter.json
    notes: dict = field(default_factory=dict)
    template_file: str = "game.tpl.js"  # in template_dir, rendered for every commit
    output_file: str = "game.js"  # what the template renders to in the game repo
    static_files: tuple[str, ...] = ("index.html",)  # copied from template_dir into the first commit
    gitignore: str = ""  # .gitignore content for the game repo (engine caches)
    engine: str = "three.js (WebGL)"

    @property
    def baseline(self) -> str:
        return self.manifest["baseline"]

    def culprit_index(self, bug: str) -> int:
        spec = self.bugs[bug]
        trigger = spec.trigger or bug
        for i, commit in enumerate(self.history):
            if trigger in commit.variants or trigger in commit.features:
                return i
        raise KeyError(bug)
