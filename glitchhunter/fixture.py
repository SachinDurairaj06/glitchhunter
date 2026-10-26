"""Builds a test game's git history with planted, known bugs.

The rendered repository is a normal git repo: each commit is a plausible change
to the game, and some of them plant a visual bug. Histories can have a side
branch merged back with a real merge commit, and commits that do not build.
The ground truth (which commit planted which bug) is written next to the repo,
never inside it, so the agent cannot read the answers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2 as cv

from .config import GAMES_DIR, REPO_ROOT, fs_path
from .games import GameSpec, get_game
from .project import MANIFEST

# Crypt Runner names, kept for older callers.
BASELINE_TAG = "release-1.0"

# Markers: `//#if x` in JavaScript/C-like templates, `#@if x` in GDScript/Python-like ones.
_IF = re.compile(r"^\s*(?://#|#@)if\s+(!?\w+)\s*$")
_ENDIF = re.compile(r"^\s*(?://#|#@)endif\s*$")
_VAR = re.compile(r"^(?P<indent>\s*)(?P<code>.*?)\s*(?://#|#@)var\s+(?P<name>\w+):\s?(?P<alt>.*)$")


def render_template(template: str, features: set[str], variants: set[str]) -> str:
    """Render a game template for one commit: keep enabled blocks, pick line variants.

    `//#if name` is true when name is an active feature or variant (`//#if !name` inverts it).
    """
    out: list[str] = []
    stack: list[bool] = []
    active = features | variants
    for line in template.splitlines():
        m = _IF.match(line)
        if m:
            name = m.group(1)
            stack.append((name[1:] not in active) if name.startswith("!") else (name in active))
            continue
        if _ENDIF.match(line):
            if not stack:
                raise ValueError("unbalanced //#endif in template")
            stack.pop()
            continue
        if not all(stack):
            continue
        v = _VAR.match(line)
        if v:
            code = v["alt"] if v["name"] in variants else v["code"]
            line = v["indent"] + code.strip()
        out.append(line.rstrip())
    if stack:
        raise ValueError("unclosed //#if in template")
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n"


# --- git helpers -------------------------------------------------------------------------------------

def _git(repo: Path, *args: str, env: dict | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=repo, env=env, capture_output=True, text=True, check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass
class FixtureInfo:
    repo: str
    head: str
    baseline: str
    baseline_tag: str
    commits: list[dict] = field(default_factory=list)
    bugs: dict[str, dict] = field(default_factory=dict)
    game: str = "crypt"
    untestable: list[str] = field(default_factory=list)
    clean: list[str] = field(default_factory=list)  # commits with no planted bug (false-alarm check)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FixtureInfo":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class _State:
    features: set[str] = field(default_factory=set)
    variants: set[str] = field(default_factory=set)
    scenarios: list[str] = field(default_factory=list)

    def copy(self) -> "_State":
        return _State(set(self.features), set(self.variants), list(self.scenarios))


def _manifest(game: GameSpec, scenarios: list[str]) -> str:
    data = {**game.manifest, "scenarios": [game.scenarios[s] for s in scenarios if s in game.scenarios]}
    return json.dumps(data, indent=2) + "\n"


def _write_tree(dest: Path, game: GameSpec, template: str, state: _State, new_features: set[str]) -> None:
    for feature in new_features:
        for asset in game.assets_by_feature.get(feature, ()):
            cv.imwrite(str(dest / "assets" / asset), game.textures[asset]())
    (dest / game.output_file).write_text(render_template(template, state.features, state.variants),
                                         encoding="utf-8", newline="\n")
    (dest / MANIFEST).write_text(_manifest(game, state.scenarios), encoding="utf-8", newline="\n")


def build_fixture(dest: Path, manifest: Path, game: str | GameSpec = "crypt") -> FixtureInfo:
    """(Re)create the game repo at dest and write the ground-truth manifest."""
    spec = get_game(game) if isinstance(game, str) else game
    template = Path(fs_path(spec.template_dir / spec.template_file)).read_text(encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest, onerror=_force_remove)
    dest.mkdir(parents=True)
    _git(dest, "init", "-q", "-b", "main")
    for key, value in (("user.name", spec.author[0]), ("user.email", spec.author[1]),
                       ("core.autocrlf", "false"), ("core.longpaths", "true"), ("commit.gpgsign", "false"),
                       ("tag.gpgsign", "false"), ("merge.ff", "false")):
        _git(dest, "config", key, value)

    _copy_static(spec, dest)
    (dest / "assets").mkdir(exist_ok=True)

    main = _State()
    branches: dict[str, tuple[_State, _State]] = {}  # name -> (state at the fork, current state)
    commits: list[dict] = []
    by_index: dict[int, str] = {}
    current = "main"
    for i, c in enumerate(spec.history):
        target_branch = c.branch or "main"
        if c.merge:
            target_branch = "main"
        if target_branch != current:
            if target_branch != "main" and target_branch not in branches:
                _git(dest, "checkout", "-q", "-b", target_branch)
                branches[target_branch] = (main.copy(), main.copy())
            else:
                _git(dest, "checkout", "-q", target_branch)
            current = target_branch
        state = main if target_branch == "main" else branches[target_branch][1]
        before = set(state.features)
        if c.merge:
            fork, side = branches[c.merge]
            _git(dest, "merge", "--no-ff", "--no-commit", c.merge, check=False)  # conflicts resolved below
            state.features |= side.features - fork.features
            state.variants |= side.variants - fork.variants
            state.scenarios += [s for s in side.scenarios if s not in state.scenarios]
        state.features |= set(c.features)
        state.variants |= set(c.variants)
        state.variants -= set(c.drop_variants)
        state.scenarios += [s for s in c.scenarios if s not in state.scenarios]
        _write_tree(dest, spec, template, state, state.features - before)
        _git(dest, "add", "-A")
        stamp = f"2026-10-01T{8 + i // 2:02d}:{(i % 2) * 30:02d}:00+00:00"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        _git(dest, "commit", "-q", "-m", c.subject, env=env)
        sha = _git(dest, "rev-parse", "HEAD")
        by_index[i] = sha
        if c.tag:
            _git(dest, "tag", c.tag, env=env)
        commits.append({"index": i + 1, "sha": sha, "subject": c.subject, "tag": c.tag, "branch": target_branch,
                        "merge": c.merge, "features": sorted(state.features), "variants": sorted(state.variants)})
    if current != "main":
        _git(dest, "checkout", "-q", "main")

    first_parent = _git(dest, "rev-list", "--first-parent", "main").splitlines()
    bugs = {}
    for name, bug in spec.bugs.items():
        idx = spec.culprit_index(name)
        sha = by_index[idx]
        entry = {**asdict(bug), "culprit_index": idx + 1, "culprit_sha": sha,
                 "culprit_subject": spec.history[idx].subject}
        if sha not in first_parent:  # planted on a side branch: first-parent bisect lands on the merge
            merge_idx = next(i for i, c in enumerate(spec.history) if c.merge == spec.history[idx].branch)
            entry["merge_sha"] = by_index[merge_idx]
        bugs[name] = entry
    baseline = next(c["sha"] for c in commits if c["tag"] == spec.baseline)
    untestable = [c["sha"] for c in commits if c["subject"] in spec.notes.get("untestable_subjects", ())]
    clean = [baseline] + [c["sha"] for c in commits if c["subject"] in spec.clean_subjects]
    info = FixtureInfo(repo=str(dest), head=_git(dest, "rev-parse", "main"), baseline=baseline,
                       baseline_tag=spec.baseline, commits=commits, bugs=bugs, game=spec.name,
                       untestable=untestable, clean=clean)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    info.save(manifest)
    return info


def _copy_static(spec: GameSpec, dest: Path) -> None:
    """Vendored libraries (or engine bridges), the game's static files and its .gitignore."""
    for src, target in spec.vendor:
        source = REPO_ROOT / src if src.startswith("bridges/") else GAMES_DIR / "vendor" / src
        (dest / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fs_path(source), fs_path(dest / target))
    for name in spec.static_files:
        shutil.copy2(fs_path(spec.template_dir / name), fs_path(dest / name))
    if spec.gitignore:
        (dest / ".gitignore").write_text(spec.gitignore, encoding="utf-8", newline="\n")


def _force_remove(func, path, _exc):
    os.chmod(path, 0o700)
    func(path)


def render_latest(out_dir: Path, *, with_bugs: bool = False, game: str = "crypt") -> Path:
    """Render a playable copy of the newest game (all features), with or without the planted bugs."""
    spec = get_game(game)
    template = Path(fs_path(spec.template_dir / spec.template_file)).read_text(encoding="utf-8")
    features = {f for c in spec.history for f in c.features}
    planted = {b.trigger or n for n, b in spec.bugs.items()}
    variants = {v for c in spec.history for v in c.variants} - {v for c in spec.history for v in c.drop_variants}
    if not with_bugs:
        variants -= planted
        features -= planted
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_static(spec, out_dir)
    (out_dir / "assets").mkdir(exist_ok=True)
    for asset, make in spec.textures.items():
        cv.imwrite(str(out_dir / "assets" / asset), make())
    (out_dir / spec.output_file).write_text(render_template(template, features, variants), encoding="utf-8",
                                            newline="\n")
    (out_dir / MANIFEST).write_text(_manifest(spec, list(spec.scenarios)), encoding="utf-8", newline="\n")
    return out_dir
