"""The native-engine bridge end to end, with a tiny Python "engine" instead of Godot or Unity.

Builds a git repo whose game is bridges/python/fake_engine.py, then captures, ray-casts, skips
a commit that does not start, and bisects a planted glitch, all over the bridge protocol.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from glitchhunter.capture import BuildError
from glitchhunter.config import REPO_ROOT, load_settings

ENGINE = (REPO_ROOT / "bridges" / "python" / "fake_engine.py").read_text(encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-01T08:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-01T08:00:00+00:00"}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def bridge_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("bridge-game")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t.invalid")
    manifest = {
        "name": "fake", "title": "Fake engine", "baseline": "v1.0", "viewport": [320, 180],
        "patch_paths": ["game.py"],
        "driver": {"type": "bridge", "ready_timeout": 30,
                   "launch": [sys.executable, "{dir}/game.py", "--gh-port", "{port}", "--width", "{width}",
                              "--height", "{height}"]},
        "scenarios": [{"name": "overview", "description": "static", "frames": 8},
                      {"name": "slide", "description": "hold D", "frames": 10, "events": [[0, "down", "KeyD"]],
                       "motion_expected": True}],
    }
    (repo / "glitchhunter.json").write_text(json.dumps(manifest, indent=2))
    steps = [
        ("first playable", ENGINE, "v1.0"),
        ("tweak floor", ENGINE.replace("(70, 80, 90)", "(72, 80, 90)"), None),
        ("WIP: does not start", ENGINE.replace("def main() -> None:", "def main() -> None\n"), None),
        ("fix the build", ENGINE.replace("(70, 80, 90)", "(72, 80, 90)"), None),
        ("new crate material", ENGINE.replace("(70, 80, 90)", "(72, 80, 90)").replace(
            'self.w, self.h, self.glitch = width, height, glitch',
            'self.w, self.h, self.glitch = width, height, "missing_texture"'), None),
        ("tidy up", ENGINE.replace("(70, 80, 90)", "(72, 80, 90)").replace(
            'self.w, self.h, self.glitch = width, height, glitch',
            'self.w, self.h, self.glitch = width, height, "missing_texture"') + "\n# tidy\n", None),
    ]
    shas = []
    for subject, code, tag in steps:
        (repo / "game.py").write_text(code, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", subject)
        shas.append(_git(repo, "rev-parse", "HEAD"))
        if tag:
            _git(repo, "tag", tag)
    return repo, shas


@pytest.fixture(scope="module")
def lab(bridge_repo, tmp_path_factory):
    from glitchhunter import service
    from glitchhunter.lab import Lab

    repo, _ = bridge_repo
    settings = load_settings(work_dir=tmp_path_factory.mktemp("work"), repo=repo)
    lab = Lab(settings)
    service._lab = lab
    yield lab
    service._lab = None
    lab.close()


def test_capture_detect_and_pick_over_the_bridge(lab):
    run = lab.capture("main", "overview", keep=False)
    assert len(run.frames) == 8 and run.frames[0].shape == (180, 320, 3)
    dets = lab.inspect(run, ["missing_texture"])
    assert dets
    x, y, w, h = dets[0][0].region
    hit = lab.pick(run, x + w / 2, y + h / 2)
    assert hit["hit"] and hit["object"] == "crate"
    assert lab.inspect(lab.capture("v1.0", "overview", keep=False), ["missing_texture"]) == []


def test_keys_and_viewport_reach_the_game(lab):
    run = lab.capture("v1.0", "slide", keep=False, viewport=(256, 144))
    assert run.frames[0].shape == (144, 256, 3)
    assert (run.frames[0] != run.frames[-1]).any()  # D was held: the crate moved


def test_a_build_that_does_not_start_is_a_build_error(lab, bridge_repo):
    _, shas = bridge_repo
    with pytest.raises(BuildError):
        lab.capture(shas[2], "overview", keep=False)


def test_bisect_over_the_bridge_skips_the_broken_build(lab, bridge_repo):
    from glitchhunter import service

    _, shas = bridge_repo
    out = service.bisect("missing_texture", "overview")
    assert out["culprit"] == shas[4][:10]
    assert set(out["skipped"]) <= {shas[2][:10]}
