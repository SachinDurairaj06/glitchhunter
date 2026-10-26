"""Low-level OpenCV 5 tools shared by the MCP server and the HTTP API.

These are the building blocks a developer agent (Claude Code, Kiro, any MCP
client) can call directly: capture a build, run detectors on a run, ray-cast
into the scene, compare against golden frames, list/show commits, bisect.
The autonomous hunt uses the higher-level agent tools in agent/tools.py.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import Settings, load_settings
from .cvinfo import opencv_info
from .lab import BuildError, Lab
from .vision import DETECTORS

_lab: Lab | None = None
_lock = threading.Lock()


def get_lab(settings: Settings | None = None) -> Lab:
    global _lab
    with _lock:
        if _lab is None:
            _lab = Lab(settings or load_settings())
        return _lab


def reset_lab(settings: Settings) -> None:
    """Close the lab and start the next one on other settings (another game)."""
    global _lab
    with _lock:
        if _lab is not None:
            _lab.close()
        _lab = Lab(settings)


def info() -> dict:
    project = get_lab().project
    return {"opencv": opencv_info(),
            "game": {"title": project.title, "baseline": project.baseline, "test_plan": project.source,
                     "viewport": list(project.viewport), "viewports": [list(v) for v in project.viewports]},
            "scenarios": {k: v.description for k, v in project.scenarios.items()},
            "detectors": {k: {"severity": v.severity, "description": v.description, "localized": v.localized,
                              "needs_golden": v.needs_golden, "family": v.family} for k, v in DETECTORS.items()}}


def list_commits(limit: int = 30) -> list[dict]:
    return get_lab().commits(limit)


def show_commit(sha: str) -> dict:
    return get_lab().show_commit(sha)


def capture(commit: str = "main", scenario: str = "overview", camera: dict | None = None, seed: int = 1,
            viewport: list[int] | None = None) -> dict:
    run = get_lab().capture(commit, scenario, camera, seed, purpose="api",
                            viewport=tuple(viewport) if viewport else None)
    return run.summary()


def detect(run_id: str, detectors: list[str] | None = None, region: list[int] | None = None) -> dict:
    lab = get_lab()
    run = lab.get_run(run_id)
    results = lab.inspect(run, detectors, tuple(region) if region else None, tag="api-")
    return {"run_id": run_id, "detections": [{**d.to_dict(), "evidence": paths} for d, paths in results]}


def pick(run_id: str, x: float, y: float, frame: int = 0) -> dict:
    lab = get_lab()
    return lab.pick(lab.get_run(run_id), x, y, frame)


def compare_to_golden(run_id: str) -> dict:
    """Every golden-build comparison for a run: structural diff, tone shift, acne, detail loss, missing
    objects, inverted relief and missing shadows."""
    lab = get_lab()
    run = lab.get_run(run_id)
    names = ["visual_diff", "geometry_explosion", "color_shift", "shadow_acne", "detail_loss", "object_missing",
             "shading_inverted", "shadow_missing"]
    results = lab.inspect(run, names, tag="golden-")
    return {"run_id": run_id, "baseline": lab.project.baseline,
            "detections": [{**d.to_dict(), "evidence": paths} for d, paths in results]}


def bisect(detector: str, scenario: str, good: str | None = None, bad: str = "main",
           camera: dict | None = None, region: list[int] | None = None, repeats: int = 1,
           viewport: list[int] | None = None) -> dict:
    """Visual git bisect with an OpenCV detector as the test (no investigation state needed).

    Commits that do not build are skipped; a merge found as the first bad mainline commit is
    searched inside.
    """
    from .agent.tools import _first_bad

    lab = get_lab()
    sc = lab.scenario(scenario)
    roi = tuple(region) if region else None
    vp = tuple(viewport) if viewport else None
    tested: dict[str, bool | None] = {}

    def is_bad(sha: str) -> bool | None:
        if sha[:10] in tested:
            return tested[sha[:10]]
        for _ in range(max(1, repeats)):
            try:
                run = lab.capture(sha, scenario, camera, keep=False, purpose="api-bisect", viewport=vp)
            except BuildError:
                tested[sha[:10]] = None
                return None
            if lab.inspect(run, [detector], roi, tag="api-bisect-"):
                tested[sha[:10]] = True
                return True
        tested[sha[:10]] = False
        return False

    good_sha, bad_sha = lab.resolve(good or lab.project.baseline), lab.resolve(bad)
    if is_bad(good_sha) is not False:
        return {"error": "good end shows the glitch or does not build", "tested": tested}
    if not is_bad(bad_sha):
        return {"error": "bad end does not show the glitch", "tested": tested}
    commits = lab.commits_between(good_sha, bad_sha)
    idx, _ = _first_bad(commits, is_bad)
    culprit, via_merge = commits[idx], None
    if len(lab.parents(culprit)) > 1:
        side = lab.side_commits(culprit)
        if side and is_bad(side[-1]):
            s_idx, _ = _first_bad(side, is_bad)
            via_merge, culprit = culprit, side[s_idx]
    info = lab.show_commit(culprit)
    return {"culprit": info["sha"][:10], "subject": info["subject"], "diff": info["diff"], "tested": tested,
            "scenario": sc.name, "via_merge": via_merge[:10] if via_merge else None,
            "skipped": [k for k, v in tested.items() if v is None]}


TOOL_FUNCS: dict[str, Any] = {
    "info": info,
    "list_commits": list_commits,
    "show_commit": show_commit,
    "capture": capture,
    "detect": detect,
    "pick": pick,
    "compare_to_golden": compare_to_golden,
    "bisect": bisect,
}
