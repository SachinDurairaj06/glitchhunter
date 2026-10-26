"""Per-game test plan, read from glitchhunter.json in the game's repository.

A studio adds one file to its repo and GlitchHunter knows how to play the game:
which scenarios to run (frames, key presses, camera moves), which build is the
approved baseline, which extra screen sizes to check the UI at, and which files a
proposed fix may touch. The agent reads the file at the branch head, so the test
plan is the same for every commit it tests.

    {
      "name": "ironworks", "title": "Ironworks", "baseline": "v1.0",
      "viewport": [640, 360], "viewports": [[480, 360], [840, 360]],
      "patch_paths": ["game.js", "index.html", "assets"],
      "scenarios": [
        {"name": "overview", "description": "...", "frames": 24},
        {"name": "flythrough", "description": "...", "frames": 60, "motion_expected": true,
         "camera_path": [{"frame": 0, "pos": [0, 9, 40], "target": [0, 7, -22]},
                         {"frame": 59, "pos": [0, 6, 4], "target": [0, 7, -22]}]}
      ]
    }
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "glitchhunter.json"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    frames: int
    events: tuple[tuple[int, str, str], ...] = ()  # (frame index, "down"|"up", KeyboardEvent.code)
    motion_expected: bool = False  # the picture should keep changing (player or camera moves)
    zoomable: bool = True  # world-space camera can be moved for a closer look
    camera: dict | None = None  # starting camera; None = the game's default
    camera_path: tuple[dict, ...] = ()  # keyframes {"frame", "pos", "target", "fov"} applied every frame
    camera_moves: bool = False  # the game moves the camera itself (a follow rig)

    @property
    def camera_static(self) -> bool:
        """True when nothing moves the camera, so any motion on screen is the scene's own."""
        return not self.camera_path and not self.camera_moves

    def camera_at(self, index: int) -> dict | None:
        """Camera for frame `index` on the path (linear interpolation between keyframes)."""
        if not self.camera_path:
            return None
        keys = sorted(self.camera_path, key=lambda k: k["frame"])
        if index <= keys[0]["frame"]:
            return _cam(keys[0])
        for a, b in zip(keys, keys[1:]):
            if a["frame"] <= index <= b["frame"]:
                t = (index - a["frame"]) / max(b["frame"] - a["frame"], 1)
                lerp = lambda u, v: [round(x + (y - x) * t, 5) for x, y in zip(u, v)]  # noqa: E731
                return {"pos": lerp(a["pos"], b["pos"]), "target": lerp(a["target"], b["target"]),
                        "fov": a.get("fov", 55) + (b.get("fov", 55) - a.get("fov", 55)) * t}
        return _cam(keys[-1])

    def to_dict(self) -> dict:
        out = {"name": self.name, "description": self.description, "frames": self.frames}
        if self.events:
            out["events"] = [list(e) for e in self.events]
        for key in ("motion_expected", "camera_moves"):
            if getattr(self, key):
                out[key] = True
        if not self.zoomable:
            out["zoomable"] = False
        if self.camera:
            out["camera"] = self.camera
        if self.camera_path:
            out["camera_path"] = list(self.camera_path)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            name=d["name"], description=d.get("description", ""), frames=int(d["frames"]),
            events=tuple((int(i), str(kind), str(key)) for i, kind, key in d.get("events", [])),
            motion_expected=bool(d.get("motion_expected", False)),
            zoomable=bool(d.get("zoomable", not d.get("camera_path") and not d.get("camera_moves"))),
            camera=d.get("camera"), camera_path=tuple(d.get("camera_path", [])),
            camera_moves=bool(d.get("camera_moves", False)),
        )


def _cam(key: dict) -> dict:
    return {"pos": list(key["pos"]), "target": list(key["target"]), "fov": key.get("fov", 55)}


@dataclass
class Project:
    name: str
    title: str
    baseline: str
    scenarios: dict[str, Scenario]
    branch: str = "main"
    entry: str = "index.html"
    viewport: tuple[int, int] = (640, 360)
    viewports: tuple[tuple[int, int], ...] = ()  # extra screen sizes for the UI sweep
    sweep_scenarios: tuple[str, ...] = ("overview",)
    patch_paths: tuple[str, ...] = ("game.js", "index.html", "assets")
    fps: float = 30.0
    driver: dict = field(default_factory=lambda: {"type": "web"})  # web (browser builds) or bridge (native)
    source: str = "default"  # where the plan came from (for the trace)
    extra: dict = field(default_factory=dict)

    def scenario(self, name: str) -> Scenario:
        if name not in self.scenarios:
            raise KeyError(f"unknown scenario {name!r}; choose from {sorted(self.scenarios)}")
        return self.scenarios[name]

    def allows_patch_path(self, path: str) -> bool:
        path = path.replace("\\", "/")
        if ".." in path.split("/"):
            return False
        return any(path == p or path.startswith(p.rstrip("/") + "/") for p in self.patch_paths)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "title": self.title, "baseline": self.baseline, "branch": self.branch,
            "entry": self.entry, "viewport": list(self.viewport), "viewports": [list(v) for v in self.viewports],
            "sweep_scenarios": list(self.sweep_scenarios), "patch_paths": list(self.patch_paths), "fps": self.fps,
            "driver": self.driver, "scenarios": [s.to_dict() for s in self.scenarios.values()], **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict, source: str = "dict") -> "Project":
        known = {"name", "title", "baseline", "branch", "entry", "viewport", "viewports", "sweep_scenarios",
                 "patch_paths", "fps", "scenarios", "driver"}
        scenarios = {s["name"]: Scenario.from_dict(s) for s in d["scenarios"]}
        return cls(
            name=d.get("name", "game"), title=d.get("title", d.get("name", "game")), baseline=d["baseline"],
            scenarios=scenarios, branch=d.get("branch", "main"), entry=d.get("entry", "index.html"),
            viewport=tuple(d.get("viewport", (640, 360))), viewports=tuple(tuple(v) for v in d.get("viewports", [])),
            sweep_scenarios=tuple(d.get("sweep_scenarios", ["overview"])),
            patch_paths=tuple(d.get("patch_paths", ["game.js", "index.html", "assets"])),
            fps=float(d.get("fps", 30.0)), driver=dict(d.get("driver") or {"type": "web"}), source=source,
            extra={k: v for k, v in d.items() if k not in known},
        )

    @classmethod
    def load(cls, repo: Path, rev: str | None = None) -> "Project":
        """The test plan at `rev` (default: the branch head); the built-in plan if the repo has none."""
        for ref in ([rev] if rev else ["main", "HEAD"]):
            result = subprocess.run(["git", "-c", "core.longpaths=true", "show", f"{ref}:{MANIFEST}"], cwd=repo,
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return cls.from_dict(json.loads(result.stdout), source=f"{ref}:{MANIFEST}")
        return DEFAULT_PROJECT


# Built-in plan for repositories without glitchhunter.json: the Crypt Runner test game.
DEFAULT_PROJECT = Project(
    name="crypt", title="Crypt Runner", baseline="release-1.0",
    scenarios={
        s.name: s
        for s in (
            Scenario("overview", "Static default camera over the whole room for 24 frames.", 24),
            Scenario("walk_to_gem", "Player holds W and walks forward through the gem pickup; 60 frames.", 60,
                     events=((0, "down", "KeyW"),), motion_expected=True),
            Scenario("map_view", "Press M to open the top-down map view; 10 frames.", 10,
                     events=((0, "down", "KeyM"), (1, "up", "KeyM")), zoomable=False),
        )
    },
    source="built-in",
)
SCENARIOS = DEFAULT_PROJECT.scenarios
