"""`glitchhunter init`: add GlitchHunter to a game project, whatever the engine.

Writes a starter glitchhunter.json (a test plan and the driver for the engine) and copies the
engine's bridge into the project:

- web (three.js, Babylon.js, PlayCanvas, any canvas): bridges/web/glitchhunter-test-api.js
- godot: bridges/godot/glitchhunter_bridge.gd (registered as an autoload by hand)
- unity: bridges/unity/GlitchHunterBridge.cs and an editor build script, under Assets/GlitchHunter/
- unreal, custom: the protocol reference (bridges/python/fake_engine.py) to port

Nothing is overwritten unless asked to.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import REPO_ROOT

ENGINES = ("web", "godot", "unity", "unreal", "custom")

_DRIVERS = {
    "web": {"type": "web"},
    "godot": {"type": "bridge",
              "launch": ["{godot}", "--path", "{dir}", "--fixed-fps", "30", "--resolution", "{width}x{height}",
                         "--", "--gh-port", "{port}"],
              "ready_timeout": 90, "error_patterns": ["Parse Error", "Failed to load script", "SCRIPT ERROR"]},
    "unity": {"type": "bridge",
              "build": ["{unity}", "-batchmode", "-quit", "-projectPath", "{dir}", "-executeMethod",
                        "GlitchHunterBuild.Build"],
              "launch": ["{dir}/Build/Game.exe", "-screen-width", "{width}", "-screen-height", "{height}",
                         "-screen-fullscreen", "0", "--gh-port", "{port}"],
              "build_timeout": 1800, "ready_timeout": 120},
    "unreal": {"type": "bridge",
               "build": ["{unreal_uat}", "BuildCookRun", "-project={dir}/Game.uproject", "-platform=Win64",
                         "-clientconfig=Development", "-build", "-cook", "-stage", "-pak", "-archive",
                         "-archivedirectory={dir}/Build"],
               "launch": ["{dir}/Build/Windows/Game.exe", "-windowed", "-ResX={width}", "-ResY={height}",
                          "-gh-port={port}"],
               "build_timeout": 3600, "ready_timeout": 180},
    "custom": {"type": "bridge",
               "launch": ["{dir}/game", "--gh-port", "{port}", "--width", "{width}", "--height", "{height}"],
               "ready_timeout": 60},
}

_FILES = {
    "web": [("bridges/web/glitchhunter-test-api.js", "glitchhunter-test-api.js")],
    "godot": [("bridges/godot/glitchhunter_bridge.gd", "glitchhunter_bridge.gd")],
    "unity": [("bridges/unity/GlitchHunterBridge.cs", "Assets/GlitchHunter/GlitchHunterBridge.cs"),
              ("bridges/unity/Editor/GlitchHunterBuild.cs", "Assets/GlitchHunter/Editor/GlitchHunterBuild.cs")],
    "unreal": [("bridges/python/fake_engine.py", "glitchhunter/reference_bridge.py")],
    "custom": [("bridges/python/fake_engine.py", "glitchhunter/reference_bridge.py")],
}

_NEXT = {
    "web": ["Import glitchhunter-test-api.js from your game and call installTestApi({...}) with your canvas,",
            "camera, scene and a tick() that advances one fixed step and renders (see the comments at its top).",
            "In test mode (?test=1) do not start your own requestAnimationFrame loop."],
    "godot": ["Project > Project Settings > Autoload: add glitchhunter_bridge.gd as 'GlitchHunter'.",
              "Poll input in _process (Input.is_key_pressed) so keys pressed while paused register.",
              "Give important objects colliders, and optionally set_meta('gh_name', 'Name') for readable names."],
    "unity": ["Add GlitchHunterBridge to a GameObject in the first scene, read input through",
              "GlitchHunterBridge.GetKey / GetKeyDown, keep randomness on UnityEngine.Random, and set GH_TOOL_UNITY",
              "to your Unity editor binary (the build step runs it in batch mode with your license)."],
    "unreal": ["Port the nine commands of reference_bridge.py to a GameInstanceSubsystem (see docs/ENGINES.md,",
               "'Unreal Engine'), and set GH_TOOL_UNREAL_UAT to RunUAT.bat."],
    "custom": ["Port the nine commands of reference_bridge.py to your engine (see docs/ENGINES.md, 'The bridge",
               "protocol'), and point driver.launch at your build."],
}


def starter_manifest(name: str, engine: str, baseline: str) -> dict:
    return {
        "name": name, "title": name, "baseline": baseline, "branch": "main",
        "entry": "index.html" if engine == "web" else "glitchhunter.json",
        "viewport": [640, 360], "viewports": [[480, 360], [840, 360]], "sweep_scenarios": ["overview"],
        "fps": 30, "driver": _DRIVERS[engine],
        "scenarios": [
            {"name": "overview", "frames": 24, "description": "Default camera, nothing pressed; 24 frames."},
            {"name": "move", "frames": 30, "motion_expected": True, "events": [[0, "down", "KeyW"], [29, "up", "KeyW"]],
             "description": "Hold W for 30 frames: the player moves."},
        ],
    }


def init_project(project: Path, engine: str, name: str | None = None, baseline: str = "v1.0",
                 force: bool = False) -> list[str]:
    """Scaffold GlitchHunter into a project folder; returns what was done and what to do next."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}")
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    out = []
    manifest = project / "glitchhunter.json"
    if manifest.exists() and not force:
        out.append(f"kept      {manifest} (exists; --force to replace)")
    else:
        data = starter_manifest(name or project.resolve().name, engine, baseline)
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        out.append(f"wrote     {manifest}")
    for src, dest in _FILES[engine]:
        target = project / dest
        if target.exists() and not force:
            out.append(f"kept      {target} (exists)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / src, target)
        out.append(f"copied    {target}")
    out += ["", "next:"] + [f"  {line}" for line in _NEXT[engine]]
    out += [f"  Edit the scenarios in glitchhunter.json, tag your approved build ({baseline}) and commit.",
            "  Check the integration:  python -m glitchhunter bridge-check --repo " + str(project)
            if engine != "web" else "  Try it:  python -m glitchhunter hunt --repo " + str(project),
            "  Hunt:  python -m glitchhunter hunt --repo " + str(project)]
    return out
