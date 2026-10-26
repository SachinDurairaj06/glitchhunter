"""Ironworks: a larger three.js level with modern engine features and ten harder glitches.

Features: a skinned robot with a keyframed walk cycle, crate physics with substeps
and settling, sun shadow maps, HDR bloom through EffectComposer, spark particles,
LOD smokestacks, an instanced mesh, progressive texture streaming, a follow camera
on its own tick, and a HUD that adapts to the screen size.

The history is not linear: one bug arrives through a merged feature branch, and
one commit does not build at all (bisect has to skip it). Ten bugs are planted
after the approved v1.0 build; one of them only shows up some of the time.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from ..config import GAMES_DIR
from .base import BugSpec, CommitSpec, GameSpec

BRANCH = "feature/robot-polish"

HISTORY: tuple[CommitSpec, ...] = (
    CommitSpec("init: yard, walls, sun and camera", features=("yard",), scenarios=("overview",)),
    CommitSpec("add conveyor belt with crates", features=("conveyor",)),
    CommitSpec("add robot with skeletal walk cycle", features=("robot",), scenarios=("robot_patrol",)),
    CommitSpec("add sun shadow maps", features=("shadows",)),
    CommitSpec("add furnace glow with HDR bloom", features=("bloom",)),
    CommitSpec("add spark particles at the furnace", features=("sparks",)),
    CommitSpec("add smokestacks with LOD", features=("stacks",), scenarios=("flythrough",)),
    CommitSpec("add crate chute and rigid-body drop", features=("chute",), scenarios=("crate_drop",)),
    CommitSpec("add crate stack on a pallet", features=("crate_stack",)),
    CommitSpec("add HUD status and objective panels", features=("hud",)),
    CommitSpec("add player drone and follow camera", features=("drone",), scenarios=("follow_cam",)),
    CommitSpec("add heat haze post pass over the furnace", features=("shimmer",)),
    CommitSpec("add instanced barrels along the wall", features=("barrels",)),
    CommitSpec("stream wall textures progressively", features=("streaming",)),
    CommitSpec("polish lighting for v1.0", variants=("polish",), tag="v1.0"),
    # --- after the approved build ---------------------------------------------------------------
    CommitSpec("add frame telemetry counters", features=("telemetry",)),
    CommitSpec("color grading: lift the shadows so the yard reads better", variants=("color_shift",)),
    CommitSpec("shadows: zero bias and front-face casting for crisper contact shadows", variants=("acne",)),
    CommitSpec("robot: add antenna", features=("antenna",), branch=BRANCH),
    CommitSpec("tweak furnace glow color", variants=("glow_tweak",)),
    CommitSpec("robot: retarget walk clip to the new rig", variants=("anim_retarget",), branch=BRANCH),
    CommitSpec("physics: one substep per frame (4x cheaper)", variants=("tunneling",)),
    CommitSpec("robot: blink the eye light", features=("eye_blink",), branch=BRANCH),
    CommitSpec("Merge branch 'feature/robot-polish'", merge=BRANCH),
    CommitSpec("WIP: inventory screen", features=("inventory",), variants=("wip_inventory",)),
    CommitSpec("fix build: finish the inventory stub", drop_variants=("wip_inventory",)),
    CommitSpec("physics: snappier crate settling", variants=("jitter",)),
    CommitSpec("texture streaming: skip reloads when a mip is resident", variants=("lowres",)),
    CommitSpec("LOD: cheaper far smokestack mesh", variants=("lod_pop",)),
    CommitSpec("heat haze: n*sqrt(n) instead of pow (cheaper)", variants=("nan",)),
    CommitSpec("follow camera: run the rig on a cheaper 20 Hz tick", variants=("stutter",)),
    CommitSpec("HUD: align the objective panel to the right edge", variants=("ui_anchor",)),
    CommitSpec("add credits header", features=("credits",)),
)

BUGS: dict[str, BugSpec] = {
    "color_shift": BugSpec(
        "color_shift", "Whole image washed out: the grading pass encodes sRGB a second time",
        "color_shift", ("overview", "robot_patrol", "crate_drop", "flythrough", "follow_cam")),
    "acne": BugSpec(
        "acne", "Shadow acne: striped self-shadowing on lit surfaces",
        "shadow_acne", ("overview", "crate_drop")),
    "anim_retarget": BugSpec(
        "anim_retarget", "Robot slides in its T-pose: the walk clip no longer binds to the renamed bones",
        "frozen_animation", ("robot_patrol",)),
    "tunneling": BugSpec(
        "tunneling", "Dropped crate sometimes falls through the shelf (physics tunneling)",
        "object_missing", ("crate_drop",), flaky=True),
    "jitter": BugSpec(
        "jitter", "Stacked crates vibrate at rest (settling overshoots every frame)",
        "physics_jitter", ("overview", "robot_patrol")),
    "lowres": BugSpec(
        "lowres", "Walls stay at the blurry 16 px mip (texture streaming never upgrades)",
        "detail_loss", ("overview", "crate_drop")),
    "lod_pop": BugSpec(
        "lod_pop", "Smokestacks visibly pop when they switch level of detail",
        "pop_in", ("flythrough",)),
    "nan": BugSpec(
        "nan", "Heat haze pass writes NaN pixels: black blotches over the furnace",
        "nan_pixels", ("overview",)),
    "stutter": BugSpec(
        "stutter", "Follow camera judders (updates on 2 of every 3 frames)",
        "stutter", ("follow_cam",)),
    "ui_anchor": BugSpec(
        "ui_anchor", "Objective panel is cut off at the screen edge on 4:3 screens",
        "ui_offscreen", ("overview",), viewport=(480, 360)),
}

SCENARIOS: dict[str, dict] = {
    s["name"]: s
    for s in (
        {"name": "overview", "frames": 24,
         "description": "Static wide camera over the yard for 24 frames: conveyor, furnace, robot, crate stack."},
        {"name": "robot_patrol", "frames": 40, "motion_expected": True,
         "description": "Static close camera on the robot's patrol route while it walks; 40 frames.",
         "camera": {"pos": [-6.5, 2.4, 8.0], "target": [-2.6, 1.0, 2.2], "fov": 50}},
        {"name": "crate_drop", "frames": 36, "events": [[1, "down", "KeyE"], [2, "up", "KeyE"]],
         "description": "Press E to drop a crate from the chute onto the shelf; static camera, 36 frames.",
         "camera": {"pos": [4.2, 4.2, 2.8], "target": [8, 2.6, -5.5], "fov": 50}},
        {"name": "flythrough", "frames": 60, "motion_expected": True,
         "description": "Camera flies in from outside the yard toward the smokestacks; 60 frames.",
         "camera_path": [{"frame": 0, "pos": [0, 9, 40], "target": [0, 7, -22], "fov": 55},
                         {"frame": 59, "pos": [0, 6.5, 6], "target": [0, 6, -22], "fov": 55}]},
        {"name": "follow_cam", "frames": 40, "motion_expected": True, "camera_moves": True,
         "events": [[0, "down", "KeyF"], [1, "up", "KeyF"], [1, "down", "KeyW"]],
         "description": "Press F for the follow camera and fly the drone forward (W) for 40 frames."},
    )
}


# --- procedural textures (deterministic seeds, no magenta, no letters) -----------------------------

def _noise(rng: np.random.Generator, size: int, cells: int, amp: float) -> np.ndarray:
    small = rng.uniform(-amp, amp, (cells, cells)).astype(np.float32)
    return cv.resize(small, (size, size), interpolation=cv.INTER_CUBIC)


def _tex_concrete() -> np.ndarray:
    rng = np.random.default_rng(21)
    img = np.full((256, 256, 3), (122, 126, 128), np.float32)
    for cells, amp in ((4, 10), (16, 7), (64, 5), (256, 4)):
        img += _noise(rng, 256, cells, amp)[..., None]
    for _ in range(6):  # stains
        c = (int(rng.integers(0, 256)), int(rng.integers(0, 256)))
        stain = np.zeros((256, 256), np.float32)
        cv.circle(stain, c, int(rng.integers(12, 36)), 1.0, -1)
        img -= cv.GaussianBlur(stain, (0, 0), 9)[..., None] * 14
    for _ in range(4):  # hairline cracks
        p = np.array([rng.uniform(0, 256), rng.uniform(0, 256)])
        for _ in range(30):
            q = p + rng.normal(0, 6, 2)
            cv.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (84, 86, 88), 1, cv.LINE_AA)
            p = q
    for k in (0, 255):  # expansion joints
        img[k, :] = (70, 72, 74)
        img[:, k] = (70, 72, 74)
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_panels() -> np.ndarray:
    rng = np.random.default_rng(22)
    img = np.zeros((256, 256, 3), np.float32)
    x = np.arange(256, dtype=np.float32)
    ribs = 0.5 + 0.5 * np.cos(2 * np.pi * x / 16.0)  # corrugated sheet metal
    img[:] = (np.array([118, 112, 104], np.float32) + 34 * (ribs[None, :, None] - 0.5))
    img += _noise(rng, 256, 32, 6)[..., None]
    for y0 in (0, 128):  # horizontal seams with rivets
        cv.rectangle(img, (0, y0), (255, y0 + 5), (70, 72, 76), -1)
        for rx in range(8, 256, 16):
            cv.circle(img, (rx, y0 + 12), 2, (178, 182, 186), -1, cv.LINE_AA)
            cv.circle(img, (rx + 1, y0 + 13), 2, (60, 62, 66), 1, cv.LINE_AA)
    for _ in range(9):  # rust streaks
        cx = int(rng.integers(0, 256))
        cy = int(rng.integers(10, 200))
        streak = np.zeros((256, 256), np.float32)
        cv.line(streak, (cx, cy), (cx + int(rng.integers(-4, 5)), cy + int(rng.integers(20, 60))), 1.0, 3)
        img += cv.GaussianBlur(streak, (0, 0), 2.5)[..., None] * np.array([-30, -8, 28], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_panels_lo() -> np.ndarray:
    return cv.resize(_tex_panels(), (16, 16), interpolation=cv.INTER_AREA)


def _tex_crate() -> np.ndarray:
    rng = np.random.default_rng(23)
    img = np.full((128, 128, 3), (64, 96, 74), np.float32)
    img += _noise(rng, 128, 16, 8)[..., None]
    cv.rectangle(img, (0, 0), (127, 127), (38, 58, 46), 10)
    cv.line(img, (8, 8), (119, 119), (42, 64, 50), 8)
    cv.line(img, (119, 8), (8, 119), (42, 64, 50), 8)
    for i in range(3):  # chevron hazard marks instead of stencilled letters
        y = 46 + i * 14
        cv.line(img, (50, y), (64, y + 8), (40, 190, 230), 4, cv.LINE_AA)
        cv.line(img, (64, y + 8), (78, y), (40, 190, 230), 4, cv.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_belt() -> np.ndarray:
    img = np.full((64, 64, 3), (44, 46, 50), np.uint8)
    for x in range(0, 64, 8):
        cv.line(img, (x, 0), (x, 63), (78, 82, 88), 2)
    return img


TEXTURES = {
    "concrete.png": _tex_concrete,
    "panels.png": _tex_panels,
    "panels_lo.png": _tex_panels_lo,
    "crate.png": _tex_crate,
    "belt.png": _tex_belt,
}

SPEC = GameSpec(
    name="ironworks",
    title="Ironworks",
    template_dir=GAMES_DIR / "ironworks",
    history=HISTORY,
    bugs=BUGS,
    textures=TEXTURES,
    assets_by_feature={
        "yard": ("concrete.png", "panels.png", "crate.png"),
        "conveyor": ("belt.png",),
        "streaming": ("panels_lo.png",),
    },
    manifest={
        "name": "ironworks", "title": "Ironworks", "baseline": "v1.0", "branch": "main", "entry": "index.html",
        "viewport": [640, 360], "viewports": [[480, 360], [840, 360]], "sweep_scenarios": ["overview"],
        "patch_paths": ["game.js", "index.html", "assets"], "fps": 30,
    },
    scenarios=SCENARIOS,
    vendor=(("three/three.module.js", "vendor/three.module.js"), ("three/three.core.js", "vendor/three.core.js"),
            ("three/LICENSE-three.txt", "vendor/LICENSE-three.txt"),
            *((f"three/addons/{p}", f"vendor/addons/{p}") for p in (
                "postprocessing/EffectComposer.js", "postprocessing/RenderPass.js",
                "postprocessing/UnrealBloomPass.js", "postprocessing/OutputPass.js", "postprocessing/ShaderPass.js",
                "postprocessing/Pass.js", "postprocessing/MaskPass.js", "shaders/CopyShader.js",
                "shaders/LuminosityHighPassShader.js", "shaders/OutputShader.js"))),
    author=("Ironworks Dev", "dev@ironworks.invalid"),
    clean_subjects=("add frame telemetry counters",),
    notes={"untestable_subjects": ("WIP: inventory screen",), "branch": BRANCH},
)
