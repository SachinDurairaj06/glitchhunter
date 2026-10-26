"""Crypt Runner: the small dungeon scene with seven classic glitches (17 commits, linear history)."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from ..config import GAMES_DIR
from ..project import DEFAULT_PROJECT
from .base import BugSpec, CommitSpec, GameSpec

HISTORY: tuple[CommitSpec, ...] = (
    CommitSpec("init: room, lights and camera", features=("room",), scenarios=("overview",)),
    CommitSpec("add wooden crate prop", features=("crate",)),
    CommitSpec("add stone pillar and rune decal", features=("pillar_decal",)),
    CommitSpec("add rock prop with procedural noise", features=("rock",)),
    CommitSpec("add player movement and gem pickup", features=("player_gem",), scenarios=("walk_to_gem",)),
    CommitSpec("add HUD panel", features=("hud",)),
    CommitSpec("add TAA-style camera jitter", features=("jitter",)),
    CommitSpec("add map view on the M key", features=("mapview",), tag="release-1.0", scenarios=("map_view",)),
    CommitSpec("tweak torch color", variants=("torch_tweak",)),
    CommitSpec("reorganize texture paths", variants=("missing_texture",)),
    CommitSpec("snap rune decal flush to the floor", variants=("z_fight",)),
    CommitSpec("add ambient lightning flashes", features=("lightning",)),
    CommitSpec("tune rock noise amplitude", variants=("explosion",)),
    CommitSpec("use per-gem reward values on pickup", variants=("frozen",)),
    CommitSpec("localize HUD labels", variants=("hud_overflow",)),
    CommitSpec("extend map view draw distance", variants=("black_map",)),
    CommitSpec("add credits header", features=("credits",)),
)

BUGS: dict[str, BugSpec] = {
    "missing_texture": BugSpec(
        "missing_texture", "Crate renders with the magenta missing-texture placeholder",
        "missing_texture", ("overview", "walk_to_gem")),
    "z_fight": BugSpec(
        "z_fight", "Rune decal z-fights with the floor and flickers",
        "zfight_flicker", ("overview",)),
    "flaky_flash": BugSpec(
        "flaky_flash", "Random full-screen white flashes (photosensitivity risk)",
        "flash", ("overview", "walk_to_gem", "map_view"), flaky=True, trigger="lightning"),
    "explosion": BugSpec(
        "explosion", "Rock mesh explodes into spikes",
        "geometry_explosion", ("overview", "walk_to_gem")),
    "frozen": BugSpec(
        "frozen", "Game freezes when the player picks up the gem",
        "frozen_frame", ("walk_to_gem",)),
    "hud_overflow": BugSpec(
        "hud_overflow", "HUD text overflows its panel",
        "hud_text_overflow", ("overview", "walk_to_gem")),
    "black_map": BugSpec(
        "black_map", "Map view renders a black screen",
        "black_frame", ("map_view",)),
}


# --- procedural textures (deterministic: fixed seeds, no magenta anywhere) -------------------------

def _noise(rng: np.random.Generator, size: int, cells: int, amp: float) -> np.ndarray:
    small = rng.uniform(-amp, amp, (cells, cells)).astype(np.float32)
    return cv.resize(small, (size, size), interpolation=cv.INTER_CUBIC)


def _tex_floor() -> np.ndarray:
    rng = np.random.default_rng(11)
    img = np.zeros((128, 128, 3), np.float32)
    for ty in range(2):
        for tx in range(2):
            base = np.array([92, 104, 118], np.float32) + rng.uniform(-8, 8, 3)
            img[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64] = base
    img += _noise(rng, 128, 8, 9)[..., None]
    for k in (0, 63, 64, 127):
        img[k, :] = (44, 48, 54)
        img[:, k] = (44, 48, 54)
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_bricks() -> np.ndarray:
    rng = np.random.default_rng(12)
    img = np.full((128, 128, 3), (96, 100, 104), np.float32)
    for row in range(4):
        y0 = row * 32
        offset = 32 if row % 2 else 0
        for col in range(-1, 3):
            x0 = col * 64 + offset
            color = np.array([58, 74, 124], np.float32) + rng.uniform(-10, 10, 3)
            cv.rectangle(img, (x0 + 2, y0 + 2), (x0 + 61, y0 + 29), color.tolist(), -1)
    img += _noise(rng, 128, 16, 7)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_crate() -> np.ndarray:
    rng = np.random.default_rng(13)
    img = np.full((128, 128, 3), (52, 112, 168), np.float32)
    for x in range(0, 128, 32):
        cv.line(img, (x, 0), (x, 127), (34, 78, 120), 2)
    grain = _noise(rng, 128, 32, 12)
    img += grain[..., None]
    cv.rectangle(img, (0, 0), (127, 127), (30, 66, 108), 12)
    cv.line(img, (10, 10), (117, 117), (30, 66, 108), 10)
    return np.clip(img, 0, 255).astype(np.uint8)


def _tex_rune() -> np.ndarray:
    img = np.full((128, 128, 3), (62, 56, 50), np.uint8)
    teal = (190, 200, 40)
    cv.circle(img, (64, 64), 50, teal, 6, cv.LINE_AA)
    pts = np.array([[64, 26], [98, 88], [30, 88]], np.int32)
    cv.polylines(img, [pts], True, teal, 5, cv.LINE_AA)
    cv.circle(img, (64, 68), 9, teal, -1, cv.LINE_AA)
    return img


TEXTURES = {
    "floor_tiles.png": _tex_floor,
    "wall_bricks.png": _tex_bricks,
    "crate_wood.png": _tex_crate,
    "rune.png": _tex_rune,
}

_manifest = DEFAULT_PROJECT.to_dict()
SPEC = GameSpec(
    name="crypt",
    title="Crypt Runner",
    template_dir=GAMES_DIR / "crypt",
    history=HISTORY,
    bugs=BUGS,
    textures=TEXTURES,
    assets_by_feature={
        "room": ("floor_tiles.png", "wall_bricks.png"),
        "crate": ("crate_wood.png",),
        "pillar_decal": ("rune.png",),
    },
    manifest={k: v for k, v in _manifest.items() if k != "scenarios"},
    scenarios={s["name"]: s for s in _manifest["scenarios"]},
    vendor=(("three/three.module.js", "vendor/three.module.js"), ("three/three.core.js", "vendor/three.core.js"),
            ("three/LICENSE-three.txt", "vendor/LICENSE-three.txt")),
    author=("Crypt Runner Dev", "dev@cryptrunner.invalid"),
    clean_subjects=("tweak torch color",),
)
