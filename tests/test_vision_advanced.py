"""The harder-glitch detectors on synthetic clips (no browser needed)."""

import cv2 as cv
import numpy as np
import pytest

from glitchhunter.textdnn import models_available
from glitchhunter.vision import (DetectContext, detect_color_shift, detect_detail_loss, detect_frozen_animation,
                                 detect_nan_pixels, detect_object_missing, detect_physics_jitter, detect_pop_in,
                                 detect_shadow_acne, detect_stutter, detect_ui_offscreen, detect_zfight_flicker)

H, W = 180, 320


def textured(seed=0, h=H, w=W):
    """A busy, deterministic background (optical flow needs texture to lock on to)."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(40, 200, (h // 6, w // 6, 3)).astype(np.float32)
    img = cv.resize(small, (w, h), interpolation=cv.INTER_CUBIC)
    img += rng.normal(0, 6, (h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def crate(img, x, y, size=36):
    """A textured box pasted at (x, y)."""
    patch = np.zeros((size, size, 3), np.uint8)
    patch[:] = (40, 90, 160)
    for k in range(0, size, 6):
        cv.line(patch, (k, 0), (k, size - 1), (20, 50, 100), 2)
    cv.rectangle(patch, (0, 0), (size - 1, size - 1), (15, 30, 60), 3)
    cv.line(patch, (0, 0), (size - 1, size - 1), (200, 220, 240), 2)
    img[y:y + size, x:x + size] = patch
    return img


# --- motion ------------------------------------------------------------------------------------------

def test_physics_jitter_found_for_a_vibrating_crate():
    bg = textured()
    frames = [crate(bg.copy(), 150, 70 + (3 if t % 2 else -3)) for t in range(24)]
    dets = detect_physics_jitter(frames, DetectContext())
    assert dets and dets[0].metrics["amplitude_px"] > 1.5 and dets[0].metrics["direction_flips"] > 0.8


def test_physics_jitter_ignores_steady_motion_and_toggles():
    bg = textured()
    moving = [crate(bg.copy(), 100 + 2 * t, 70) for t in range(24)]
    assert detect_physics_jitter(moving, DetectContext()) == []
    toggling = []
    for t in range(24):
        f = bg.copy()
        cv.rectangle(f, (150, 60), (200, 100), (200, 200, 40) if t % 2 else (90, 100, 110), -1)
        toggling.append(f)
    assert detect_physics_jitter(toggling, DetectContext()) == []


def test_zfight_does_not_claim_a_vibrating_crate():
    bg = textured()
    frames = [crate(bg.copy(), 150, 70 + (3 if t % 2 else -3)) for t in range(24)]
    assert detect_zfight_flicker(frames, DetectContext()) == []


def pan(n, speed, stall_every=0, seed=1):
    """A camera panning over a wide texture; with stall_every=k every k-th frame repeats the previous one."""
    world = textured(seed, H, W + 6 * n + 20)
    frames, x = [], 0.0
    for t in range(n):
        if not (stall_every and t % stall_every == 0 and t > 0):
            x += speed
        frames.append(world[:, int(x):int(x) + W].copy())
    return frames


def test_stutter_found_for_a_judder_pattern():
    frames = pan(30, 4, stall_every=3)
    dets = detect_stutter(frames, DetectContext(motion_expected=True, camera_static=False))
    assert dets and dets[0].metrics["hitches"] >= 3 and dets[0].metrics["periodic"]


def test_smooth_pan_is_not_stutter():
    frames = pan(30, 4)
    assert detect_stutter(frames, DetectContext(motion_expected=True, camera_static=False)) == []


def test_a_long_freeze_is_not_stutter():
    frames = pan(12, 4) + [pan(12, 4)[-1]] * 14
    assert detect_stutter(frames, DetectContext(motion_expected=True, camera_static=False)) == []


def popping(appear_at=12):
    frames = pan(24, 3, seed=2)
    for t in range(appear_at, 24):
        x = 180 - 3 * (t - appear_at)  # the new object sits in the world, so it pans with the background
        cv.rectangle(frames[t], (x, 50), (x + 40, 110), (30, 30, 230), -1)
    return frames


def test_pop_in_found_when_content_appears_in_one_frame():
    dets = detect_pop_in(popping(), DetectContext(camera_static=False))
    assert dets and dets[0].metrics["frame"] == 12


def test_pop_in_the_golden_build_also_has_is_not_reported():
    frames = popping()
    assert detect_pop_in(frames, DetectContext(camera_static=False, golden=popping())) == []


def walker(articulated: bool, n=30):
    """A stick figure crossing the screen; its legs swing only when articulated."""
    bg = textured(3)
    frames = []
    for t in range(n):
        f = bg.copy()
        x = 60 + 3 * t
        cv.rectangle(f, (x - 10, 40), (x + 10, 100), (230, 120, 40), -1)  # body
        cv.circle(f, (x, 30), 10, (230, 120, 40), -1)
        swing = int(18 * np.sin(2 * np.pi * t / 12)) if articulated else 0
        for side in (-1, 1):
            cv.line(f, (x + 5 * side, 100), (x + 5 * side + side * swing, 150), (200, 90, 30), 8)
            cv.line(f, (x - 10 * side, 50), (x - 10 * side - side * swing, 90), (200, 90, 30), 6)
        frames.append(f)
    return frames


def test_frozen_animation_found_when_a_character_slides():
    dets = detect_frozen_animation(walker(False), DetectContext(golden=walker(True)))
    assert dets and dets[0].metrics["ratio"] < 0.72


def test_animated_character_is_not_frozen():
    assert detect_frozen_animation(walker(True), DetectContext(golden=walker(True))) == []


# --- compare with the golden build ---------------------------------------------------------------------

def test_color_shift_finds_a_double_srgb_encode_and_fits_the_gamma():
    gold = [textured(4)] * 6
    lifted = [np.clip(255 * (g.astype(np.float32) / 255) ** (1 / 2.2), 0, 255).astype(np.uint8) for g in gold]
    dets = detect_color_shift(lifted, DetectContext(golden=gold))
    assert dets and dets[0].metrics["kind"].startswith("washed")
    assert dets[0].metrics["gamma_fit"] == pytest.approx(0.45, abs=0.12)


def test_small_tint_is_not_a_color_shift():
    gold = [textured(4)] * 6
    tinted = [np.clip(g.astype(np.int16) + 3, 0, 255).astype(np.uint8) for g in gold]
    assert detect_color_shift(tinted, DetectContext(golden=gold)) == []


def floor_scene(acne=False):
    img = np.full((H, W, 3), 150, np.uint8)
    img[:60] = (180, 140, 90)
    img = cv.GaussianBlur(np.clip(img.astype(np.int16) + np.random.default_rng(5).normal(0, 2, img.shape), 0, 255)
                          .astype(np.uint8), (0, 0), 1.2)
    if acne:
        yy, xx = np.mgrid[0:H, 0:W]
        rings = (np.sin(np.hypot(xx - 160, yy - 300) * 1.9) > 0.3) & (yy > 70)
        img[rings] = (img[rings] * 0.55).astype(np.uint8)
    return img


def test_shadow_acne_found_on_a_moire_floor():
    dets = detect_shadow_acne([floor_scene(True)] * 6, DetectContext(golden=[floor_scene(False)] * 6))
    assert dets and dets[0].metrics["fine_detail_ratio"] > 2.2


def test_same_floor_has_no_acne():
    assert detect_shadow_acne([floor_scene()] * 6, DetectContext(golden=[floor_scene()] * 6)) == []


def test_detail_loss_found_on_a_blurred_texture():
    gold = textured(6)
    blurred = gold.copy()
    blurred[40:160, 60:280] = cv.GaussianBlur(gold, (0, 0), 3.0)[40:160, 60:280]
    dets = detect_detail_loss([blurred] * 6, DetectContext(golden=[gold] * 6))
    assert dets
    x, y, w, h = dets[0].region
    assert 40 <= x + w / 2 <= 300 and 30 <= y + h / 2 <= 170


def test_object_missing_found_when_a_resting_object_is_gone():
    bg = textured(7)
    gold = [crate(bg.copy(), 140, 60, 40) for _ in range(24)]
    now = [bg.copy() for _ in range(24)]
    dets = detect_object_missing(now, DetectContext(golden=gold))
    assert dets
    x, y, w, h = dets[0].region
    assert abs(x - 140) < 12 and abs(y - 60) < 12


def test_object_still_there_is_not_missing():
    bg = textured(7)
    gold = [crate(bg.copy(), 140, 60, 40) for _ in range(24)]
    assert detect_object_missing([g.copy() for g in gold], DetectContext(golden=gold)) == []


# --- pixels --------------------------------------------------------------------------------------------

def test_nan_blocks_in_a_lit_area_are_found():
    img = np.full((H, W, 3), (60, 140, 230), np.uint8)
    for x, y in ((120, 50), (132, 62), (144, 50), (156, 62)):  # a checker of blocks, as a NaN haze pass leaves
        img[y:y + 12, x:x + 12] = 0
    dets = detect_nan_pixels([img] * 8, DetectContext())
    assert len(dets) == 1 and dets[0].metrics["dead_px"] >= 4 * 144 - 8


def test_dark_corner_is_not_nan():
    img = np.full((H, W, 3), (60, 140, 230), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    shade = np.clip((xx - 200) / 120.0, 0, 1)[..., None]  # a soft fade to black
    img = (img * (1 - shade)).astype(np.uint8)
    assert detect_nan_pixels([img] * 8, DetectContext()) == []


@pytest.mark.skipif(not models_available(), reason="OpenCV Zoo text models not downloaded")
def test_ui_panel_cut_by_the_screen_edge():
    img = np.full((360, 480, 3), (120, 110, 100), np.uint8)
    cv.rectangle(img, (370, 16), (600, 76), (18, 34, 64), -1)
    cv.rectangle(img, (370, 16), (600, 76), (112, 176, 255), 3)
    cv.putText(img, "SHIP 3 CRATES", (384, 54), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    dets = detect_ui_offscreen([img] * 4, DetectContext())
    assert dets and dets[0].metrics["side"] == "right"
