"""Detector tests on synthetic frames (no browser needed)."""

import cv2 as cv
import numpy as np
import pytest

from glitchhunter.textdnn import models_available
from glitchhunter.vision import (DetectContext, detect_black_frame, detect_flash, detect_frozen_frame,
                                 detect_geometry_explosion, detect_hud_text_overflow, detect_missing_texture,
                                 detect_zfight_flicker, run_detectors, ssim_map)

H, W = 180, 320


def scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (90, 100, 110), np.uint8)
    noise = cv.resize(rng.uniform(-12, 12, (9, 16)).astype(np.float32), (W, H), interpolation=cv.INTER_CUBIC)
    img = np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    cv.rectangle(img, (20, 120), (90, 170), (40, 70, 120), -1)
    return img


def checker(img, x, y, size=48, cell=6):
    for j in range(size // cell):
        for i in range(size // cell):
            color = (255, 0, 255) if (i + j) % 2 == 0 else (0, 0, 0)
            cv.rectangle(img, (x + i * cell, y + j * cell), (x + (i + 1) * cell - 1, y + (j + 1) * cell - 1), color, -1)
    return img


def test_missing_texture_found_and_localized():
    frames = [checker(scene(), 150, 60) for _ in range(6)]
    dets = detect_missing_texture(frames, DetectContext())
    assert dets and dets[0].present
    x, y, w, h = dets[0].region
    assert abs(x - 150) <= 4 and abs(y - 60) <= 4 and w >= 40
    assert dets[0].confidence > 0.6


def test_missing_texture_ignores_clean_scene():
    assert detect_missing_texture([scene() for _ in range(6)], DetectContext()) == []


def test_flat_error_magenta_is_a_missing_shader_but_a_purple_sign_is_not():
    unity = scene()
    cv.rectangle(unity, (150, 60), (210, 110), (255, 0, 255), -1)  # Unity's error shader: flat, pure magenta
    dets = detect_missing_texture([unity] * 6, DetectContext())
    assert dets and "error magenta" in dets[0].note
    sign = scene()
    cv.rectangle(sign, (150, 60), (210, 110), (170, 60, 180), -1)  # a purple prop, not a placeholder
    assert detect_missing_texture([sign] * 6, DetectContext()) == []


def test_zfight_toggle_detected_but_smooth_animation_is_not():
    toggling, smooth = [], []
    for t in range(24):
        a = scene()
        cv.rectangle(a, (150, 60), (200, 100), (200, 200, 40) if t % 2 else (90, 100, 110), -1)
        toggling.append(a)
        b = scene()
        level = int(140 + 60 * np.sin(2 * np.pi * t / 24))  # slow, smooth change: animation, not flicker
        cv.rectangle(b, (150, 60), (200, 100), (level, level, level), -1)
        smooth.append(b)
    assert detect_zfight_flicker(toggling, DetectContext())
    assert detect_zfight_flicker(smooth, DetectContext()) == []


def test_zfight_ignores_one_pixel_edge_shimmer():
    frames = []
    for t in range(24):
        f = scene()
        x = 160 + (t % 2)  # a hard edge that moves by one pixel every frame (camera jitter)
        f[:, x:] = (30, 30, 30)
        frames.append(f)
    assert detect_zfight_flicker(frames, DetectContext()) == []


def moving_then_frozen(frozen_from=12, n=30):
    frames = []
    for t in range(n):
        k = min(t, frozen_from)
        f = scene()
        cv.circle(f, (40 + 5 * k, 60), 10, (255, 120, 40), -1)
        frames.append(f)
    return frames


def test_frozen_frame_needs_expected_motion():
    frames = moving_then_frozen()
    dets = detect_frozen_frame(frames, DetectContext(motion_expected=True))
    assert dets and dets[0].metrics["frozen_from"] == 13
    assert detect_frozen_frame(frames, DetectContext(motion_expected=False)) == []


def test_never_moving_scene_is_not_a_freeze():
    frames = [scene() for _ in range(30)]
    assert detect_frozen_frame(frames, DetectContext(motion_expected=True)) == []


def test_black_frames():
    frames = [np.zeros((H, W, 3), np.uint8) for _ in range(8)]
    cv.rectangle(frames[0], (5, 5), (60, 30), (255, 255, 255), -1)
    dets = detect_black_frame(frames, DetectContext())
    assert dets and dets[0].metrics["black_frames"] == 8
    assert detect_black_frame([scene() for _ in range(8)], DetectContext()) == []


def test_flash_and_wcag_count():
    frames = [scene() for _ in range(30)]
    for i in (3, 10, 17, 24):  # four flashes inside one second at 30 fps
        frames[i] = np.full_like(frames[i], 250)
    dets = detect_flash(frames, DetectContext())
    assert dets and dets[0].metrics["flash_events"] == 4
    assert dets[0].metrics["wcag_2_3_1_violation"] is True
    assert detect_flash([scene() for _ in range(30)], DetectContext()) == []


def star(img, center, r_out, r_in, points=9, color=(40, 60, 90)):
    cx, cy = center
    pts = []
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = np.pi * k / points
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    cv.fillPoly(img, [np.array(pts, np.int32)], color)
    return img


def test_geometry_explosion_spiky_vs_compact():
    # The rock is darker than the floor, as in a lit render: the spikes bring new outlines.
    golden = [cv.circle(scene(), (160, 90), 18, (40, 60, 90), -1) for _ in range(4)]
    exploded = [star(scene(), (160, 90), 80, 12) for _ in range(4)]
    dets = detect_geometry_explosion(exploded, DetectContext(golden=golden))
    assert dets and dets[0].metrics["solidity"] < 0.62
    moved = [cv.rectangle(scene(), (120, 60), (200, 120), (20, 160, 220), -1) for _ in range(4)]
    assert detect_geometry_explosion(moved, DetectContext(golden=golden)) == []
    assert detect_geometry_explosion(exploded, DetectContext()) == []  # needs a golden frame


def test_ssim_identical_is_one():
    img = scene()
    assert float(ssim_map(img, img).mean()) == pytest.approx(1.0, abs=1e-4)


def test_run_detectors_reports_full_frame_coordinates_inside_roi():
    frames = [checker(scene(), 150, 60) for _ in range(6)]
    dets = run_detectors(frames, ["missing_texture"], DetectContext(roi=(120, 40, 120, 100)))
    assert dets and abs(dets[0].region[0] - 150) <= 4


@pytest.mark.skipif(not models_available(), reason="OpenCV Zoo text models not downloaded")
def test_hud_text_overflow_with_dnn():
    def hud(text):
        img = np.full((360, 640, 3), (60, 70, 80), np.uint8)
        cv.rectangle(img, (18, 18), (188, 78), (74, 42, 27), -1)
        cv.rectangle(img, (18, 18), (188, 78), (255, 168, 111), 3)
        cv.putText(img, text, (28, 58), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        return [img] * 3

    over = detect_hud_text_overflow(hud("HIT POINTS 100 SCORE 0"), DetectContext())
    assert over and over[0].metrics["overflow_px"] > 6
    assert detect_hud_text_overflow(hud("HP 100"), DetectContext()) == []
