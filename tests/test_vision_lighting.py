"""The lighting detectors on synthetic frames: inverted relief and missing shadows."""

import cv2 as cv
import numpy as np

from glitchhunter.vision import DetectContext, detect_object_missing, detect_shading_inverted, detect_shadow_missing

H, W = 180, 320


def wall(inverted=False):
    """Grey cladding with horizontal ribs between two plain areas, lit from above: each rib has a
    bright top and a dark underside. An inverted normal map swaps them."""
    img = np.full((H, W, 3), (118, 122, 128), np.float32)
    y = np.arange(H, dtype=np.float32)[:, None]
    relief = 22.0 * np.sin(2 * np.pi * y / 10.0) * (-1 if inverted else 1)
    img[30:150, 70:250] += relief[30:150, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def floor(seed=3):
    rng = np.random.default_rng(seed)
    small = rng.uniform(90, 150, (H // 8, W // 8)).astype(np.float32)
    base = cv.resize(small, (W, H), interpolation=cv.INTER_CUBIC) + rng.normal(0, 5, (H, W)).astype(np.float32)
    return np.clip(np.stack([base * 0.95, base, base * 1.05], axis=2), 0, 255).astype(np.uint8)


def shadowed(img):
    out = img.astype(np.float32)
    pts = np.array([[120, 60], [215, 70], [230, 130], [130, 125]], np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv.fillPoly(mask, [pts], 255)
    mask = cv.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
    return np.clip(out * (1 - 0.55 * mask), 0, 255).astype(np.uint8)


def test_inverted_relief_is_found_and_localized():
    dets = detect_shading_inverted([wall(True)] * 6, DetectContext(golden=[wall()] * 6))
    assert dets and dets[0].metrics["relief_correlation"] < -0.6
    x, y, w, h = dets[0].region
    assert 50 <= x <= 90 and x + w >= 220


def test_same_or_blurred_relief_is_not_inverted():
    gold = [wall()] * 6
    assert detect_shading_inverted([wall()] * 6, DetectContext(golden=gold)) == []
    blurred = cv.GaussianBlur(wall(), (0, 0), 2.5)
    assert detect_shading_inverted([blurred] * 6, DetectContext(golden=gold)) == []


def test_missing_shadow_is_found_but_not_as_a_missing_object():
    gold = [shadowed(floor())] * 12
    now = [floor()] * 12
    dets = detect_shadow_missing(now, DetectContext(golden=gold))
    assert dets and dets[0].metrics["kind"] == "lit" and dets[0].metrics["brightness_ratio"] > 1.5
    x, y, w, h = dets[0].region
    assert abs(x - 120) < 15 and abs(y - 60) < 15
    assert detect_object_missing(now, DetectContext(golden=gold)) == []


def test_unchanged_or_globally_brighter_scene_has_no_missing_shadow():
    gold = [shadowed(floor())] * 12
    assert detect_shadow_missing([g.copy() for g in gold], DetectContext(golden=gold)) == []
    brighter = [np.clip(g.astype(np.float32) * 1.3, 0, 255).astype(np.uint8) for g in gold]
    assert detect_shadow_missing(brighter, DetectContext(golden=gold)) == []


def test_a_new_dark_area_is_reported_with_less_confidence():
    gold = [floor()] * 12
    dets = detect_shadow_missing([shadowed(floor())] * 12, DetectContext(golden=gold))
    assert dets and dets[0].metrics["kind"] == "dark" and dets[0].confidence <= 0.8
