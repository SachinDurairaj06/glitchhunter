"""Pixel- and layout-level detectors: invalid shader output and UI cut off by the screen edge."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .. import evidence as ev
from ..textdnn import get_reader
from .classic import _ui_panels
from .core import DetectContext, Detection, clean_median, clip01, components, kernel, ring


def detect_nan_pixels(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Blocks of exactly-black pixels inside lit surroundings: NaN or Inf written by a shader.

    A lit 3D scene almost never produces RGB (0,0,0) next to bright pixels; NaN does (it is
    stored as 0 in an 8-bit framebuffer, and blurs and mip chains spread it into blocks).
    """
    sample = frames[:: max(1, len(frames) // 8)]
    freq = np.zeros(frames[0].shape[:2], np.float32)
    dead_level = 12 if ctx.lossy else 1  # video compression rings around pure black
    for f in sample:
        freq += (f.max(axis=2) <= dead_level)
    freq /= len(sample)
    black = (freq >= 0.5).astype(np.uint8) * 255
    if np.count_nonzero(black) < 20:
        return []
    groups = cv.morphologyEx(black, cv.MORPH_CLOSE, kernel(11))
    med = clean_median(frames)
    luma = cv.cvtColor(med, cv.COLOR_BGR2GRAY).astype(np.float32)
    gold_luma = None
    if ctx.golden and ctx.golden[0].shape == frames[0].shape:
        gold_luma = cv.cvtColor(clean_median(ctx.golden), cv.COLOR_BGR2GRAY).astype(np.float32)
    out = []
    for _, region, area, comp in components(groups, 25):
        dead = comp & (black > 0)
        n_dead = int(np.count_nonzero(dead))
        if n_dead < 20:
            continue
        # NaN pixels end abruptly: right next to them the picture is bright. A shadow or a dark
        # object fades into its surroundings, so the pixels next to it are dark too.
        around = ring(dead, 2) & ~(black > 0)
        around_luma = float(np.median(luma[around])) if np.count_nonzero(around) else 0.0
        if around_luma < (80 if ctx.lossy else 45):
            continue  # a dark corner, not dead pixels in a lit area (compression blurs edges: ask for more light)
        blobs = list(components(dead.astype(np.uint8) * 255, 4))
        blockiness = float(np.mean([a / float(r[2] * r[3]) for _, r, a, _ in blobs])) if blobs else 0.0
        was_visible = float(gold_luma[dead].mean()) if gold_luma is not None else None
        if was_visible is not None and was_visible < 30:
            continue  # the golden build is (nearly) black there too
        score = clip01(n_dead / 600)
        confidence = clip01(0.45 + 0.2 * clip01(blockiness) + 0.2 * clip01((around_luma - 45) / 80)
                            + (0.15 if was_visible is not None and was_visible > 40 else 0.0))
        det = Detection("nan_pixels", True, score, confidence, region, list(range(len(frames))),
                        {"dead_px": n_dead, "blocks": len(blobs), "blockiness": blockiness,
                         "surround_luma": around_luma, "golden_luma_there": was_visible,
                         "persistence": float(freq[dead].mean())},
                        f"{n_dead} px of pure black in {len(blobs)} block(s) inside a lit area "
                        f"(surroundings at luma {around_luma:.0f}): NaN/Inf from a shader")
        det.evidence["annotated"] = ev.box(med, region, ev.RED, "NaN pixels")
        det.evidence["closeup"] = ev.crop(med, region, pad=16, scale_to=320)
        if ctx.golden and gold_luma is not None:
            det.evidence["golden_vs_now"] = ev.side_by_side(
                [ev.crop(clean_median(ctx.golden), region, pad=16), ev.crop(med, region, pad=16)],
                ["golden", "this build"])
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def detect_ui_offscreen(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """A UI panel cut off by the edge of the screen, with its text truncated.

    Seen on screen sizes the layout was not built for (4:3, ultrawide) when an element is
    anchored with hard-coded pixel positions. Panels come from their outlines and flat fill;
    text comes from the OpenCV DNN text detector.
    """
    img = clean_median(frames)
    H, W = img.shape[:2]
    panels = _ui_panels(img)
    edge = []
    for p in panels:
        x, y, w, h = p
        at_left, at_right = x <= 2, x + w >= W - 3
        if at_left == at_right:
            continue  # inside the screen, or a full-width bar by design
        edge.append((p, "right" if at_right else "left"))
    if not edge:
        return []
    reader = get_reader()
    boxes = reader.read(img) if reader is not None else []
    out = []
    for (px, py, pw, ph), side in edge:
        inside = [tb for tb in boxes if py <= tb.rect[1] + tb.rect[3] / 2 <= py + ph
                  and tb.rect[0] < px + pw and tb.rect[0] + tb.rect[2] > px]
        cut = [tb for tb in inside if (side == "right" and tb.rect[0] + tb.rect[2] >= W - 4)
               or (side == "left" and tb.rect[0] <= 3)]
        if not cut:
            continue
        cut_ids = {id(tb) for tb in cut}
        texts = [tb.text for tb in inside]
        region = (px, py, pw, ph)
        vis = img.copy()
        cv.rectangle(vis, (px, py), (px + pw - 1, py + ph - 1), ev.BLUE, 2)
        for tb in inside:
            cv.polylines(vis, [tb.quad.astype(np.int32)], True, ev.RED if id(tb) in cut_ids else ev.GREEN, 2)
        edge_x = W - 2 if side == "right" else 1
        cv.line(vis, (edge_x, max(0, py - 10)), (edge_x, min(H - 1, py + ph + 10)), ev.RED, 3)
        ev.label(vis, f"panel cut by the {side} screen edge ({W}x{H})", (8, min(H - 8, py + ph + 22)), ev.RED)
        det = Detection("ui_offscreen", True, 0.8, 0.85, region, list(range(len(frames))),
                        {"panel": list(region), "side": side, "screen": [W, H], "texts": texts,
                         "cut_text": [tb.text for tb in cut], "visible_width_px": pw},
                        f"UI panel runs off the {side} edge of a {W}x{H} screen; text {[tb.text for tb in cut]} "
                        "is cut off")
        det.evidence["annotated"] = vis
        det.evidence["closeup"] = ev.crop(vis, region, pad=30, scale_to=360)
        out.append(det)
    return out
