"""The seven classic glitch detectors (missing texture, z-fighting, frozen and black frames,
flashes, exploded geometry, HUD text overflow)."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .. import evidence as ev
from ..textdnn import get_reader
from .core import (DetectContext, Detection, clean_median, clip01 as _clip01, components as _components,
                   flash_frames, frame_luma, gray_list, kernel as _kernel, median_frame, optical_flow, rel_detail,
                   ssim_map, to_gray_stack, tone_normalize, warp_back)
from .lighting import surface_change

FPS = 30.0


# --- detectors -------------------------------------------------------------------------------------

def detect_missing_texture(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Engine placeholder: magenta cells interleaved with black cells (Source, Godot, three.js
    test textures), or flat saturated magenta (Unity's error shader for a missing or broken
    shader). A purple prop, sign or sky is neither: it is not checkered and not pure magenta."""
    lum = frame_luma(frames)
    med = median_frame(frames, flash_frames(lum))
    hsv = cv.cvtColor(med, cv.COLOR_BGR2HSV)
    magenta = cv.inRange(hsv, (138, 100, 60), (162, 255, 255))
    pure = cv.inRange(hsv, (143, 215, 80), (157, 255, 255))
    dark = cv.inRange(hsv, (0, 0, 0), (180, 255, 50))
    merged = cv.morphologyEx(magenta, cv.MORPH_CLOSE, _kernel(7))
    out = []
    for _, region, area, comp in _components(merged, 25):
        n_mag = int(np.count_nonzero(magenta[comp]))
        n_dark = int(np.count_nonzero(dark[comp]))
        if n_mag < 20:
            continue
        checker = 1.0 - abs(n_mag - n_dark) / max(n_mag + n_dark, 1)
        purity = np.count_nonzero(pure[comp]) / n_mag
        if checker < 0.3 and purity < 0.5:
            continue
        x, y, w, h = region
        persist = float(np.mean([np.count_nonzero(cv.inRange(cv.cvtColor(f[y:y + h, x:x + w], cv.COLOR_BGR2HSV),
                                                              (138, 100, 60), (162, 255, 255))) > 0.3 * n_mag
                                 for f in frames[:: max(1, len(frames) // 8)]]))
        pattern = max(checker, 0.8 * purity)
        score = _clip01(n_mag / 400.0) * (0.6 + 0.4 * pattern)
        confidence = _clip01(0.35 + 0.4 * _clip01(n_mag / 250.0) + 0.25 * pattern) * persist
        kind = "magenta/black checker" if checker >= 0.3 else "flat error magenta (missing or broken shader)"
        det = Detection("missing_texture", True, score, confidence, region, list(range(len(frames))),
                        {"magenta_px": n_mag, "dark_px": n_dark, "checker": checker, "purity": round(purity, 3),
                         "persistence": persist},
                        f"{n_mag} px of {kind} (engine missing-texture placeholder)")
        det.evidence["annotated"] = ev.box(med, region, ev.RED, "missing texture")
        det.evidence["closeup"] = ev.crop(med, region)
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def detect_zfight_flicker(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Surfaces that toggle on and off between frames (z-fighting) with a static camera.

    Animated objects also change over time, so a region only counts when its mean
    brightness swings frame to frame (high temporal roughness), not smoothly. Things
    that move back and forth (a vibrating crate) also toggle pixels at their edges, so
    a region only counts when optical flow cannot explain the change: warping the
    previous frame by the flow leaves most of the difference in place.
    """
    if len(frames) < 6 or not ctx.camera_static:
        return []
    lum = frame_luma(frames)
    keep = ~flash_frames(lum)
    kept = [f for f, k in zip(frames, keep) if k]
    stack = to_gray_stack(kept)
    if len(stack) < 6:
        return []
    std = stack.std(axis=0)
    step = np.abs(np.diff(stack, axis=0)).mean(axis=0)
    cand = ((std > 12) & (step > 10)).astype(np.uint8) * 255
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, _kernel(3))  # drops 1-2 px edge shimmer from camera jitter
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, _kernel(5))
    base = median_frame(frames, ~keep)
    out = []
    for _, region, area, comp in _components(cand, 40):
        series = stack[:, comp].mean(axis=1)
        swing = float(series.std())
        roughness = float(np.abs(np.diff(series)).mean() / (swing + 1e-3))
        if swing < 6.0 or roughness < 0.6:
            continue
        explained = _motion_explained(kept, region)
        if explained > 0.45 or _vibrates(kept, region):
            continue  # moving, not flickering (see physics_jitter)
        toggles = int(np.count_nonzero(np.abs(np.diff(series)) > max(4.0, 0.5 * swing)))
        score = _clip01(swing / 20.0)
        confidence = _clip01(area / 500.0) * _clip01(roughness / 1.0) * _clip01(swing / 10.0)
        det = Detection("zfight_flicker", True, score, confidence, region,
                        [i for i, k in enumerate(keep) if k],
                        {"area_px": area, "swing": swing, "roughness": roughness, "toggles": toggles,
                         "mean_std": float(std[comp].mean()), "motion_explained": explained},
                        f"{area} px region toggles {toggles}x in {len(stack)} frames (z-fighting / flicker)")
        det.evidence["heatmap"] = ev.box(ev.heatmap(base, std, vmax=40), region, ev.YELLOW, "flicker")
        det.evidence["closeup"] = ev.crop(base, region)
        det.evidence["series"] = ev.series_plot(series, title="region brightness per frame")
        out.append(det)
    return sorted(out, key=lambda d: -(d.score * (0.5 + d.confidence)))


def flow_coherence(vectors: np.ndarray) -> float:
    """|mean flow| / mean |flow| over a set of flow vectors: 1 when they all point the same way (a rigid
    move), lower for the inconsistent flow optical flow invents for an intensity toggle."""
    return float(np.linalg.norm(vectors.mean(axis=0)) / (np.linalg.norm(vectors, axis=1).mean() + 1e-6))


def _vibrates(frames: list[np.ndarray], region, pairs: int = 8) -> bool:
    """True when the region moves rigidly back and forth nearly every frame (a vibrating object)."""
    x, y, w, h = region
    H, W = frames[0].shape[:2]
    x0, y0, x1, y1 = max(0, x - 16), max(0, y - 16), min(W, x + w + 16), min(H, y + h + 16)
    crops = gray_list([f[y0:y1, x0:x1] for f in frames[:pairs + 1]])
    if crops[0].shape[0] < 32 or crops[0].shape[1] < 32:
        return False
    inner = (slice(y - y0, y - y0 + h), slice(x - x0, x - x0 + w))
    flows = [optical_flow(a, b)[inner].reshape(-1, 2) for a, b in zip(crops, crops[1:])]
    series = np.array([fl.mean(axis=0) for fl in flows])
    coherence = float(np.median([flow_coherence(fl) for fl in flows]))
    axis = 0 if series[:, 0].std() >= series[:, 1].std() else 1
    s = series[:, axis]
    flips = float(np.mean(np.sign(s[1:]) * np.sign(s[:-1]) < 0))
    # A real vibration moves the object by a pixel or more each frame (physics_jitter asks for the
    # same); an intensity toggle can fool DIS into a coherent sub-pixel shimmer.
    return flips >= 0.6 and float(np.median(np.abs(s))) >= 1.2 and coherence >= 0.85


def _motion_explained(frames: list[np.ndarray], region, pairs: int = 8, pad: int = 16) -> float:
    """Share of a region's frame-to-frame change that optical flow accounts for (0 = none, 1 = all).

    Measured on a padded crop over the first `pairs` consecutive frame pairs: 1 - |cur - warp(prev)| / |cur - prev|.
    """
    x, y, w, h = region
    H, W = frames[0].shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 - x0 < 32 or y1 - y0 < 32:  # DIS needs a few patches of context
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = max(0, cx - 16), min(W, cx + 16)
        y0, y1 = max(0, cy - 16), min(H, cy + 16)
    crops = gray_list([f[y0:y1, x0:x1] for f in frames[:pairs + 1]])
    raw = comp = 0.0
    inner = (slice(y - y0, y - y0 + h), slice(x - x0, x - x0 + w))
    for prev, cur in zip(crops, crops[1:]):
        warped = warp_back(prev, cur)
        raw += float(np.abs(cur.astype(np.float32) - prev.astype(np.float32))[inner].mean())
        comp += float(np.abs(cur.astype(np.float32) - warped.astype(np.float32))[inner].mean())
    if raw < 1e-6:
        return 0.0
    return float(max(0.0, 1.0 - comp / raw))


def detect_frozen_frame(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Consecutive identical frames while the scenario says the picture must keep changing."""
    if not ctx.motion_expected or len(frames) < 8:
        return []
    if ctx.lossy:  # a re-encoded frozen frame is not bit-identical: use the mean change instead
        diffs = np.array([cv.norm(frames[i], frames[i - 1], cv.NORM_L1) / frames[i].size
                          for i in range(1, len(frames))], np.float32)
        static = diffs <= 0.6
    else:
        diffs = np.array([cv.norm(frames[i], frames[i - 1], cv.NORM_INF) for i in range(1, len(frames))],
                         np.float32)
        static = diffs <= 2
    best_start, best_len, run_start = -1, 0, None
    for i, s in enumerate(list(static) + [False]):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    moving_before = int(np.count_nonzero(~static[:max(best_start, 0)]))
    if best_len < 6 or moving_before < 3:
        # never moving at all means the scenario did not exercise anything, which is not a freeze
        return []
    first = best_start + 1  # index of the first frame identical to its predecessor
    tail = best_start + best_len == len(diffs)
    score = _clip01(best_len / 15.0)
    confidence = 0.95 if best_len >= 10 else 0.7
    note = (f"picture stops changing at frame {first} (t={first / FPS:.2f}s) and stays frozen "
            f"for {best_len} frames{' until the end' if tail else ''}")
    det = Detection("frozen_frame", True, score, confidence, None, list(range(best_start, best_start + best_len + 1)),
                    {"frozen_from": first, "frozen_frames": best_len, "until_end": tail,
                     "max_diff_before": float(diffs[:best_start].max()) if best_start > 0 else 0.0},
                    note)
    det.evidence["diff_plot"] = ev.series_plot(diffs, title="max pixel change vs previous frame",
                                               marks=[first], threshold=2)
    det.evidence["before_after"] = ev.side_by_side(
        [frames[max(0, best_start - 6)], frames[first]],
        [f"frame {max(0, best_start - 6)}", f"frame {first}+ (frozen)"])
    return [det]


def detect_black_frame(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Frames where the scene is not drawn (almost every pixel near black)."""
    fracs = []
    for f in frames:
        gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        fracs.append(np.count_nonzero(gray < 32) / gray.size)
    fracs = np.array(fracs, np.float32)
    black = fracs >= 0.85
    count = int(black.sum())
    if count < max(2, len(frames) // 2):
        return []
    idx = [i for i, b in enumerate(black) if b]
    sample = frames[idx[len(idx) // 2]]
    lit = (cv.cvtColor(sample, cv.COLOR_BGR2GRAY) >= 32).astype(np.uint8)
    visible = cv.boundingRect(lit) if np.count_nonzero(lit) else None
    det = Detection("black_frame", True, float(fracs[black].mean()), 0.95, None, idx,
                    {"black_frames": count, "of": len(frames), "dark_fraction": float(fracs[black].mean()),
                     "still_visible_bbox": list(visible) if visible else None},
                    f"{count}/{len(frames)} frames are >85% black; only the UI (if anything) is still drawn")
    det.evidence["frame"] = ev.box(sample, visible, ev.YELLOW, "still drawn") if visible else sample
    det.evidence["dark_plot"] = ev.series_plot(fracs * 100, title="% near-black pixels", threshold=85)
    return [det]


def detect_flash(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Full-screen brightness spikes. More than 3 flashes per second breaks WCAG 2.3.1."""
    if len(frames) < 4:
        return []
    lum = frame_luma(frames)
    flashes = flash_frames(lum)
    if not flashes.any():
        return []
    events, prev = [], False
    for i, f in enumerate(flashes):
        if f and not prev:
            events.append(i)
        prev = bool(f)
    base = float(np.median(lum))
    jump = float(lum.max() - base)
    per_second = len(events) / (len(frames) / FPS)
    worst_window = max(sum(1 for e in events if s <= e < s + int(FPS)) for s in range(len(frames)))
    score = _clip01(jump / 60.0)
    det = Detection("flash", True, score, 0.9, None, [int(i) for i in np.flatnonzero(flashes)],
                    {"flash_events": len(events), "max_jump": jump, "flashes_per_second": per_second,
                     "max_flashes_in_1s": worst_window, "wcag_2_3_1_violation": worst_window > 3},
                    f"{len(events)} full-screen flash(es), brightness +{jump:.0f} over the median "
                    "(photosensitivity risk)")
    det.evidence["luma_plot"] = ev.series_plot(lum, title="mean brightness per frame", marks=events)
    first = events[0]
    det.evidence["flash_frame"] = ev.side_by_side([frames[max(0, first - 1)], frames[first]],
                                                  [f"frame {max(0, first - 1)}", f"frame {first} (flash)"])
    return [det]


def detect_geometry_explosion(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Large spiky regions that differ from the approved golden frame (stretched / exploded meshes)."""
    if not ctx.golden or not ctx.camera_static:
        return []
    lum = frame_luma(frames)
    cur = median_frame(frames, flash_frames(lum))
    gold = median_frame(ctx.golden, flash_frames(frame_luma(ctx.golden)))
    if cur.shape != gold.shape:
        return []
    cur = tone_normalize(cur, gold)  # a global tone change (wrong gamma) is color_shift's finding, not a local one
    # A broken mesh is often close in color to what it covers, so look for structural change
    # (SSIM) as well as perceptual color change (CIE Lab distance).
    ssim = ssim_map(cur, gold)
    lab_cur = cv.cvtColor(cur, cv.COLOR_BGR2Lab).astype(np.float32)
    lab_gold = cv.cvtColor(gold, cv.COLOR_BGR2Lab).astype(np.float32)
    delta_e = np.linalg.norm(lab_cur - lab_gold, axis=2)
    diff = np.maximum((1.0 - ssim) * 120.0, delta_e)
    mask = ((ssim < 0.45) | (delta_e > 20)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _kernel(3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _kernel(9))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    edges_now = cv.Canny(cv.cvtColor(cur, cv.COLOR_BGR2GRAY), 50, 150)
    edges_gold = cv.dilate(cv.Canny(cv.cvtColor(gold, cv.COLOR_BGR2GRAY), 50, 150), _kernel(5))
    edges = cv.bitwise_and(edges_now, cv.bitwise_not(edges_gold))  # edges that are new in this build
    total = cur.shape[0] * cur.shape[1]
    # A broken mesh changes the coarse picture; a new surface pattern (shadow acne, noise) does not.
    g_cur = cv.cvtColor(cur, cv.COLOR_BGR2GRAY).astype(np.float32)
    g_gold = cv.cvtColor(gold, cv.COLOR_BGR2GRAY).astype(np.float32)
    coarse = np.abs(cv.GaussianBlur(g_cur, (0, 0), 4) - cv.GaussianBlur(g_gold, (0, 0), 4))
    pattern = (rel_detail(g_cur, "fine", 21) + 0.01) / (rel_detail(g_gold, "fine", 21) + 0.01)
    # Changes a surface explains: relief shaded the other way, the same texture relit, or a
    # recoloured surface (what a grade leaves after tone normalization: same shapes and
    # lightness, other chroma). Several side by side can merge into one spiky region. (New
    # fine detail is not counted: the spikes of a broken mesh bring plenty of it.)
    recolored = (ssim > 0.6) & (np.abs(lab_cur[..., 0] - lab_gold[..., 0]) < 20)
    surface = surface_change(g_cur, g_gold) | recolored
    # A broken mesh is a local change in an otherwise unchanged picture. When the whole frame
    # still differs after tone normalization (lighting, shadows, grading changed), a spiky
    # difference region says nothing about geometry.
    if float(np.abs(g_cur - g_gold)[mask == 0].mean()) > 4.0:
        return []
    out = []
    for c in contours:
        area = cv.contourArea(c)
        if area < 0.004 * total or area > 0.2 * total:
            continue  # a broken mesh is one object; a change over most of the screen is something else
        hull = cv.convexHull(c)
        solidity = area / max(cv.contourArea(hull), 1.0)
        region = cv.boundingRect(c)
        x, y, w, h = region
        comp = np.zeros_like(mask)
        cv.drawContours(comp, [c], -1, 255, -1)
        coarse_change = float(coarse[comp > 0].mean())
        pattern_change = float(np.median(pattern[comp > 0]))
        if coarse_change < 6 or pattern_change > 3:
            continue  # same coarse picture, or a new surface pattern (shadow acne): not a broken mesh
        inside = comp > 0
        if float(surface[inside].mean()) > 0.5:
            continue  # mostly surfaces shaded or lit differently (lighting detectors), not new geometry
        dead = float(np.mean((cur[inside].max(axis=1) <= 12) & (gold[inside].max(axis=1) > 40)))
        if dead > 0.25:
            continue  # pixels gone pure black where the golden build had content: invalid shader output (nan_pixels)
        new_edges = cv.bitwise_and(edges, comp)
        if np.count_nonzero(new_edges) < 0.02 * area:
            continue  # no new outlines: the same shapes in different tones, not new geometry
        lines = cv.HoughLinesP(new_edges, 1, np.pi / 180, 25, minLineLength=25, maxLineGap=3)
        long_lines = 0 if lines is None else len(lines)
        spiky = solidity < 0.62 or (solidity < 0.75 and long_lines >= 6)
        if not spiky:
            continue
        score = _clip01(area / (0.012 * total)) * _clip01((1 - solidity) * 2.0)
        confidence = _clip01(0.4 + 0.6 * (1 - solidity) + 0.02 * long_lines)
        det = Detection("geometry_explosion", True, score, confidence, region, list(range(len(frames))),
                        {"area_px": float(area), "solidity": float(solidity), "long_edges": long_lines,
                         "new_edge_px": int(np.count_nonzero(new_edges)), "coarse_change": coarse_change},
                        f"{int(area)} px spiky region (solidity {solidity:.2f}) differs from the golden frame")
        vis = ev.heatmap(cur, diff.astype(np.float32), vmax=120, alpha=0.45)
        cv.drawContours(vis, [c], -1, ev.RED, 2)
        det.evidence["diff"] = vis
        det.evidence["golden_vs_now"] = ev.side_by_side([ev.box(gold, region, ev.GREEN), ev.box(cur, region)],
                                                        ["golden", "this build"])
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def _panels_from_rules(edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Panels from pairs of long horizontal rules with the same extent (top and bottom border).

    Text that runs across a panel's side border breaks its outline into pieces, but the top
    and bottom borders stay intact; glyph edges are too short to survive a 40 px opening.
    """
    rules = cv.morphologyEx(edges, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (40, 1)))
    n, _, stats, _ = cv.connectedComponentsWithStats(rules)
    segs = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][2] >= 60]
    panels = []
    for i, (x0, y0, w0, _h0) in enumerate(segs):
        for x1, y1, w1, h1 in segs[i + 1:]:
            if abs(x0 - x1) <= 4 and abs((x0 + w0) - (x1 + w1)) <= 4 and 24 <= abs(y1 - y0) <= 240:
                top, bottom = min(y0, y1), max(y0, y1) + h1
                panels.append((min(x0, x1), top, max(w0, w1), bottom - top))
    return panels


def _flat_fill(gray: np.ndarray, rect) -> bool:
    """UI panels have a flat fill behind their text; a textured 3D surface (a brick wall seen
    head-on is also a screen rectangle) does not."""
    x, y, w, h = rect
    inner = gray[y + 5:y + h - 5, x + 5:x + w - 5].astype(np.float32)
    if inner.size < 200:
        return False
    background = inner[np.abs(inner - np.median(inner)) < 40]  # leave the text out
    return background.size > 0.5 * inner.size and float(background.std()) < 10.0


def _ui_panels(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Axis-aligned rectangular UI panels, found from their outlines."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    raw_edges = cv.Canny(gray, 60, 160)
    edges = cv.dilate(raw_edges, _kernel(3))
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    panels = list(_panels_from_rules(raw_edges))
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue
        x, y, w, h = cv.boundingRect(approx)
        if w < 60 or h < 24 or w * h < 1500 or w * h > 0.9 * img.shape[0] * img.shape[1]:
            continue
        if cv.contourArea(approx) / float(w * h) < 0.9:  # must be (nearly) axis-aligned
            continue
        panels.append((x, y, w, h))
    # outlines are dilated, so the same panel shows up as an inner and an outer contour: keep the outer;
    # rectangles mostly inside a bigger panel are pieces of its outline or contents, not panels
    panels.sort(key=lambda r: -r[2] * r[3])
    kept: list[tuple[int, int, int, int]] = []
    for p in panels:
        if any(abs(p[0] - k[0]) < 8 and abs(p[1] - k[1]) < 8 for k in kept):
            continue
        if any(_overlap(p, k) > 0.6 * p[2] * p[3] for k in kept):
            continue
        if _flat_fill(gray, p):
            kept.append(p)
    return kept


def _overlap(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return float(ix * iy)


def detect_hud_text_overflow(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """UI text that runs past the edge of its panel (OpenCV DNN text detection + panel outlines)."""
    lum = frame_luma(frames)
    img = median_frame(frames, flash_frames(lum))
    panels = _ui_panels(img)
    if not panels:
        return []
    reader = get_reader()
    if reader is None:
        return []
    boxes = reader.read(img)
    out = []
    for px, py, pw, ph in panels:
        inside, overflow = [], []
        for tb in boxes:
            tx, ty, tw, th = tb.rect
            cy = ty + th / 2
            if not (py <= cy <= py + ph):
                continue
            starts_inside = px <= tx + 2 <= px + pw or px <= tx + tw - 2 <= px + pw
            if not starts_inside:
                continue
            over = max(0, (tx + tw) - (px + pw)) + max(0, px - tx)
            (overflow if over > 6 else inside).append((tb, over))
        if not overflow:
            continue
        worst = max(o for _, o in overflow)
        texts = [tb.text for tb, _ in inside + overflow]
        region = (px, py, pw, ph)
        vis = img.copy()
        cv.rectangle(vis, (px, py), (px + pw, py + ph), ev.BLUE, 2)
        for tb, over in inside + overflow:
            color = ev.RED if over > 6 else ev.GREEN
            cv.polylines(vis, [tb.quad.astype(np.int32)], True, color, 2)
        ev.label(vis, f"text runs {worst}px past the panel", (px, py + ph + 18), ev.RED)
        det = Detection("hud_text_overflow", True, _clip01(worst / 60.0), 0.9, region, list(range(len(frames))),
                        {"panel": list(region), "overflow_px": worst, "texts": texts,
                         "overflowing_text": [tb.text for tb, _ in overflow]},
                        f"UI text {[tb.text for tb, _ in overflow]} extends {worst}px beyond its panel")
        det.evidence["annotated"] = vis
        det.evidence["closeup"] = ev.crop(vis, region, pad=60, scale_to=420)
        out.append(det)
    return out


def detect_visual_diff(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Informational: regions whose structure changed versus the golden frame (SSIM < 0.5)."""
    if not ctx.golden:
        return []
    cur = median_frame(frames, flash_frames(frame_luma(frames)))
    gold = median_frame(ctx.golden, flash_frames(frame_luma(ctx.golden)))
    if cur.shape != gold.shape:
        return []
    s = ssim_map(cur, gold)
    mask = (s < 0.5).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _kernel(7))
    out = []
    for _, region, area, comp in _components(mask, 300):
        mean_ssim = float(s[comp].mean())
        det = Detection("visual_diff", True, _clip01((1 - mean_ssim)), 0.5, region, [],
                        {"area_px": area, "mean_ssim": mean_ssim}, f"{area} px changed vs golden (SSIM {mean_ssim:.2f})")
        det.evidence["annotated"] = ev.box(cur, region, ev.YELLOW, "changed")
        out.append(det)
    return sorted(out, key=lambda d: -d.metrics["area_px"])[:5]
