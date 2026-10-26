"""Detectors that compare a build with the approved (golden) build of the same scenario.

They separate global tone changes from local ones: color_shift measures the global
change (and estimates the gamma error), the others first histogram-match the frame
to the golden one so that a darker build does not look like a hundred local bugs.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .. import evidence as ev
from .core import (DetectContext, Detection, clean_median, clip01, components, interior_point, kernel, lab,
                   match_to, mostly_black, rel_detail, ring, ssim_map, tone_normalize)
from .lighting import looks_relit, relit


def _gray(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)


def _tone_plot(gold_l: np.ndarray, cur_l: np.ndarray, gamma: float | None, size: int = 220) -> np.ndarray:
    """Tone curve: median current lightness per golden lightness bin (identity = no change)."""
    img = np.full((size, size, 3), 24, np.uint8)
    pad = 24
    span = size - 2 * pad

    def pt(x, y):
        return int(pad + x / 100 * span), int(size - pad - y / 100 * span)

    cv.rectangle(img, pt(0, 0), pt(100, 100), (70, 70, 70), 1)
    cv.line(img, pt(0, 0), pt(100, 100), (90, 90, 90), 1, cv.LINE_AA)
    bins = np.linspace(0, 100, 26)
    pts = []
    for lo, hi in zip(bins, bins[1:]):
        sel = (gold_l >= lo) & (gold_l < hi)
        if np.count_nonzero(sel) > 30:
            pts.append(pt((lo + hi) / 2, float(np.median(cur_l[sel]))))
    if len(pts) > 1:
        cv.polylines(img, [np.array(pts, np.int32)], False, (90, 220, 255), 2, cv.LINE_AA)
    if gamma:
        curve = [pt(x, 100 * (x / 100) ** gamma) for x in np.linspace(0, 100, 40)]
        cv.polylines(img, [np.array(curve, np.int32)], False, (60, 60, 200), 1, cv.LINE_AA)
    ev.label(img, "tone curve: golden L -> this build L", (4, 14), ev.WHITE, 0.38)
    if gamma:
        ev.label(img, f"fit: L^{gamma:.2f}", (pad + 4, size - pad - 6), (120, 120, 240), 0.4)
    return img


def detect_color_shift(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """The whole picture changed tone or color versus the golden build while the content stayed the same.

    Typical causes: linear values sent to an sRGB display (gamma), a lost tone-mapping or exposure
    setting, a grading or saturation change, a wrong color space on a texture batch. Only pixels
    whose texture did not change are measured, so a local glitch that darkens a big surface (shadow
    acne, a blurred texture) is not mistaken for a global tone change. The lightness curve between
    the builds is fitted with a power law: a 2.2 gamma error reads as ~2.2 (or ~0.45 the other way).
    """
    if not ctx.golden:
        return []
    cur = clean_median(frames)
    gold = clean_median(ctx.golden)
    if cur.shape != gold.shape or mostly_black(cur) or mostly_black(gold):
        return []
    g_cur, g_gold = _gray(cur), _gray(gold)
    fine = (rel_detail(g_cur, "fine", 21) + 0.01) / (rel_detail(g_gold, "fine", 21) + 0.01)
    valid = (fine < 1.8) & (fine > 0.55) & (g_gold > 8)
    if valid.mean() < 0.2:
        return []
    lc, lg = lab(cur), lab(gold)
    # Same scene? Structure must survive once the tone is matched back.
    matched = match_to(cur, gold)
    # Median, not mean: other bugs in the same build (an inverted normal map over a wall, a
    # missing object) pull the mean down without making it a different picture.
    same_scene = float(np.median(ssim_map(matched, gold)[valid]))
    if same_scene < 0.55:
        return []  # a different picture altogether (camera moved, scene changed), not a tone change
    l_cur, l_gold = lc[..., 0], lg[..., 0]
    c_cur, c_gold = np.hypot(lc[..., 1], lc[..., 2]), np.hypot(lg[..., 1], lg[..., 2])
    d_l = float(np.median(l_cur[valid]) - np.median(l_gold[valid]))
    contrast = float((l_cur[valid].std() + 1e-3) / (l_gold[valid].std() + 1e-3))
    chroma = float((c_cur[valid].mean() + 1e-3) / (c_gold[valid].mean() + 1e-3))
    tint = (float(lc[..., 1][valid].mean() - lg[..., 1][valid].mean()),
            float(lc[..., 2][valid].mean() - lg[..., 2][valid].mean()))

    def hist_distance(a, b, top):
        ha = cv.calcHist([a.astype(np.float32).reshape(-1, 1)], [0], None, [50], [0, top])
        hb = cv.calcHist([b.astype(np.float32).reshape(-1, 1)], [0], None, [50], [0, top])
        cv.normalize(ha, ha, 1, 0, cv.NORM_L1)
        cv.normalize(hb, hb, 1, 0, cv.NORM_L1)
        return float(cv.compareHist(ha, hb, cv.HISTCMP_BHATTACHARYYA))

    hist_l = hist_distance(l_cur[valid], l_gold[valid], 100)
    hist_c = hist_distance(c_cur[valid], c_gold[valid], 80)
    change = max(hist_l, hist_c)
    # How global is it? Lightness and chroma per tile of an 8x6 grid, same direction in most tiles.
    h, w = l_cur.shape
    d_ls, d_cs = [], []
    for ty in range(6):
        for tx in range(8):
            sl = (slice(ty * h // 6, (ty + 1) * h // 6), slice(tx * w // 8, (tx + 1) * w // 8))
            v = valid[sl]
            if v.mean() < 0.3:
                continue
            d_ls.append(float(np.median(l_cur[sl][v]) - np.median(l_gold[sl][v])))
            d_cs.append(float(np.log((c_cur[sl][v].mean() + 1.0) / (c_gold[sl][v].mean() + 1.0))))

    def consistency(values, threshold):
        values = np.array(values)
        if not len(values):
            return 0.0
        moved = np.abs(values) > threshold
        if not moved.any():
            return 0.0
        same = max(np.mean(values[moved] > 0), np.mean(values[moved] < 0))
        return float(moved.mean() * same)

    coverage = max(consistency(d_ls, 3.0), consistency(d_cs, 0.2))
    # Gamma fit on mid-tones: L_cur/100 = (L_gold/100) ^ gamma
    sel = valid & (l_gold > 15) & (l_gold < 85) & (l_cur > 1)
    gamma = None
    if np.count_nonzero(sel) > 500:
        x = np.log(l_gold[sel] / 100.0)
        y = np.log(l_cur[sel] / 100.0)
        gamma = float(np.median(y / np.minimum(x, -1e-3)))
    tone_off = abs(d_l) > 8 or abs(np.log(contrast)) > 0.3 or abs(np.log(chroma)) > 0.35 or np.hypot(*tint) > 8
    if not (tone_off and change > 0.15 and coverage > 0.55):
        return []
    if d_l < -8:
        kind = "too dark"
    elif d_l > 8:
        kind = "washed out / too bright"
    elif abs(np.log(chroma)) > 0.35:
        kind = "desaturated" if chroma < 1 else "oversaturated"
    else:
        kind = "color tint"
    gamma_note = ""
    if kind in ("too dark", "washed out / too bright") and gamma and abs(gamma - 2.2) < 0.5:
        gamma_note = "; tone curve fits gamma ~2.2: linear values shown on an sRGB display"
    elif kind in ("too dark", "washed out / too bright") and gamma and abs(gamma - 0.45) < 0.15:
        gamma_note = "; tone curve fits gamma ~0.45: sRGB encoded twice"
    score = clip01(change / 0.4)
    confidence = clip01(0.45 + 0.35 * same_scene + 0.2 * coverage)
    det = Detection("color_shift", True, score, confidence, None, list(range(len(frames))),
                    {"median_L_change": d_l, "contrast_ratio": contrast, "chroma_ratio": chroma,
                     "tint_ab": tint, "lightness_hist_distance": hist_l, "chroma_hist_distance": hist_c,
                     "same_scene_ssim": same_scene, "global_coverage": coverage, "measured_share": float(valid.mean()),
                     "gamma_fit": gamma, "kind": kind},
                    f"whole picture {kind} vs the approved build (median L {d_l:+.0f}, chroma x{chroma:.2f}, "
                    f"{coverage:.0%} of the screen){gamma_note}")
    det.evidence["golden_vs_now"] = ev.side_by_side([gold, cur], ["golden", "this build"])
    det.evidence["tone_curve"] = _tone_plot(l_gold[valid], l_cur[valid], gamma)
    return [det]


_rel_detail = rel_detail


def _stripe_coherence(gray: np.ndarray, mask: np.ndarray) -> float:
    """Structure-tensor coherence of the fine detail inside a mask (1 = parallel stripes)."""
    band = cv.GaussianBlur(gray, (0, 0), 0.7) - cv.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv.Sobel(band, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(band, cv.CV_32F, 0, 1, ksize=3)
    jxx = cv.GaussianBlur(gx * gx, (0, 0), 4)
    jyy = cv.GaussianBlur(gy * gy, (0, 0), 4)
    jxy = cv.GaussianBlur(gx * gy, (0, 0), 4)
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coherence = lam / (jxx + jyy + 1e-6)
    return float(coherence[mask].mean()) if mask.any() else 0.0


def _zoom_box(region, w: int = 120, h: int = 68) -> tuple[int, int, int, int]:
    x, y, rw, rh = region
    cx, cy = x + rw // 2, y + rh // 2
    return max(0, cx - w // 2), max(0, cy - h // 2), w, h


def detect_shadow_acne(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Surfaces covered in a new fine stripe/moire pattern that the golden build does not have.

    Shadow acne is a shadow map shadowing its own surface: lit surfaces gain thin rings and
    stripes and get darker. Measured as contrast-normalized fine detail (what a 3x3 median
    removes, divided by local brightness), which multiplies on acne-covered surfaces while a
    global tone change barely moves it.
    """
    if not ctx.golden or not ctx.camera_static:
        return []
    cur = clean_median(frames)
    gold = clean_median(ctx.golden)
    if cur.shape != gold.shape or mostly_black(cur):
        return []
    g_cur, g_gold = _gray(cur), _gray(gold)
    f_cur, f_gold = _rel_detail(g_cur, "fine", 21), _rel_detail(g_gold, "fine", 21)
    ratio = (f_cur + 0.01) / (f_gold + 0.01)
    darker = cv.GaussianBlur(g_gold - g_cur, (0, 0), 6)
    cand = ((ratio > 2.2) & (f_cur > 0.03)).astype(np.uint8) * 255
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, kernel(5))
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel(15))
    total = cand.size
    out = []
    for _, region, area, comp in components(cand, int(0.012 * total)):
        x, y, w, h = region
        fill = area / float(w * h)
        if fill < 0.3:
            continue
        energy_ratio = float(np.median(ratio[comp]))
        coherence = _stripe_coherence(g_cur, comp)
        darkening = float(darker[comp].mean())
        score = clip01((energy_ratio - 2.0) / 6.0) * clip01(area / (0.05 * total) + 0.3)
        confidence = clip01(0.4 + 0.25 * clip01(area / (0.04 * total)) + 0.2 * clip01(coherence / 0.5)
                            + 0.15 * clip01((energy_ratio - 2.2) / 4))
        det = Detection("shadow_acne", True, score, confidence, region, list(range(len(frames))),
                        {"area_px": area, "fine_detail_ratio": energy_ratio, "stripe_coherence": coherence,
                         "darkening": darkening, "fill": fill},
                        f"{area} px of lit surface gained a fine stripe/moire pattern ({energy_ratio:.1f}x the "
                        f"golden build's fine detail, {darkening:+.0f} darker): shadow acne")
        heat = ev.heatmap(cur, np.clip(ratio, 0, 12).astype(np.float32), vmax=12, alpha=0.5)
        det.evidence["heatmap"] = ev.box(heat, region, ev.YELLOW, "new stripe pattern")
        zoom = _zoom_box(region)
        det.evidence["golden_vs_now"] = ev.side_by_side([ev.crop(gold, zoom, pad=0, scale_to=320),
                                                         ev.crop(cur, zoom, pad=0, scale_to=320)],
                                                        ["golden", "this build"])
        det.evidence["closeup"] = ev.crop(cur, zoom, pad=0, scale_to=320)
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def detect_detail_loss(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Surfaces that lost their fine detail versus the golden build (blurry / low-resolution textures).

    Typical causes: a texture stuck at a low mip or a streaming placeholder, a wrong filter,
    a downscaled asset. Contrast-normalized Laplacian energy collapses, and the surface looks
    like a blurred copy of the golden one (which tells a blur from a content change).
    """
    if not ctx.golden or not ctx.camera_static:
        return []
    cur_raw = clean_median(frames)
    gold = clean_median(ctx.golden)
    if cur_raw.shape != gold.shape or mostly_black(cur_raw):
        return []
    g_raw, g_gold = _gray(cur_raw), _gray(gold)
    d_cur, d_gold = _rel_detail(g_raw, "lap", 15), _rel_detail(g_gold, "lap", 15)
    ratio = (d_cur + 0.02) / (d_gold + 0.02)
    lit = cv.blur(g_raw, (9, 9)) >= 10
    cand = ((d_gold > 0.12) & (ratio < 0.35) & lit).astype(np.uint8) * 255
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, kernel(5))
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel(15))
    total = cand.size
    if not np.count_nonzero(cand):
        return []
    cur = tone_normalize(cur_raw, gold)
    s_direct = ssim_map(cur, gold)
    s_blurred = ssim_map(cur, cv.GaussianBlur(gold, (0, 0), 2.5))
    out = []
    for _, region, area, comp in components(cand, int(0.008 * total)):
        x, y, w, h = region
        fill = area / float(w * h)
        loss = float(np.median(ratio[comp]))
        like_blur = float(s_blurred[comp].mean())
        like_gold = float(s_direct[comp].mean())
        if fill < 0.25 or like_blur < 0.55 or like_blur < like_gold + 0.05:
            continue  # not a blurred copy of the golden surface: a content change, not lost detail
        score = clip01((1 - loss) * 1.2) * clip01(area / (0.03 * total) + 0.3)
        confidence = clip01(0.4 + 0.25 * clip01(area / (0.03 * total)) + 0.2 * clip01((0.35 - loss) / 0.25)
                            + 0.15 * clip01((like_blur - like_gold) / 0.2))
        det = Detection("detail_loss", True, score, confidence, region, list(range(len(frames))),
                        {"area_px": area, "detail_ratio": loss, "ssim_to_blurred_golden": like_blur,
                         "ssim_to_golden": like_gold, "fill": fill},
                        f"{area} px of textured surface kept only {loss:.0%} of its detail and looks like a "
                        "blurred copy of the golden build (low-resolution or stuck-mip texture)")
        heat = ev.heatmap(cur_raw, np.clip(1 - ratio, 0, 1).astype(np.float32), vmax=1, alpha=0.5)
        det.evidence["heatmap"] = ev.box(heat, region, ev.YELLOW, "detail lost")
        zoom = _zoom_box(region)
        det.evidence["golden_vs_now"] = ev.side_by_side([ev.crop(gold, zoom, pad=0, scale_to=320),
                                                         ev.crop(cur_raw, zoom, pad=0, scale_to=320)],
                                                        ["golden", "this build"])
        det.evidence["closeup"] = ev.crop(cur_raw, zoom, pad=0, scale_to=320)
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def detect_object_missing(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """An object the golden build ends the scenario with is not there (or ended up somewhere else).

    Compares the end state (median of the last frames), so it catches physics bugs such as an
    object tunnelling through a floor, despawning, or never spawning. A region counts as
    missing when, in the golden build, it stands out from its surroundings and in this build it
    looks like its surroundings (background shows through).
    """
    if not ctx.golden or not ctx.camera_static or len(frames) < 8:
        return []
    k = min(6, len(frames) // 4)
    cur_raw = np.median(np.stack(frames[-k:]), axis=0).astype(np.uint8)
    gold = np.median(np.stack(ctx.golden[-k:]), axis=0).astype(np.uint8)
    if cur_raw.shape != gold.shape or mostly_black(cur_raw):
        return []
    cur = tone_normalize(cur_raw, gold)
    lab_c, lab_g = lab(cur), lab(gold)
    de = np.linalg.norm(lab_c - lab_g, axis=2)
    ssim = ssim_map(cur, gold)
    mask = ((de > 14) & (ssim < 0.6)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel(3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel(9))
    total = mask.size
    missing, appeared = [], []
    gold_tail = np.stack([cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in ctx.golden[-2 * k:]]).astype(np.float32)
    cur_tail = np.stack([cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames[-2 * k:]]).astype(np.float32)
    gold_motion, cur_motion = gold_tail.std(axis=0), cur_tail.std(axis=0)
    for _, region, area, comp in components(mask, int(0.002 * total)):
        if area > 0.1 * total:
            continue  # an object, not most of the screen
        if float(gold_motion[comp].mean()) > 6.0 or float(cur_motion[comp].mean()) > 6.0:
            continue  # still moving at the end (a walking character, a vibrating crate): not a rest state
        around = ring(comp, 10) & ~(mask > 0)
        if np.count_nonzero(around) < 50:
            continue
        stands_gold = float(np.linalg.norm(lab_g[comp].mean(axis=0) - lab_g[around].mean(axis=0)))
        stands_cur = float(np.linalg.norm(lab_c[comp].mean(axis=0) - lab_c[around].mean(axis=0)))
        # A missing object changes the average colour of its area; texture detail that got
        # blurred (a painted line on a low-resolution floor) only spreads it around.
        sigma = max(3.0, 0.3 * min(region[2], region[3]))
        avg_shift = float(np.linalg.norm(cv.GaussianBlur(lab_c, (0, 0), sigma)[comp].mean(axis=0)
                                         - cv.GaussianBlur(lab_g, (0, 0), sigma)[comp].mean(axis=0)))
        edges_gold = float(np.abs(cv.Laplacian(_gray(gold), cv.CV_32F))[comp].mean())
        edges_cur = float(np.abs(cv.Laplacian(_gray(cur), cv.CV_32F))[comp].mean())
        entry = {"region": region, "area": area, "stands_gold": stands_gold, "stands_cur": stands_cur,
                 "edges_gold": edges_gold, "edges_cur": edges_cur, "avg_shift": avg_shift,
                 "point": interior_point(comp)}
        if avg_shift < 0.35 * max(stands_gold, stands_cur):
            continue
        if looks_relit(relit(cur, gold, comp)):
            continue  # same surface, brighter or darker: a shadow or a light changed (shadow_missing)
        if stands_gold > 1.6 * stands_cur + 4 and stands_cur < 12:
            missing.append(entry)
        elif stands_cur > 1.6 * stands_gold + 4 and stands_gold < 12:
            appeared.append(entry)
    out = []
    for m in sorted(missing, key=lambda e: -e["area"]):
        region = m["region"]
        elsewhere = min(appeared, key=lambda a: abs(a["area"] - m["area"]), default=None)
        where = (f"; something similar appeared at {list(elsewhere['region'])}" if elsewhere and
                 0.3 < elsewhere["area"] / m["area"] < 3 else "")
        contrast = m["stands_gold"] - m["stands_cur"]
        score = clip01(m["area"] / (0.01 * total)) * clip01(contrast / 25)
        confidence = clip01(0.4 + 0.3 * clip01(contrast / 30) + 0.2 * clip01(m["area"] / (0.006 * total))
                            + (0.1 if where else 0.0))
        det = Detection("object_missing", True, score, confidence, region,
                        list(range(len(frames) - k, len(frames))),
                        {"area_px": m["area"], "stands_out_golden": m["stands_gold"],
                         "stands_out_now": m["stands_cur"], "average_shift": m["avg_shift"],
                         "appeared_elsewhere": list(elsewhere["region"]) if where else None},
                        f"an object visible here at the end of the golden run is missing ({m['area']} px; this "
                        f"build shows the background){where}")
        det.pick_golden = True  # name it from the build where it is still there
        if m["point"]:
            det.points = [(*m["point"], len(frames) - 1)]
        vis_gold = ev.box(gold, region, ev.GREEN, "golden: here")
        vis_cur = ev.box(cur_raw, region, ev.RED, "missing")
        if where:
            vis_cur = ev.box(vis_cur, elsewhere["region"], ev.YELLOW, "appeared")
        det.evidence["golden_vs_now"] = ev.side_by_side([vis_gold, vis_cur], ["golden (end)", "this build (end)"])
        det.evidence["closeup"] = ev.side_by_side([ev.crop(gold, region, pad=20), ev.crop(cur_raw, region, pad=20)],
                                                  ["golden", "this build"])
        out.append(det)
    return out
