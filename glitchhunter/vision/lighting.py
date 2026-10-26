"""Lighting detectors: the same geometry and textures as the golden build, lit differently.

- shading_inverted: surface relief lit from the opposite side (a normal map in the other
  engine's convention, DirectX vs OpenGL green channel, or a negated normal scale);
- shadow_missing: an area the golden build has in shadow is now lit (a caster stopped casting
  shadows, a light lost its shadow map), or an area turned dark (a light stopped reaching it).

Both compare the end state of a static shot with the golden build after tone normalization,
so a global grade or exposure change is color_shift's finding, not theirs.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .. import evidence as ev
from .core import (DetectContext, Detection, clean_median, clip01, components, kernel, lab, mostly_black, ring,
                   tone_normalize)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)


def _band(gray: np.ndarray) -> np.ndarray:
    """Relief detail: a difference of Gaussians divided by local brightness, so a dim and a
    bright surface weigh alike and a global tone change does not move it."""
    low = cv.GaussianBlur(gray, (0, 0), 6)
    return (cv.GaussianBlur(gray, (0, 0), 1.0) - low) / (low + 8.0)


def _local_ncc(a: np.ndarray, b: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed normalized cross-correlation of two zero-mean detail images, and their energies."""
    ab = cv.GaussianBlur(a * b, (0, 0), sigma)
    aa = cv.GaussianBlur(a * a, (0, 0), sigma)
    bb = cv.GaussianBlur(b * b, (0, 0), sigma)
    return ab / np.sqrt(aa * bb + 1e-9), aa, bb


def _zoom_box(region, w: int = 160, h: int = 90) -> tuple[int, int, int, int]:
    x, y, rw, rh = region
    w, h = max(w, rw + 16), max(h, rh + 16)
    return max(0, x + rw // 2 - w // 2), max(0, y + rh // 2 - h // 2), w, h


def relit(cur: np.ndarray, gold: np.ndarray, comp: np.ndarray) -> dict:
    """How a region changed between two builds, in lighting terms.

    A lighting change multiplies the brightness of a surface fairly evenly and keeps its texture
    and colour; an object that appeared or vanished changes all three. Shared with
    object_missing, so a shadow that went missing is not reported as a missing object.

    Measured over the whole changed area inside the component's box (a detector's mask may
    only hold the edges of a shadow), and the texture away from the shadow's own edge.
    """
    ys, xs = np.nonzero(comp)
    y0, y1 = max(0, ys.min() - 16), min(comp.shape[0], ys.max() + 17)
    x0, x1 = max(0, xs.min() - 16), min(comp.shape[1], xs.max() + 17)
    box = np.zeros(comp.shape, bool)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    cur, gold, comp, box = cur[y0:y1, x0:x1], gold[y0:y1, x0:x1], comp[y0:y1, x0:x1], box[y0:y1, x0:x1]
    lc, lg = lab(cur), lab(gold)
    smooth = np.log((cv.GaussianBlur(lc[..., 0], (0, 0), 2) + 3.0) / (cv.GaussianBlur(lg[..., 0], (0, 0), 2) + 3.0))
    area = comp | (box & (np.abs(smooth) > 0.14))
    inner = cv.erode(area.astype(np.uint8), kernel(7)) > 0
    if np.count_nonzero(inner) < 60:
        inner = area
    log_ratio = np.log((lc[..., 0][area] + 3.0) / (lg[..., 0][area] + 3.0))
    ncc, _, e_g = _local_ncc(_band(_gray(cur)), _band(_gray(gold)), 5)
    textured = float(np.sqrt(e_g[inner].mean())) > 0.015
    chroma = float(np.linalg.norm(lc[..., 1:][area].mean(axis=0) - lg[..., 1:][area].mean(axis=0)))
    return {"ratio": float(np.exp(np.median(log_ratio))), "ratio_spread": float(np.std(log_ratio)),
            "texture_corr": float(ncc[inner].mean()) if textured else None, "chroma_shift": chroma}


def looks_relit(m: dict) -> bool:
    """Shadowed and sunlit ground differ in colour too (sky fill is blue, the sun is warm), and soft
    shadow filtering adds its own grain, so the texture and colour tests are loose; an object
    that vanished changes them far more."""
    texture_kept = m["texture_corr"] is None or m["texture_corr"] > 0.25
    return texture_kept and m["chroma_shift"] < 22.0 and m["ratio_spread"] < 0.3 and abs(np.log(m["ratio"])) > 0.2


def surface_change(g_cur: np.ndarray, g_gold: np.ndarray) -> np.ndarray:
    """Pixels whose change a surface explains (the relief inverted, or the same texture lit
    differently) rather than new geometry in front of it. Float grayscale inputs."""
    ncc, _, _ = _local_ncc(_band(g_cur), _band(g_gold), 5)
    log_ratio = np.abs(np.log((cv.GaussianBlur(g_cur, (0, 0), 2) + 3.0) / (cv.GaussianBlur(g_gold, (0, 0), 2) + 3.0)))
    return (ncc < -0.3) | ((ncc > 0.3) & (log_ratio > 0.1))


def detect_shading_inverted(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Surface relief whose shading is the mirror image of the golden build's.

    Bumps lit from above become dents lit from below when a normal map's green channel uses
    the other engine's convention (or its scale is negated): the relief detail keeps its
    contrast but anti-correlates with the golden build's, tile after tile. A changed texture
    does not anti-correlate coherently, and a blur loses contrast instead.
    """
    if not ctx.golden or not ctx.camera_static:
        return []
    cur_raw = clean_median(frames)
    gold = clean_median(ctx.golden)
    if cur_raw.shape != gold.shape or mostly_black(cur_raw):
        return []
    # No tone normalization: the detail band is divided by local brightness and correlated per
    # window, so a grade or gamma change cannot flip its sign, and an inverted relief over a
    # large area would bend a fitted tone curve.
    ncc, e_c, e_g = _local_ncc(_band(_gray(cur_raw)), _band(_gray(gold)), 7)
    contrast = np.sqrt(e_c / (e_g + 1e-9))
    strong = (np.sqrt(e_g) > 0.02) & (contrast > 0.5) & (contrast < 2.0)
    cand = ((ncc < -0.35) & strong).astype(np.uint8) * 255
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, kernel(3))
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel(9))
    total = cand.size
    out = []
    for _, region, area, comp in components(cand, int(0.006 * total)):
        x, y, w, h = region
        fill = area / float(w * h)
        if fill < 0.3:
            continue
        corr = float(ncc[comp].mean())
        kept = float(np.median(contrast[comp]))
        score = clip01(-corr) * clip01(area / (0.03 * total) + 0.3)
        confidence = clip01(0.45 + 0.3 * clip01((-corr - 0.35) / 0.4) + 0.25 * clip01(area / (0.02 * total)))
        det = Detection("shading_inverted", True, score, confidence, region, list(range(len(frames))),
                        {"area_px": area, "relief_correlation": corr, "contrast_kept": kept, "fill": fill},
                        f"{area} px of surface relief is shaded the opposite way to the golden build (detail "
                        f"correlation {corr:+.2f}, contrast kept {kept:.0%}): bumps read as dents, an inverted "
                        "normal map (DirectX vs OpenGL green channel, or a negative normal scale)")
        heat = ev.heatmap(cur_raw, np.clip(-ncc, 0, 1).astype(np.float32), vmax=1, alpha=0.5)
        det.evidence["heatmap"] = ev.box(heat, region, ev.YELLOW, "relief inverted")
        zoom = _zoom_box(region)
        det.evidence["golden_vs_now"] = ev.side_by_side([ev.crop(gold, zoom, pad=0, scale_to=320),
                                                         ev.crop(cur_raw, zoom, pad=0, scale_to=320)],
                                                        ["golden", "this build"])
        det.evidence["closeup"] = ev.crop(cur_raw, zoom, pad=0, scale_to=320)
        out.append(det)
    return sorted(out, key=lambda d: -d.score)


def detect_shadow_missing(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """A shadow the golden build has is gone (or an area that was lit is now dark).

    The region keeps its texture and colour and only its brightness is multiplied, evenly, and
    in the golden build it is darker than its surroundings (a shadow edge) while in this build
    it is not. Typical causes: a prop's cast-shadows flag turned off, a light's shadow map
    disabled or out of range (Unity: Cast Shadows off; Unreal: Cast Shadow unchecked; Godot:
    cast_shadow = OFF).
    """
    if not ctx.golden or not ctx.camera_static:
        return []
    cur_raw = clean_median(frames)
    gold = clean_median(ctx.golden)
    if cur_raw.shape != gold.shape or mostly_black(cur_raw):
        return []
    cur = tone_normalize(cur_raw, gold)
    l_c = cv.GaussianBlur(lab(cur)[..., 0], (0, 0), 2)
    l_g = cv.GaussianBlur(lab(gold)[..., 0], (0, 0), 2)
    ratio = (l_c + 3.0) / (l_g + 3.0)
    total = ratio.size
    out = []
    for kind, mask in (("lit", (ratio > 1.3) & (l_g > 6) & (l_c < 97)), ("dark", (ratio < 1 / 1.3) & (l_c > 6))):
        cand = cv.morphologyEx(mask.astype(np.uint8) * 255, cv.MORPH_OPEN, kernel(5))
        cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel(11))
        for _, region, area, comp in components(cand, int(0.004 * total)):
            x, y, w, h = region
            fill = area / float(w * h)
            around = ring(comp, 8) & ~(cand > 0)
            if fill < 0.3 or np.count_nonzero(around) < 50:
                continue
            m = relit(cur, gold, comp)
            if not looks_relit(m):
                continue  # texture or colour changed too: an object, not the light
            # a shadow edge in the golden build that this build does not have (or the reverse)
            edge_gold = float(l_g[around].mean() - l_g[comp].mean())
            edge_cur = float(l_c[around].mean() - l_c[comp].mean())
            if kind == "lit" and not (edge_gold > 6 and edge_cur < 0.5 * edge_gold):
                continue
            if kind == "dark" and not (edge_cur > 6 and edge_gold < 0.5 * edge_cur):
                continue
            step = abs(edge_gold - edge_cur)
            score = clip01(step / 25.0) * clip01(area / (0.02 * total) + 0.3)
            confidence = clip01(0.4 + 0.25 * clip01(step / 20.0) + 0.2 * clip01(area / (0.015 * total))
                                + 0.15 * clip01((0.3 - m["ratio_spread"]) / 0.2))
            if kind == "dark":
                confidence *= 0.8  # a new shadow can be intended (a new prop); a lost one rarely is
            what = ("is lit where the golden build has it in shadow: a shadow is missing (a caster stopped "
                    "casting shadows, or the light lost its shadow map)" if kind == "lit" else
                    "is dark where the golden build has it lit: a light no longer reaches it, or a new shadow")
            det = Detection("shadow_missing", True, score, confidence, region, list(range(len(frames))),
                            {"area_px": area, "kind": kind, "brightness_ratio": m["ratio"],
                             "ratio_spread": m["ratio_spread"], "texture_corr": m["texture_corr"],
                             "chroma_shift": m["chroma_shift"], "edge_golden": edge_gold, "edge_now": edge_cur},
                            f"{area} px of surface {what}; same texture and colour, {m['ratio']:.2f}x the "
                            "golden brightness")
            vis_gold = ev.box(gold, region, ev.GREEN, "golden: shadow" if kind == "lit" else "golden: lit")
            vis_cur = ev.box(cur_raw, region, ev.RED, "no shadow" if kind == "lit" else "dark")
            det.evidence["golden_vs_now"] = ev.side_by_side([vis_gold, vis_cur], ["golden", "this build"])
            zoom = _zoom_box(region)
            det.evidence["closeup"] = ev.side_by_side([ev.crop(gold, zoom, pad=0, scale_to=300),
                                                       ev.crop(cur_raw, zoom, pad=0, scale_to=300)],
                                                      ["golden", "this build"])
            out.append(det)
    return sorted(out, key=lambda d: -d.score)
