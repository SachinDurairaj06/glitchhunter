"""Temporal detectors built on dense optical flow (OpenCV 5 DIS optical flow).

They look at how things move rather than how a single frame looks: objects that
vibrate in place, camera motion that judders, content that pops in or out between
two frames, and characters that slide without animating.
"""

from __future__ import annotations

import hashlib

import cv2 as cv
import numpy as np

from .. import evidence as ev
from .core import (DetectContext, Detection, clip01, components, flash_frames, flow_pairs, frame_luma,
                   gray_list, interior_point, iou, kernel, optical_flow, ring, tone_mapper, tone_normalize,
                   warp_back)

_CACHE: dict[tuple, object] = {}


def _clip_id(frames: list[np.ndarray]) -> str:
    h = hashlib.sha1()
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        h.update(f[::7, ::7].tobytes())
    h.update(str(len(frames)).encode())
    return h.hexdigest()


def _cached(name: str, frames: list[np.ndarray], fn):
    """Golden clips are analysed once per process (they do not change during a hunt)."""
    key = (name, _clip_id(frames))
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = fn(frames)
    return _CACHE[key]


def flow_coherence(vectors: np.ndarray) -> float:
    from .classic import flow_coherence as _fc

    return _fc(vectors)


def motion_explained(frames: list[np.ndarray], region, pairs: int = 8, pad: int = 16) -> float:
    """Share of a region's frame-to-frame change that optical flow accounts for (0 = none, 1 = all)."""
    from .classic import _motion_explained

    return _motion_explained(frames, region, pairs, pad)


# --- physics jitter -----------------------------------------------------------------------------------

def detect_physics_jitter(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Objects that move back and forth every frame while they should be at rest.

    Per pixel, the optical flow series of a vibrating object has no net displacement but a
    large, sign-alternating motion (lag-1 autocorrelation near -1). Global camera motion
    (e.g. TAA jitter) is removed first; a walking character or a conveyor moves steadily
    (autocorrelation near +1) and particles move randomly (near 0).
    """
    if not ctx.camera_static or len(frames) < 8:
        return []
    keep = ~flash_frames(frame_luma(frames))
    use = [f for f, k in zip(frames, keep) if k][:32]
    if len(use) < 8:
        return []
    gray = gray_list(use)
    flows = flow_pairs(gray)
    flows = flows - np.median(flows.reshape(len(flows), -1, 2), axis=1)[:, None, None, :]
    u, v = flows[..., 0], flows[..., 1]
    power = u * u + v * v
    rms = np.sqrt(power.mean(axis=0))
    net = np.hypot(u.mean(axis=0), v.mean(axis=0))
    num = (u[1:] * u[:-1] + v[1:] * v[:-1]).sum(axis=0)
    den = power[1:].sum(axis=0) + power[:-1].sum(axis=0)
    lag1 = 2.0 * num / (den + 1e-6)
    med = np.median(np.stack(gray), axis=0).astype(np.float32)
    grad = cv.magnitude(cv.Sobel(med, cv.CV_32F, 1, 0), cv.Sobel(med, cv.CV_32F, 0, 1))
    textured = cv.dilate((grad > 20).astype(np.uint8), kernel(7)) > 0
    cand = ((rms > 0.8) & (net < 0.35 * rms) & (lag1 < -0.3) & textured).astype(np.uint8) * 255
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, kernel(3))
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel(9))
    out = []
    for _, region, area, comp in components(cand, 50):
        su = u[:, comp].mean(axis=1)
        sv = v[:, comp].mean(axis=1)
        axis = np.array([su.std(), sv.std()])
        s = su if axis[0] >= axis[1] else sv
        flips = float(np.mean(np.sign(s[1:]) * np.sign(s[:-1]) < 0))
        amplitude = float(np.median(np.abs(s)) / 2.0)  # positions alternate +-a: frame-to-frame motion is 2a
        explained = motion_explained(use, region)
        coherence = float(np.median([flow_coherence(fl[comp]) for fl in flows]))
        if flips < 0.6 or amplitude < 0.6 or explained < 0.15 or coherence < 0.8:
            continue  # incoherent or unexplained flow is an intensity toggle (z-fighting), not motion
        score = clip01(amplitude / 2.0)
        confidence = clip01(0.35 + 0.3 * flips + 0.2 * clip01(area / 400) + 0.15 * clip01(explained))
        frames_idx = [i for i, k in enumerate(keep) if k][:len(use)]
        det = Detection("physics_jitter", True, score, confidence, region, frames_idx,
                        {"area_px": area, "amplitude_px": amplitude, "direction_flips": flips,
                         "lag1_autocorr": float(lag1[comp].mean()), "motion_explained": explained,
                         "flow_coherence": coherence,
                         "axis": "horizontal" if s is su else "vertical"},
                        f"{area} px object vibrates +-{amplitude:.1f} px, reversing direction on {flips:.0%} of "
                        "frames while it should be at rest")
        heat = ev.heatmap(use[len(use) // 2], np.clip(rms, 0, 4).astype(np.float32), vmax=4, alpha=0.5)
        det.evidence["heatmap"] = ev.box(heat, region, ev.YELLOW, "vibrating")
        det.evidence["series"] = ev.series_plot(np.cumsum(s), title="object position along its motion axis (px)")
        det.evidence["closeup"] = ev.crop(use[len(use) // 2], region, pad=16)
        out.append(det)
    return sorted(out, key=lambda d: -(d.score * (0.5 + d.confidence)))


# --- stutter -------------------------------------------------------------------------------------------

def _motion_series(frames: list[np.ndarray], camera_static: bool) -> np.ndarray:
    gray = gray_list(frames, 0.5)
    flows = flow_pairs(gray, cv.DISOPTICAL_FLOW_PRESET_FAST)
    mag = np.linalg.norm(flows, axis=3) * 2.0  # back to full-resolution pixels
    if camera_static:
        out = []
        for m in mag:  # moving things only: mean speed of pixels that move
            moving = m > 0.6
            out.append(float(m[moving].mean()) if np.count_nonzero(moving) > 150 else 0.0)
        return np.array(out, np.float32)
    return np.median(mag.reshape(len(mag), -1), axis=1).astype(np.float32)  # camera motion


def detect_stutter(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Uneven motion: single frames where movement stalls or jumps while it should be smooth.

    Per frame pair the motion is measured with optical flow (camera motion for moving cameras,
    speed of the moving parts for static ones) and compared with its local median. A frame that
    moves < 40% or > 190% of its neighbours is a hitch or a jump; a judder pattern repeats.
    Long stops are a freeze (frozen_frame), not stutter.
    """
    if not ctx.motion_expected or len(frames) < 12:
        return []
    m = _motion_series(frames, ctx.camera_static)
    skip = 2  # the first frames include scenario setup (mode switches, spawns)
    s = m[skip:]
    level = float(np.median(s))
    if level < 0.6:
        return []  # too little motion to judge smoothness
    loc = np.array([np.median(s[max(0, i - 2):i + 3]) for i in range(len(s))])
    ratio = s / np.maximum(loc, 0.3 * level)
    low, high = ratio < 0.4, ratio > 1.9
    irregular = low | high
    # ignore long stops (that is a freeze) and runs at the very end
    runs, start = [], None
    for i, flag in enumerate(list(low) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    long_stop = {i for st, ln in runs if ln > 2 for i in range(st, st + ln)}
    events = [i for i in np.flatnonzero(irregular) if i not in long_stop]
    frac = len(events) / len(s)
    if len(events) < (4 if ctx.lossy else 3) or (ctx.lossy and frac < 0.15):
        return []  # compression makes optical flow noisier: a recording needs more evidence
    gaps = np.diff(events)
    common = np.bincount(gaps).argmax() if len(gaps) else 0
    periodic = bool(len(gaps) >= 2 and np.mean(gaps == common) >= 0.6)
    score = clip01(frac * 2.5)
    confidence = clip01(0.55 + 0.25 * clip01(len(events) / 6) + (0.15 if periodic else 0.0))
    marks = [int(i) + skip for i in events]
    det = Detection("stutter", True, score, confidence, None, marks,
                    {"irregular_frames": len(events), "of": len(s), "typical_motion_px": level,
                     "hitches": int(np.count_nonzero(low[events])), "jumps": int(np.count_nonzero(high[events])),
                     "periodic": periodic, "period_frames": int(common) if periodic else None,
                     "motion": [round(float(x), 2) for x in m]},
                    f"motion is uneven on {len(events)} of {len(s)} frames ({int(np.count_nonzero(low[events]))} "
                    f"stalls, {int(np.count_nonzero(high[events]))} jumps"
                    + (f", repeating every {common} frames" if periodic else "") + ")")
    det.evidence["motion_plot"] = ev.series_plot(m, title="motion per frame (px, optical flow)", marks=marks)
    k = marks[0]
    det.evidence["frames"] = ev.side_by_side([frames[k - 1], frames[k], frames[min(k + 1, len(frames) - 1)]],
                                             [f"frame {k - 1}", f"frame {k} (uneven)", f"frame {k + 1}"])
    return [det]


# --- pop-in -------------------------------------------------------------------------------------------

def _moves_on_its_own(gray: list[np.ndarray], t: int, region, flows: dict) -> bool:
    """Whether the content around a candidate pop moves independently of the scene just before or
    after it. An LOD or asset switch happens to something that moves with the scene (or not at all);
    a falling crate emerging from a chute or landing on a shelf moves on its own."""
    x, y, w, h = region
    hgt, wid = gray[0].shape[:2]
    near = np.zeros((hgt, wid), bool)
    near[max(0, y - 12):y + h + 12, max(0, x - 12):x + w + 12] = True
    around = ring(near, 10)
    if not np.count_nonzero(around):
        return False
    worst = 0.0
    for a, b in ((t - 2, t - 1), (t, t + 1), (t + 1, t + 2)):
        if a < 0 or b >= len(gray):
            continue
        if (a, b) not in flows:
            flows[(a, b)] = optical_flow(gray[a], gray[b])
        fl = flows[(a, b)]
        rel = np.percentile(fl[near], 90, axis=0) - np.median(fl[around], axis=0)
        worst = max(worst, float(np.linalg.norm(rel)))
    return worst > 6.0


def _residuals(frames: list[np.ndarray]) -> list[np.ndarray | None]:
    """Per frame, what optical flow from the previous frame cannot explain (None around flashes)."""
    flash = flash_frames(frame_luma(frames))
    gray = gray_list(frames)
    residuals: list[np.ndarray | None] = [None]
    h, w = gray[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for t in range(1, len(gray)):
        if flash[t] or flash[t - 1]:
            residuals.append(None)
            continue
        flow = optical_flow(gray[t], gray[t - 1])
        warped = cv.remap(gray[t - 1], gx + flow[..., 0], gy + flow[..., 1], cv.INTER_LINEAR,
                          borderMode=cv.BORDER_REPLICATE)
        r = np.abs(gray[t].astype(np.float32) - warped.astype(np.float32))
        residuals.append(cv.GaussianBlur(r, (3, 3), 0))
    return residuals


def _golden_residuals(golden: list[np.ndarray]) -> list[np.ndarray | None]:
    return [None if r is None else np.clip(r + 0.5, 0, 255).astype(np.uint8) for r in _residuals(golden)]


def _in_golden(e: dict, gold_res: list) -> bool:
    """The golden run has a comparable discontinuity at the same place within two frames (a fast
    object both builds draw: thresholds may catch it in one build and not in the other)."""
    x, y, w, h = e["region"]
    peaks = [float(gold_res[k][y:y + h, x:x + w].mean()) for k in range(e["t"] - 2, e["t"] + 3)
             if 0 <= k < len(gold_res) and gold_res[k] is not None]
    return bool(peaks) and max(peaks) >= 0.5 * e["peak"]


def _pop_events(frames: list[np.ndarray]) -> list[dict]:
    gray = gray_list(frames)
    residuals = _residuals(frames)
    events = []
    flows: dict[tuple[int, int], np.ndarray] = {}
    for t in range(2, len(residuals)):
        r = residuals[t]
        if r is None:
            continue
        mask = (r > 30).astype(np.uint8) * 255
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel(5))  # occlusion edges are thin; pops are areas
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel(5))
        for _, region, area, comp in components(mask, 120):
            peak = float(r[comp].mean())
            neighbours = [float(residuals[k][comp].mean()) for k in (t - 1, t + 1)
                          if 0 < k < len(residuals) and residuals[k] is not None]
            if not neighbours or max(neighbours) > 0.4 * peak:
                continue  # sustained change (motion, an animation): not a one-frame discontinuity
            if _moves_on_its_own(gray, t, region, flows):
                continue  # a moving object (falling, landing, coming out from behind something), not a pop
            events.append({"t": t, "region": region, "area": area, "peak": peak,
                           "isolation": 1.0 - max(neighbours) / peak})
    return events


def _golden_diffs(frames: list[np.ndarray], golden: list[np.ndarray]) -> list[np.ndarray]:
    """Per frame, how this build differs from the golden run (blurred gray, after tone matching)."""
    mid = len(frames) // 2
    mapper = tone_mapper(frames[mid], golden[mid])  # a grade or gamma change is not a difference here
    g_cur = gray_list([mapper(f) for f in frames] if mapper else frames)
    g_gold = gray_list(golden)
    return [cv.GaussianBlur(cv.absdiff(a, b).astype(np.float32), (5, 5), 0) for a, b in zip(g_cur, g_gold)]


def _golden_jumps(diff: list[np.ndarray]) -> list[dict]:
    """One-frame jumps in the difference to the golden build (same scripted run, frame by frame).

    Optical flow can explain a pop away by stretching the previous frame (a tower that suddenly
    grows into an empty sky). The difference to the golden run does not care: it stays smooth while
    both builds move alike and jumps on the frame where one of them switches.
    """
    jumps = {t: np.abs(diff[t] - diff[t - 1]) for t in range(1, len(diff))}
    events = []
    for t in range(2, len(diff) - 1):
        mask = (jumps[t] > 30).astype(np.uint8) * 255
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel(5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel(5))
        for _, region, area, comp in components(mask, 120):
            peak = float(jumps[t][comp].mean())
            neighbours = [float(jumps[k][comp].mean()) for k in (t - 1, t + 1)]
            if max(neighbours) > 0.4 * peak:
                continue
            # The same test over the box around it: an edge sweeping across (a shadow that one build
            # lacks, under a moving camera) changes different pixels every frame, a switch does not.
            x, y, w, h = region
            box = (slice(max(0, y - 8), y + h + 8), slice(max(0, x - 8), x + w + 8))
            if max(float(jumps[k][box].mean()) for k in (t - 1, t + 1)) > 0.5 * float(jumps[t][box].mean()):
                continue
            if min(float(diff[t - 1][comp].mean()), float(diff[t][comp].mean())) > 10.0:
                continue  # it differs from golden before and after: other content flickering, not a switch
            events.append({"t": t, "region": region, "area": area, "peak": peak,
                           "isolation": 1.0 - max(neighbours) / peak, "vs_golden": True})
    return events


def _same_as_golden(e: dict, frames: list[np.ndarray], golden: list[np.ndarray], toned: dict) -> bool:
    """Whether the event's region looks the same in the golden run on both frames of the jump."""
    x, y, w, h = e["region"]
    for k in (e["t"] - 1, e["t"]):
        if k not in toned:
            toned[k] = tone_normalize(frames[k], golden[k])
        a = toned[k][y:y + h, x:x + w].astype(np.int16)
        b = golden[k][y:y + h, x:x + w].astype(np.int16)
        if float(np.abs(a - b).mean()) > 6.0:
            return False
    return True


def _near(a, b) -> bool:
    """Same event in two runs: overlapping boxes, or centres closer than the larger box size."""
    if iou(a, b) > 0.1:
        return True
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < max(a[2], a[3], b[2], b[3])


def detect_pop_in(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """Content that changes abruptly between two frames in a way motion cannot explain.

    The previous frame is warped onto the current one with dense optical flow; what remains
    is what appeared, vanished or switched (an LOD level or texture mip switching in view,
    an object teleporting). Only one-frame discontinuities count, so steady motion and
    animation do not. Pops the golden build also has (an intended spawn) are dropped.
    """
    if len(frames) < 6:
        return []
    events = _pop_events(frames)
    if ctx.golden and len(ctx.golden) == len(frames):
        gold_events = _cached("pop_events", ctx.golden, _pop_events)
        gold_res = _cached("residuals", ctx.golden, _golden_residuals)
        gray = gray_list(frames)
        flows: dict = {}
        if not ctx.camera_static:
            # Under a moving camera (where LOD switches happen) flow residuals also fire at every
            # occlusion edge (a near object sliding over a far floor). The difference to the golden
            # run, which flew the same path, stays smooth there and jumps only where one build
            # switches: a flow event must jump against golden too, and a jump of its own counts.
            diff = _golden_diffs(frames, ctx.golden)

            def golden_jump(e) -> float:
                """The step in the golden difference at the event (0 unless it is a one-frame step:
                an edge sweeping across the box changes it on frame after frame)."""
                x, y, w, h = e["region"]
                t = e["t"]

                def step(k: int) -> float:
                    if k < 1 or k >= len(diff):
                        return 0.0
                    return float(np.abs(diff[k] - diff[k - 1])[y:y + h, x:x + w].mean())

                here = step(t)
                return here if max(step(t - 1), step(t + 1)) <= 0.5 * here else 0.0

            events = [e for e in events if golden_jump(e) >= max(12.0, 0.3 * e["peak"])]
            for j in _golden_jumps(diff):
                if any(abs(j["t"] - o["t"]) <= 1 and _near(j["region"], o["region"]) for o in events):
                    continue
                if not _moves_on_its_own(gray, j["t"], j["region"], flows):
                    events.append(j)
        # within a few frames: the approved build's timing can differ a little (random spawn heights)
        events = [e for e in events if not any(abs(e["t"] - g["t"]) <= 3 and _near(e["region"], g["region"])
                                               for g in gold_events)]
        # and anything the golden run also jumps at, or shows identically
        toned: dict[int, np.ndarray] = {}
        events = [e for e in events if e.get("vs_golden") or not (
            _in_golden(e, gold_res) or _same_as_golden(e, frames, ctx.golden, toned))]
    if not events:
        return []
    groups: list[list[dict]] = []
    for e in sorted(events, key=lambda e: -e["area"]):
        for g in groups:
            if iou(g[0]["region"], e["region"]) > 0.15 and abs(g[0]["t"] - e["t"]) <= 2:
                g.append(e)
                break
        else:
            groups.append([e])
    out = []
    for g in groups[:4]:
        if len({e["t"] for e in g}) >= 3:
            continue  # the same place changing again and again is judder or flicker, not a one-off pop
        e = max(g, key=lambda e: e["area"])
        t, region = e["t"], e["region"]
        score = clip01(e["area"] / 900) * clip01(e["peak"] / 60)
        confidence = clip01(0.45 + 0.35 * e["isolation"] + 0.2 * clip01(e["area"] / 600))
        det = Detection("pop_in", True, score, confidence, region, [t - 1, t],
                        {"frame": t, "area_px": e["area"], "residual": e["peak"], "isolation": e["isolation"],
                         "events": len(g)},
                        f"{e['area']} px of content changes abruptly between frames {t - 1} and {t} "
                        "(motion does not explain it: level-of-detail or asset pop)")
        det.evidence["before_after"] = ev.side_by_side(
            [ev.box(frames[t - 1], region, ev.GREEN), ev.box(frames[t], region, ev.RED)],
            [f"frame {t - 1}", f"frame {t} (pop)"])
        det.evidence["closeup"] = ev.side_by_side([ev.crop(frames[t - 1], region, pad=20),
                                                   ev.crop(frames[t], region, pad=20)], ["before", "after"])
        out.append(det)
    return out


# --- frozen animation ---------------------------------------------------------------------------------

def _tracks(frames: list[np.ndarray]) -> list[dict]:
    """Moving blobs linked over time, with how much they articulate.

    Articulation is the 90th percentile of the optical flow inside the blob after removing the
    blob's own translation (eroded interior only: flow at silhouette edges is smeared by the
    background). Limbs swinging push it up; a rigid slide leaves only flow noise.
    """
    gray = gray_list(frames)
    flows = flow_pairs(gray)
    tracks: list[dict] = []
    for t, fl in enumerate(flows):
        mag = np.linalg.norm(fl, axis=2)
        moving = (mag > 0.7).astype(np.uint8) * 255
        moving = cv.morphologyEx(moving, cv.MORPH_CLOSE, kernel(9))
        moving = cv.morphologyEx(moving, cv.MORPH_OPEN, kernel(3))
        for _, region, area, comp in components(moving, 250):
            inner = cv.erode(comp.astype(np.uint8), kernel(3)) > 0
            if np.count_nonzero(inner) < 50:
                continue
            vecs = fl[inner]
            med = np.median(vecs, axis=0)
            art = float(np.percentile(np.linalg.norm(vecs - med, axis=1), 90))
            trans = float(np.linalg.norm(med))
            # where to ask the game what this is: inside the upper part of the blob (a walking
            # figure's shadow moves with it, low down on the floor)
            upper = inner.copy()
            upper[region[1] + int(0.6 * region[3]):] = False
            blob = {"t": t + 1, "region": region, "art": art, "trans": trans, "area": area,
                    "pt": interior_point(upper) or interior_point(inner)}
            best = max(tracks, key=lambda tr: iou(tr["blobs"][-1]["region"], region), default=None)
            if best and best["blobs"][-1]["t"] >= t and iou(best["blobs"][-1]["region"], region) > 0.2:
                best["blobs"].append(blob)
            else:
                tracks.append({"blobs": [blob]})
    out = []
    for tr in tracks:
        b = tr["blobs"]
        if len(b) < 0.4 * len(flows):
            continue
        xs = [r["region"] for r in b]
        x0 = min(r[0] for r in xs)
        y0 = min(r[1] for r in xs)
        x1 = max(r[0] + r[2] for r in xs)
        y1 = max(r[1] + r[3] for r in xs)
        mid = b[len(b) // 2]
        out.append({"union": (x0, y0, x1 - x0, y1 - y0), "mid": mid["region"], "mid_t": mid["t"],
                    "articulation": float(np.median([r["art"] for r in b])),
                    "translation": float(np.median([r["trans"] for r in b])),
                    "frames": [r["t"] for r in b], "area": float(np.median([r["area"] for r in b])),
                    "points": [(*r["pt"], r["t"]) for r in (b[len(b) // 4], mid, b[3 * len(b) // 4]) if r["pt"]]})
    return out


def detect_frozen_animation(frames: list[np.ndarray], ctx: DetectContext) -> list[Detection]:
    """A character that moves across the screen but no longer animates (sliding T-pose).

    Optical flow inside each moving blob is split into the blob's rigid translation and the
    rest; the rest is articulation (legs and arms swinging). A blob that articulates in the
    golden build but only translates in this build has lost its animation (a clip that no
    longer binds to the skeleton, a stopped mixer, a missing state machine transition).
    """
    if not ctx.golden or not ctx.camera_static or len(frames) < 10 or len(ctx.golden) != len(frames):
        return []
    gold_tracks = _cached("tracks", ctx.golden, _tracks)
    animated = [g for g in gold_tracks if g["articulation"] > 0.6 and g["translation"] > 0.3]
    if not animated:
        return []
    cur_tracks = _tracks(frames)
    out = []
    for g in animated:
        match = max(cur_tracks, key=lambda c: iou(c["union"], g["union"]), default=None)
        if match is None or iou(match["union"], g["union"]) < 0.25:
            continue
        ratio = match["articulation"] / g["articulation"]
        if ratio > 0.72 or match["translation"] < 0.3:
            continue
        score = clip01((1 - ratio) * 1.5)
        confidence = clip01(0.55 + 0.25 * clip01((0.72 - ratio) / 0.3) + 0.2 * clip01(g["articulation"] / 1.2))
        t = match["mid_t"]
        det = Detection("frozen_animation", True, score, confidence, match["mid"], match["frames"],
                        {"articulation_now": match["articulation"], "articulation_golden": g["articulation"],
                         "ratio": ratio, "translation_px": match["translation"], "track_bbox": list(match["union"])},
                        f"moving object articulates {ratio:.0%} as much as in the golden build "
                        f"({match['articulation']:.2f} vs {g['articulation']:.2f} px/frame of non-rigid motion): "
                        "it slides without animating")
        det.points = match["points"]
        gt = min(t, len(ctx.golden) - 1)
        det.evidence["golden_vs_now"] = ev.side_by_side(
            [ev.crop(ctx.golden[gt], g["union"], pad=20, scale_to=300),
             ev.crop(frames[t], match["union"], pad=20, scale_to=300)], ["golden (animated)", "this build"])
        strip = [ev.crop(frames[k], match["union"], pad=10, scale_to=160)
                 for k in range(0, len(frames), max(1, len(frames) // 6))][:6]
        det.evidence["strip"] = ev.side_by_side(strip, height=160)
        out.append(det)
    return out
