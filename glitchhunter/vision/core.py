"""Shared types and OpenCV helpers for the detectors.

Every detector takes the captured frames (BGR uint8) plus a DetectContext and
returns Detection objects. A detection carries a score (strength of the visual
evidence), a confidence (how much to trust it: small or ambiguous evidence is
low confidence, which is what makes the agent zoom in), the screen region, the
frames involved, raw metrics and evidence images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import threading

import cv2 as cv
import numpy as np

FPS = 30.0


@dataclass
class Detection:
    detector: str
    present: bool
    score: float
    confidence: float
    region: tuple[int, int, int, int] | None = None
    frames: list[int] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    note: str = ""
    evidence: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    # Where to ask the game what the glitched thing is: (x, y, frame) on the thing itself (a thin
    # walking figure fills little of its box), and whether it only exists in the golden build
    # (an object that went missing is named from the build that still has it).
    points: list[tuple[int, int, int]] = field(default_factory=list, repr=False)
    pick_golden: bool = False

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "present": self.present,
            "score": round(float(self.score), 3),
            "confidence": round(float(self.confidence), 3),
            "region": [int(v) for v in self.region] if self.region else None,
            "frames": [int(f) for f in self.frames[:60]],
            "metrics": jsonable(self.metrics),
            "note": self.note,
        }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class DetectContext:
    motion_expected: bool = False
    golden: list[np.ndarray] | None = None
    roi: tuple[int, int, int, int] | None = None
    camera_static: bool = True  # nothing moves the camera: motion on screen is the scene's own
    fps: float = FPS
    key: tuple | None = None  # identifies the golden clip, so golden-side analysis can be cached
    lossy: bool = False  # frames come from a compressed video: exact pixel values are not exact


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    fn: Callable[[list[np.ndarray], DetectContext], list[Detection]]
    description: str
    severity: str  # critical | major | minor | info
    localized: bool  # evidence is a world-space region the camera can move closer to
    needs_golden: bool = False
    golden_suppress: bool = False  # also run on the golden clip; drop what the approved build shows too
    family: str = "classic"  # classic | motion | compare | pixels (for docs and the planner prompt)


# --- frame helpers --------------------------------------------------------------------------------

def to_gray_stack(frames: list[np.ndarray]) -> np.ndarray:
    return np.stack([cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)


def frame_luma(frames: list[np.ndarray]) -> np.ndarray:
    return np.array([cv.mean(cv.cvtColor(f, cv.COLOR_BGR2GRAY))[0] for f in frames], np.float32)


def flash_frames(lum: np.ndarray) -> np.ndarray:
    """Boolean mask of frames much brighter than the clip's median (global flashes)."""
    base = float(np.median(lum))
    mad = float(np.median(np.abs(lum - base)))
    return (lum - base) > max(20.0, 6.0 * mad, 0.3 * base)


def median_frame(frames: list[np.ndarray], exclude: np.ndarray | None = None) -> np.ndarray:
    keep = [f for i, f in enumerate(frames) if exclude is None or not exclude[i]] or frames
    if len(keep) > 15:
        keep = keep[:: max(1, len(keep) // 15)]
    return np.median(np.stack(keep), axis=0).astype(np.uint8)


def clean_median(frames: list[np.ndarray]) -> np.ndarray:
    """Median frame without flash frames."""
    return median_frame(frames, flash_frames(frame_luma(frames)))


def components(mask: np.ndarray, min_area: int):
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            yield i, (x, y, w, h), area, labels == i


def kernel(size: int) -> np.ndarray:
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))


def clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Structural similarity map (Wang et al. 2004) computed with Gaussian windows."""
    if a.ndim == 3:
        a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def match_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Histogram-match each channel of img to ref (photometric normalization).

    A global tone or color change (wrong gamma, exposure, color space) then stops
    looking like a local difference, so local detectors compare structure, not tone.
    """
    out = np.empty_like(img)
    levels = np.arange(256, dtype=np.float32)
    for c in range(img.shape[2]):
        h_img = np.bincount(img[..., c].ravel(), minlength=256).astype(np.float64)
        h_ref = np.bincount(ref[..., c].ravel(), minlength=256).astype(np.float64)
        cdf_img = np.cumsum(h_img) / h_img.sum()
        cdf_ref = np.cumsum(h_ref) / h_ref.sum()
        lut = np.interp(cdf_img, cdf_ref, levels)
        out[..., c] = cv.LUT(img[..., c], np.clip(lut + 0.5, 0, 255).astype(np.uint8))
    return out


def tone_normalize(img: np.ndarray, ref: np.ndarray, min_shift: float = 4.0) -> np.ndarray:
    """img in ref's tones when the two builds differ in tone overall, else img unchanged.

    The tone curve is applied only when the difference covers most of the picture (a grade or
    gamma change touches nearly everything; a large local change does not), and when it moves
    pixels by more than `min_shift` levels on average and clearly brings the two pictures
    closer: re-mapping an image whose tone did not change would only smear local content
    changes into the curve.
    """
    mapper = tone_mapper(img, ref, min_shift)
    return mapper(img) if mapper else img


def tone_mapper(img: np.ndarray, ref: np.ndarray, min_shift: float = 4.0):
    """The tone transform tone_normalize would apply to img, as a function (None: leave it).

    Fitted once on one frame pair, it can be applied to every frame of a clip.
    """
    def dist(a, b) -> float:
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    if _changed_share(img, ref) < 0.5:
        return None
    before = dist(img, ref)
    luts = paired_luts(img, ref)

    def curves(x: np.ndarray) -> np.ndarray:
        return cv.merge([cv.LUT(x[..., c], luts[c]) for c in range(3)])

    mapped = curves(img)
    m = affine_fit(mapped, ref)

    def curves_and_matrix(x: np.ndarray) -> np.ndarray:
        return apply_affine(curves(x), m)

    best, best_err = None, before
    # Per-channel curves undo gamma and exposure; a colour matrix on top undoes what mixes
    # channels (saturation and hue grading), which curves alone cannot.
    for fn, cand in ((curves, mapped), (curves_and_matrix, apply_affine(mapped, m))):
        err = dist(cand, ref)
        if dist(cand, img) > min_shift and err < 0.7 * before and err < best_err:
            best, best_err = fn, err
    return best


def _changed_share(img: np.ndarray, ref: np.ndarray, grid: int = 8) -> float:
    """Share of the visible tiles (not near-black in both) whose pixels differ between the builds."""
    diff = cv.absdiff(img, ref).astype(np.float32).mean(axis=2)
    lit = np.maximum(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.cvtColor(ref, cv.COLOR_BGR2GRAY)) > 12
    h, w = diff.shape
    changed = valid = 0
    for i in range(grid):
        for j in range(grid):
            ys = slice(i * h // grid, (i + 1) * h // grid)
            xs = slice(j * w // grid, (j + 1) * w // grid)
            if lit[ys, xs].mean() < 0.5:
                continue
            valid += 1
            changed += float(diff[ys, xs].mean()) > 1.5
    return changed / valid if valid else 0.0


def color_affine(img: np.ndarray, ref: np.ndarray, step: int = 4, iters: int = 3) -> np.ndarray:
    """img through the 3x4 colour transform that best maps it onto ref."""
    return apply_affine(img, affine_fit(img, ref, step, iters))


def apply_affine(img: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    if m is None:
        return img
    out = img.reshape(-1, 3).astype(np.float32) @ m[:3] + m[3]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8).reshape(img.shape)


def affine_fit(img: np.ndarray, ref: np.ndarray, step: int = 4, iters: int = 3) -> np.ndarray | None:
    """The 3x4 colour transform (4x3 matrix with the offset row) that best maps img onto ref.

    Fitted by least squares on a pixel grid; pixels whose content changed (a missing object,
    a placeholder texture) do not fit a global transform, so they are dropped between
    iterations and cannot bend it.
    """
    a = img[::step, ::step].reshape(-1, 3).astype(np.float32)
    b = ref[::step, ::step].reshape(-1, 3).astype(np.float32)
    design = np.hstack([a, np.ones((len(a), 1), np.float32)])
    keep = np.ones(len(a), bool)
    m = None
    for _ in range(iters):
        if np.count_nonzero(keep) < 100:
            break
        m, *_ = np.linalg.lstsq(design[keep], b[keep], rcond=None)
        res = np.linalg.norm(design @ m - b, axis=1)
        keep = res < max(6.0, 2.5 * float(np.median(res[keep])))
    return m


def interior_point(mask: np.ndarray) -> tuple[int, int] | None:
    """The pixel deepest inside a mask (on a thin figure, its centre may lie outside it)."""
    m = mask.astype(np.uint8)
    if not m.any():
        return None
    d = cv.distanceTransform(m, cv.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y)


def mostly_black(img: np.ndarray, level: int = 32, share: float = 0.7) -> bool:
    """Most of the picture is near-black: the scene is not drawn (black_frame's case, not a comparison)."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray < level)) / gray.size >= share


def match_paired(img: np.ndarray, ref: np.ndarray, bins: int = 16) -> np.ndarray:
    """Map img into ref's tones with a per-channel curve estimated from co-located pixels.

    For two captures of the same view (a build and the golden build), most pixels show the
    same content, so pairing their values gives the tone curve between the builds (gamma,
    grading, exposure). The curve is fitted through medians of quantile bins (each holds 1/16
    of the pixels), so a small region whose content changed (a magenta placeholder texture)
    cannot bend it; unlike histogram matching, local content changes survive.
    """
    luts = paired_luts(img, ref, bins)
    return cv.merge([cv.LUT(img[..., c], luts[c]) for c in range(img.shape[2])])


def paired_luts(img: np.ndarray, ref: np.ndarray, bins: int = 16) -> list[np.ndarray]:
    """The per-channel lookup tables of match_paired."""
    luts = []
    levels = np.arange(256, dtype=np.float32)
    for c in range(img.shape[2]):
        a = img[..., c].ravel()
        b = ref[..., c].ravel()
        order = np.argsort(a, kind="stable")
        a_sorted, b_sorted = a[order].astype(np.float32), b[order].astype(np.float32)
        xs, ys = [], []
        for chunk_a, chunk_b in zip(np.array_split(a_sorted, bins), np.array_split(b_sorted, bins)):
            if len(chunk_a):
                xs.append(float(np.median(chunk_a)))
                ys.append(float(np.median(chunk_b)))
        xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))  # tone curves are monotonic
        keep = np.concatenate([[True], np.diff(xs) > 0])  # np.interp needs increasing x
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 2:
            luts.append(np.arange(256, dtype=np.uint8))
            continue
        lo_slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1.0)
        hi_slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1.0)
        xs = np.concatenate([[0.0], xs, [255.0]])
        ys = np.concatenate([[ys[0] - lo_slope * xs[1]], ys, [ys[-1] + hi_slope * (255.0 - xs[-2])]])
        luts.append(np.clip(np.interp(levels, xs, ys) + 0.5, 0, 255).astype(np.uint8))
    return luts


def rel_detail(gray: np.ndarray, op: str, win: int) -> np.ndarray:
    """Local detail energy divided by local brightness (contrast-normalized), so a darker or
    brighter build (a tone change) does not read as more or less detail."""
    if op == "fine":  # 1-2 px structure: what a median filter removes (moire lines, dots, noise)
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        detail = np.abs(gray - cv.medianBlur(g8, 3).astype(np.float32))
    else:  # texture detail: Laplacian magnitude
        detail = np.abs(cv.Laplacian(gray, cv.CV_32F, ksize=3))
    return cv.blur(detail, (win, win)) / (cv.blur(gray, (win, win)) + 8.0)


def lab(img: np.ndarray) -> np.ndarray:
    """CIE Lab with L in 0..100 and a, b centred on 0 (float32)."""
    f = cv.cvtColor(img.astype(np.float32) / 255.0, cv.COLOR_BGR2Lab)
    return f


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab(a) - lab(b), axis=2)


def ring(mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels just outside a mask (its surroundings)."""
    m = mask.astype(np.uint8) * 255
    return (cv.dilate(m, kernel(2 * width + 1)) > 0) & ~(m > 0)


# --- optical flow -----------------------------------------------------------------------------------

_DIS: dict[tuple, cv.DISOpticalFlow] = {}


def dis(preset: int = cv.DISOPTICAL_FLOW_PRESET_MEDIUM, shape: tuple = ()) -> cv.DISOpticalFlow:
    """A DIS optical flow engine for one image size, on this thread.

    DIS adapts its patch size and finest scale to the first small image it sees and keeps them,
    so one engine shared between a 640x360 frame and a 40x40 crop answers differently depending
    on what ran before. One engine per (preset, size) keeps every answer the same, and one per
    thread lets detectors run in parallel (an engine is not thread-safe).
    """
    key = (preset, tuple(shape[:2]), threading.get_ident())
    if key not in _DIS:
        if len(_DIS) > 32:
            _DIS.clear()
        _DIS[key] = cv.DISOpticalFlow_create(preset)
    return _DIS[key]


def optical_flow(prev: np.ndarray, cur: np.ndarray, preset: int = cv.DISOPTICAL_FLOW_PRESET_MEDIUM) -> np.ndarray:
    """Dense optical flow (DIS) from prev to cur, (H, W, 2) in pixels."""
    return dis(preset, prev.shape).calc(prev, cur, None)


def flow_pairs(gray: list[np.ndarray], preset: int = cv.DISOPTICAL_FLOW_PRESET_MEDIUM) -> np.ndarray:
    """Dense optical flow (DIS) from each frame to the next: shape (T-1, H, W, 2), pixels per frame."""
    return np.stack([optical_flow(gray[i - 1], gray[i], preset) for i in range(1, len(gray))])


def warp_back(prev: np.ndarray, cur: np.ndarray, preset: int = cv.DISOPTICAL_FLOW_PRESET_MEDIUM) -> np.ndarray:
    """prev warped into cur's frame (flow from cur to prev), so |cur - warp| is what motion cannot explain."""
    flow = optical_flow(cur, prev, preset)
    h, w = cur.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv.remap(prev, gx + flow[..., 0], gy + flow[..., 1], cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)


def gray_list(frames: list[np.ndarray], scale: float = 1.0) -> list[np.ndarray]:
    out = []
    for f in frames:
        g = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        if scale != 1.0:
            g = cv.resize(g, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        out.append(g)
    return out


def expand(region, frame_shape, factor: float = 1.6, min_size: int = 48) -> tuple[int, int, int, int]:
    x, y, w, h = region
    H, W = frame_shape[:2]
    cx, cy = x + w / 2, y + h / 2
    nw, nh = max(min_size, w * factor), max(min_size, h * factor)
    x0, y0 = int(max(0, cx - nw / 2)), int(max(0, cy - nh / 2))
    x1, y1 = int(min(W, cx + nw / 2)), int(min(H, cy + nh / 2))
    return x0, y0, x1 - x0, y1 - y0


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def scale_region(region, s: float) -> tuple[int, int, int, int]:
    x, y, w, h = region
    return int(round(x * s)), int(round(y * s)), max(1, int(round(w * s))), max(1, int(round(h * s)))
