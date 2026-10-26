"""Evidence images drawn with OpenCV: boxes, heatmaps, series plots, contact sheets."""

from __future__ import annotations

import cv2 as cv
import numpy as np

RED = (40, 40, 230)
GREEN = (70, 200, 70)
BLUE = (230, 160, 60)
YELLOW = (40, 220, 240)
WHITE = (255, 255, 255)


def label(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.45) -> None:
    x, y = org
    (tw, th), base = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv.putText(img, text, (x, y - 2), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


def box(img: np.ndarray, region, color=RED, text: str | None = None, thickness: int = 2) -> np.ndarray:
    out = img.copy()
    if region:
        x, y, w, h = (int(v) for v in region)
        cv.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if text:
            label(out, text, (x, max(14, y - 2)), color)
    return out


def heatmap(base: np.ndarray, values: np.ndarray, vmax: float | None = None, alpha: float = 0.55) -> np.ndarray:
    vmax = float(values.max()) if vmax is None else vmax
    norm = np.clip(values / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv.applyColorMap(norm, cv.COLORMAP_INFERNO)
    return cv.addWeighted(base, 1 - alpha, colored, alpha, 0)


def series_plot(values, *, width: int = 480, height: int = 140, title: str = "",
                marks: list[int] | None = None, threshold: float | None = None) -> np.ndarray:
    """Line plot of a 1-D series, drawn with OpenCV primitives."""
    img = np.full((height, width, 3), 24, np.uint8)
    vals = np.asarray(values, np.float32)
    if vals.size == 0:
        return img
    lo, hi = float(vals.min()), float(vals.max())
    if threshold is not None:
        lo, hi = min(lo, threshold), max(hi, threshold)
    span = max(hi - lo, 1e-6)
    pad_l, pad_r, pad_t, pad_b = 36, 10, 22, 18
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def pt(i, v):
        x = pad_l + (i / max(len(vals) - 1, 1)) * pw
        y = pad_t + (1 - (v - lo) / span) * ph
        return int(round(x)), int(round(y))

    cv.rectangle(img, (pad_l, pad_t), (pad_l + pw, pad_t + ph), (70, 70, 70), 1)
    for i in marks or []:
        x, _ = pt(i, lo)
        cv.line(img, (x, pad_t), (x, pad_t + ph), (60, 60, 160), 1)
    if threshold is not None:
        _, y = pt(0, threshold)
        cv.line(img, (pad_l, y), (pad_l + pw, y), YELLOW, 1, cv.LINE_AA)
    pts = np.array([pt(i, v) for i, v in enumerate(vals)], np.int32)
    cv.polylines(img, [pts], False, (90, 220, 255), 2, cv.LINE_AA)
    cv.putText(img, title, (6, 15), cv.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv.LINE_AA)
    cv.putText(img, f"{hi:.0f}", (2, pad_t + 10), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv.putText(img, f"{lo:.0f}", (2, pad_t + ph), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv.putText(img, "frame", (width - 50, height - 4), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    return img


def contact_sheet(frames: list[np.ndarray], *, cols: int = 6, thumb_w: int = 160,
                  marks: dict[int, str] | None = None) -> np.ndarray:
    if not frames:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = frames[0].shape[:2]
    th = int(round(h * thumb_w / w))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * (th + 2), cols * (thumb_w + 2), 3), 16, np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = cv.resize(frame, (thumb_w, th), interpolation=cv.INTER_AREA)
        y0, x0 = r * (th + 2), c * (thumb_w + 2)
        sheet[y0:y0 + th, x0:x0 + thumb_w] = thumb
        tag = f"{i}" + (f" {marks[i]}" if marks and i in marks else "")
        color = RED if marks and i in marks else WHITE
        label(sheet, tag, (x0 + 4, y0 + 14), color, 0.4)
    return sheet


def side_by_side(images: list[np.ndarray], titles: list[str] | None = None, height: int = 240) -> np.ndarray:
    tiles = []
    for i, img in enumerate(images):
        scale = height / img.shape[0]
        tile = cv.resize(img, (int(img.shape[1] * scale), height), interpolation=cv.INTER_AREA)
        if titles:
            label(tile, titles[i], (6, 18))
        tiles.append(tile)
        tiles.append(np.full((height, 4, 3), 40, np.uint8))
    return np.hstack(tiles[:-1])


def crop(img: np.ndarray, region, pad: int = 24, scale_to: int = 240) -> np.ndarray:
    x, y, w, h = (int(v) for v in region)
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return img
    s = scale_to / max(patch.shape[:2])
    return cv.resize(patch, None, fx=s, fy=s, interpolation=cv.INTER_NEAREST)
