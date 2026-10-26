"""OpenCV 5 glitch detectors and the registry the agent, MCP server and API use.

Five families:
- classic: single-image and simple temporal checks (missing texture, z-fighting, frozen and
  black frames, flashes, exploded geometry, HUD text overflow)
- motion: dense optical flow (DIS) over the clip (physics jitter, stutter, pop-in, frozen animation)
- compare: against the approved golden build (color shift, shadow acne, detail loss, missing object)
- lighting: same surfaces lit differently than in the golden build (inverted normal maps, missing shadows)
- pixels: invalid pixels and layout (NaN blocks, UI cut off by the screen edge)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .classic import (detect_black_frame, detect_flash, detect_frozen_frame, detect_geometry_explosion,
                      detect_hud_text_overflow, detect_missing_texture, detect_visual_diff, detect_zfight_flicker)
from .compare import detect_color_shift, detect_detail_loss, detect_object_missing, detect_shadow_acne
from .core import (FPS, DetectContext, Detection, DetectorSpec, expand, flash_frames, frame_luma, iou, match_to,
                   median_frame, ssim_map, to_gray_stack)
from .lighting import detect_shading_inverted, detect_shadow_missing
from .motion import detect_frozen_animation, detect_physics_jitter, detect_pop_in, detect_stutter
from .pixels import detect_nan_pixels, detect_ui_offscreen

DETECTORS: dict[str, DetectorSpec] = {
    s.name: s
    for s in (
        DetectorSpec("missing_texture", detect_missing_texture,
                     "Magenta/black checker placeholder where a texture failed to load.", "major", True),
        DetectorSpec("zfight_flicker", detect_zfight_flicker,
                     "Surfaces toggling between frames under a static camera (z-fighting); motion-compensated, "
                     "so vibrating objects do not count.", "minor", True, golden_suppress=True),
        DetectorSpec("frozen_frame", detect_frozen_frame,
                     "Picture stops changing while the scenario expects motion (hang / dead game loop).",
                     "critical", False),
        DetectorSpec("black_frame", detect_black_frame,
                     "Most frames are near-black: the scene is not being drawn.", "critical", False),
        DetectorSpec("flash", detect_flash,
                     "Full-screen brightness spikes (photosensitivity / WCAG 2.3.1).", "major", False),
        DetectorSpec("geometry_explosion", detect_geometry_explosion,
                     "Large spiky shapes that differ from the golden frame (exploded or stretched mesh).",
                     "major", True, needs_golden=True),
        DetectorSpec("hud_text_overflow", detect_hud_text_overflow,
                     "UI text running past its panel (OpenCV DNN text detection).", "minor", False),
        DetectorSpec("color_shift", detect_color_shift,
                     "Whole picture changed tone or color vs the golden build, same content (gamma / color "
                     "space / tone mapping); fits the tone curve.", "major", False, needs_golden=True,
                     family="compare"),
        DetectorSpec("shadow_acne", detect_shadow_acne,
                     "Surfaces gained a fine stripe/moire pattern vs the golden build (shadow acne, "
                     "self-shadowing).", "minor", True, needs_golden=True, family="compare"),
        DetectorSpec("detail_loss", detect_detail_loss,
                     "Surfaces lost fine detail vs the golden build (blurry, low-resolution or stuck mip "
                     "textures).", "minor", True, needs_golden=True, family="compare"),
        DetectorSpec("object_missing", detect_object_missing,
                     "An object present at the end of the golden run is missing (physics tunnelling, "
                     "despawn).", "major", True, needs_golden=True, family="compare"),
        DetectorSpec("shading_inverted", detect_shading_inverted,
                     "Surface relief shaded the opposite way to the golden build: bumps read as dents (normal "
                     "map in the other engine's green-channel convention, negative normal scale).", "minor", True,
                     needs_golden=True, family="lighting"),
        DetectorSpec("shadow_missing", detect_shadow_missing,
                     "A shadow the golden build has is gone (cast shadows turned off, shadow map lost), or a lit "
                     "area turned dark; same texture and colour, only the light changed.", "minor", True,
                     needs_golden=True, family="lighting"),
        DetectorSpec("physics_jitter", detect_physics_jitter,
                     "An object vibrates in place (optical flow reverses direction every frame).", "minor",
                     True, golden_suppress=True, family="motion"),
        DetectorSpec("stutter", detect_stutter,
                     "Uneven motion: frames where movement stalls or jumps (judder, frame pacing).", "major",
                     False, golden_suppress=True, family="motion"),
        DetectorSpec("pop_in", detect_pop_in,
                     "Content changes abruptly between two frames in a way motion cannot explain (LOD or "
                     "asset pop).", "minor", True, family="motion"),
        DetectorSpec("frozen_animation", detect_frozen_animation,
                     "A character moves but no longer animates (sliding T-pose).", "major", True,
                     needs_golden=True, family="motion"),
        DetectorSpec("nan_pixels", detect_nan_pixels,
                     "Blocks of pure-black pixels in lit areas: NaN/Inf from a shader.", "major", True,
                     family="pixels"),
        DetectorSpec("ui_offscreen", detect_ui_offscreen,
                     "UI panel cut off by the screen edge with truncated text (check other screen sizes).",
                     "minor", False, family="pixels"),
        DetectorSpec("visual_diff", detect_visual_diff,
                     "Informational: structural change versus the golden frame (SSIM).", "info", True,
                     needs_golden=True),
    )
}
BUG_DETECTORS = [name for name, spec in DETECTORS.items() if spec.severity != "info"]
MIN_CONFIDENCE = 0.1  # below this a detection is noise, not even worth a zoom

_GOLDEN_DETS: dict[tuple, list[Detection]] = {}


def _golden_detections(spec: DetectorSpec, ctx: DetectContext) -> list[Detection]:
    from .motion import _clip_id

    key = (spec.name, _clip_id(ctx.golden), ctx.roi)
    if key not in _GOLDEN_DETS:
        if len(_GOLDEN_DETS) > 256:
            _GOLDEN_DETS.clear()
        g_ctx = DetectContext(ctx.motion_expected, None, None, ctx.camera_static, ctx.fps, ctx.key, ctx.lossy)
        _GOLDEN_DETS[key] = spec.fn(ctx.golden, g_ctx)
    return _GOLDEN_DETS[key]


def _also_in_golden(det: Detection, golden: list[Detection]) -> bool:
    for g in golden:
        if det.region is None or g.region is None:
            return True
        if iou(det.region, g.region) > 0.2:
            return True
    return False


def _inside(region, roi) -> bool:
    """A detection belongs to the ROI when they overlap by 30% of the smaller of the two."""
    if region is None:
        return True
    ix = min(region[0] + region[2], roi[0] + roi[2]) - max(region[0], roi[0])
    iy = min(region[1] + region[3], roi[1] + roi[3]) - max(region[1], roi[1])
    if ix <= 0 or iy <= 0:
        return False
    return ix * iy >= 0.3 * min(region[2] * region[3], roi[2] * roi[3])


_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor | None:
    """Detector threads (GH_DETECT_THREADS, default up to 6; 1 runs them one after another)."""
    global _POOL
    size = int(os.environ.get("GH_DETECT_THREADS", min(6, os.cpu_count() or 1)))
    if size <= 1:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=size, thread_name_prefix="detect")
    return _POOL


def run_detectors(frames, names: list[str] | None, ctx: DetectContext) -> list[Detection]:
    """Run detectors, optionally keeping only what overlaps ctx.roi.

    The detectors always see the whole frame, even with an ROI: what they measure is relative to
    the picture (an object that is gone covers a small part of the frame but most of a crop; tone
    and camera motion are estimated from everything), so a re-check of one region during replay
    or bisect must see what the survey saw. Several detectors run in parallel threads (OpenCV
    releases the GIL); results come back in the order asked for.
    """
    names = [n for n in (names or BUG_DETECTORS) if ctx.golden or not DETECTORS[n].needs_golden]
    full_ctx = DetectContext(ctx.motion_expected, ctx.golden, None, ctx.camera_static, ctx.fps, ctx.key, ctx.lossy)

    def one(name: str) -> list[Detection]:
        spec = DETECTORS[name]
        found = spec.fn(frames, full_ctx)
        if found and spec.golden_suppress and ctx.golden:
            golden = _golden_detections(spec, full_ctx)
            found = [d for d in found if not _also_in_golden(d, golden)]
        return [d for d in found if d.confidence >= MIN_CONFIDENCE and (not ctx.roi or _inside(d.region, ctx.roi))]

    nested = threading.current_thread().name.startswith("detect")  # never wait on the pool from inside it
    pool = _pool() if len(names) > 1 and not nested else None
    batches = list(pool.map(one, names)) if pool else [one(n) for n in names]
    return [d for batch in batches for d in batch]


__all__ = ["BUG_DETECTORS", "DETECTORS", "FPS", "DetectContext", "Detection", "DetectorSpec", "expand",
           "flash_frames", "frame_luma", "iou", "match_to", "median_frame", "run_detectors", "ssim_map",
           "to_gray_stack", "detect_black_frame", "detect_color_shift", "detect_detail_loss", "detect_flash",
           "detect_frozen_animation", "detect_frozen_frame", "detect_geometry_explosion", "detect_hud_text_overflow",
           "detect_missing_texture", "detect_nan_pixels", "detect_object_missing", "detect_physics_jitter",
           "detect_pop_in", "detect_shading_inverted", "detect_shadow_acne", "detect_shadow_missing",
           "detect_stutter", "detect_ui_offscreen", "detect_visual_diff", "detect_zfight_flicker"]
