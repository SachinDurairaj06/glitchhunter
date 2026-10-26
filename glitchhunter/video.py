"""Video triage: find glitches in a gameplay recording from any engine (no game API, no repo).

QA teams record playtests (OBS, console capture, Unity Recorder, Unreal Movie Render Queue).
This mode reads such a file with OpenCV, splits it into shots, runs the detectors that need
no game hooks over sliding windows, and then investigates what it found the way the hunt
does, within what a recording allows:

- spatial zoom: re-read the region at the recording's native resolution, upscaled, and
  re-run the detector (a recording is often scanned downscaled);
- temporal zoom: re-read the time range at every frame (the scan may skip frames);
- persistence: a glitch seen in neighbouring windows or shots gains confidence;
- a person approves before anything is filed, as in a hunt.

There is no bisect or fix here: a recording has no history. With `--golden` (a recording of
the same scripted run on an approved build, frame-aligned) the golden comparisons run too.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from . import evidence as ev
from .agent.trace import Trace
from .vision import DETECTORS, DetectContext, Detection, iou, run_detectors
from .vision.core import flow_pairs, gray_list

# Detectors that work on any recording; golden comparisons are added when a golden video is given.
VIDEO_DETECTORS = ["missing_texture", "zfight_flicker", "frozen_frame", "black_frame", "flash", "physics_jitter",
                   "stutter", "pop_in", "nan_pixels", "hud_text_overflow", "ui_offscreen"]
GOLDEN_DETECTORS = ["color_shift", "shadow_acne", "detail_loss", "object_missing", "frozen_animation",
                    "geometry_explosion"]
ANALYSIS_WIDTH = 640  # the detectors are calibrated around this width


@dataclass
class Clip:
    path: Path
    fps: float
    size: tuple[int, int]  # native (w, h)
    frames: list[np.ndarray]  # downscaled to ANALYSIS_WIDTH
    every: int  # frame stride used when reading

    @property
    def scale(self) -> float:
        return self.frames[0].shape[1] / self.size[0] if self.frames else 1.0


@dataclass
class VideoFinding:
    id: str
    detector: str
    shot: int
    start: int  # frame index in the recording
    end: int
    score: float
    confidence: float
    region: list[int] | None  # in analysis pixels
    note: str
    metrics: dict = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    windows: int = 1
    zoomed: list[str] = field(default_factory=list)
    status: str = "candidate"
    history: list[dict] = field(default_factory=list)

    def set_status(self, status: str, why: str) -> None:
        self.history.append({"from": self.status, "to": status, "why": why})
        self.status = status

    def timestamp(self, fps: float) -> str:
        return f"{_ts(self.start / fps)}-{_ts(self.end / fps)}"


def _ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def read_video(path: Path, every: int = 1, max_frames: int | None = None, start: int = 0,
               width: int = ANALYSIS_WIDTH) -> Clip:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    if start:
        cap.set(cv.CAP_PROP_POS_FRAMES, start)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            if width and frame.shape[1] != width:
                h = int(round(frame.shape[0] * width / frame.shape[1]))
                frame = cv.resize(frame, (width, h), interpolation=cv.INTER_AREA)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return Clip(path, fps, size, frames, every)


def shots(frames: list[np.ndarray], cut: float = 0.5) -> list[tuple[int, int]]:
    """Shot boundaries: a frame that differs sharply from the one before (colour histogram or
    pixels) while the frames after it agree with it. A single odd frame (a flash) is not a cut."""
    hists, small = [], []
    for f in frames:
        hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        h = cv.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv.normalize(h, h, 1, 0, cv.NORM_L1)
        hists.append(h)
        small.append(cv.resize(cv.cvtColor(f, cv.COLOR_BGR2GRAY), (80, 45), interpolation=cv.INTER_AREA)
                     .astype(np.float32))

    def corr(a, b):
        return cv.compareHist(hists[a], hists[b], cv.HISTCMP_CORREL)

    def pix(a, b):
        return float(np.abs(small[a] - small[b]).mean())

    steps = [0.0] + [pix(t - 1, t) for t in range(1, len(frames))]
    bounds = [0]
    for t in range(1, len(frames)):
        local = np.median(steps[max(1, t - 6):t] + steps[t + 1:t + 7]) if len(frames) > 2 else 0.0
        jump = steps[t] > max(18.0, 4.0 * local)
        if corr(t - 1, t) >= cut and not jump:
            continue
        if t + 1 < len(frames) and (corr(t - 1, t + 1) >= cut and pix(t - 1, t + 1) < 0.5 * steps[t]):
            continue  # one frame differs from both neighbours: a flash or a dropped frame, not a new shot
        if t - bounds[-1] >= 4:
            bounds.append(t)
    bounds.append(len(frames))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 4]


def camera_moves(frames: list[np.ndarray]) -> bool:
    """Whether the camera itself moves (median optical flow over the frame)."""
    sample = frames[:: max(1, len(frames) // 8)][:9]
    if len(sample) < 2:
        return False
    flows = flow_pairs(gray_list(sample, 0.5))
    return float(np.median(np.abs(flows))) * 2 > 0.4


class VideoTriage:
    def __init__(self, path: Path, out_dir: Path, golden: Path | None = None, every: int = 1, window: int = 48,
                 max_frames: int | None = None):
        self.path = Path(path)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "evidence").mkdir(exist_ok=True)
        self.trace = Trace(self.out / "trace.jsonl")
        self.every, self.window, self.max_frames = every, window, max_frames
        self.clip = read_video(self.path, every=every, max_frames=max_frames)
        self.golden = read_video(Path(golden), every=every, max_frames=max_frames) if golden else None
        self.findings: list[VideoFinding] = []
        self.shots = shots(self.clip.frames)

    # --- scan ----------------------------------------------------------------------------------------
    def scan(self) -> None:
        names = VIDEO_DETECTORS + (GOLDEN_DETECTORS if self.golden else [])
        self.trace.log("decision", step=1, planner="video", tool="scan",
                       args={"shots": len(self.shots), "frames": len(self.clip.frames)},
                       reason=f"{len(self.shots)} shot(s) found by histogram cuts; run {len(names)} detectors over "
                              f"{self.window}-frame windows")
        for s_idx, (a, b) in enumerate(self.shots):
            moving = camera_moves(self.clip.frames[a:b])
            starts = list(range(a, max(a + 1, b - self.window + 1), max(1, self.window // 2))) or [a]
            for w0 in starts:
                w1 = min(b, w0 + self.window)
                frames = self.clip.frames[w0:w1]
                if len(frames) < 6:
                    continue
                golden = None
                if self.golden and len(self.golden.frames) >= w1:
                    golden = self.golden.frames[w0:w1]
                ctx = DetectContext(motion_expected=moving, golden=golden, camera_static=not moving,
                                    fps=self.clip.fps / self.every, lossy=True)
                for det in run_detectors(frames, [n for n in names if golden or not DETECTORS[n].needs_golden], ctx):
                    self._register(det, s_idx, w0, w1)
        self.trace.log("result", step=1, tool="scan", ok=True,
                       summary=f"{len(self.findings)} candidate(s): "
                               + ", ".join(f"{f.id}={f.detector}@{f.timestamp(self.clip.fps / self.every)}"
                                           for f in self.findings))

    def _register(self, det: Detection, shot: int, w0: int, w1: int) -> None:
        first = w0 + (det.frames[0] if det.frames else 0)
        last = w0 + (det.frames[-1] if det.frames else w1 - w0 - 1)
        for f in self.findings:
            same_place = det.region is None or f.region is None or iou(det.region, f.region) > 0.2
            if det.detector == "pop_in" and f.detector == "pop_in" and f.shot == shot and abs(first - f.start) <= 1:
                same_place = True  # several regions popping on the same frame are one event
            if f.detector == det.detector and f.shot == shot and same_place and first <= f.end + self.window:
                f.start, f.end = min(f.start, first), max(f.end, last)
                f.windows += 1
                if det.score * (0.5 + det.confidence) > f.score * (0.5 + f.confidence):
                    f.score, f.confidence, f.note = det.score, det.confidence, det.note
                    f.region = list(det.region) if det.region else None
                    f.metrics = det.to_dict()["metrics"]
                    f.evidence = self._save(det, f.id)
                return
        f = VideoFinding(f"V{len(self.findings) + 1}", det.detector, shot, first, last, det.score, det.confidence,
                         list(det.region) if det.region else None, det.note, det.to_dict()["metrics"])
        f.evidence = self._save(det, f.id)
        self.findings.append(f)

    def _save(self, det: Detection, fid: str) -> dict[str, str]:
        paths = {}
        for name, img in det.evidence.items():
            p = self.out / "evidence" / f"{fid}-{det.detector}-{name}.png"
            cv.imwrite(str(p), img)
            paths[name] = str(p)
        return paths

    # --- investigate ---------------------------------------------------------------------------------
    def investigate(self, step: int = 2) -> None:
        for f in self.findings:
            spec = DETECTORS[f.detector]
            if f.confidence < 0.6 and spec.localized and f.region:
                self.trace.log("decision", step=step, planner="video", tool="zoom", args={"finding_id": f.id},
                               reason=f"{f.id} {f.detector}: confidence {f.confidence:.2f} < 0.6 on a small region; "
                                      "re-read it at native resolution and every frame")
                before = f.confidence
                self._zoom(f)
                self.trace.log("result", step=step, tool="zoom", ok=True,
                               summary=f"{f.id} confidence {before:.2f} -> {f.confidence:.2f}")
                step += 1
            if f.windows >= 2 and f.confidence < 0.9:
                f.confidence = min(0.95, f.confidence + 0.05 * (f.windows - 1))
                f.zoomed.append(f"seen in {f.windows} windows")
            if f.confidence < 0.3:
                f.set_status("dismissed", f"evidence stayed weak (confidence {f.confidence:.2f})")
            else:
                f.set_status("confirmed", f"confidence {f.confidence:.2f} over {f.windows} window(s)")
            self.trace.log("decision", step=step, planner="video",
                           tool="dismiss" if f.status == "dismissed" else "keep", args={"finding_id": f.id},
                           reason=f.history[-1]["why"])
            step += 1

    def _zoom(self, f: VideoFinding) -> None:
        """Spatial and temporal zoom: native resolution, every frame, around the finding."""
        a = max(0, f.start * self.every - 8)
        n = (f.end - f.start + 1) * self.every + 16
        native = read_video(self.path, every=1, max_frames=n, start=a, width=0)
        if not native.frames:
            return
        s = native.frames[0].shape[1] / self.clip.frames[0].shape[1]
        x, y, w, h = f.region
        pad = int(max(w, h) * 0.6) + 8
        x0, y0 = max(0, int((x - pad) * s)), max(0, int((y - pad) * s))
        x1 = min(native.frames[0].shape[1], int((x + w + pad) * s))
        y1 = min(native.frames[0].shape[0], int((y + h + pad) * s))
        up = max(1.0, 320.0 / max(1, x1 - x0))
        crops = [cv.resize(fr[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv.INTER_CUBIC) for fr in native.frames]
        moving = camera_moves(crops)
        ctx = DetectContext(motion_expected=moving, camera_static=not moving, fps=native.fps, lossy=True)
        dets = DETECTORS[f.detector].fn(crops, ctx)
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            f.confidence = max(f.confidence, best.confidence)
            f.zoomed.append(f"zoom x{up * s:.1f}: confidence {best.confidence:.2f}")
            for name, img in best.evidence.items():
                p = self.out / "evidence" / f"{f.id}-{f.detector}-zoom-{name}.png"
                cv.imwrite(str(p), img)
                f.evidence[f"zoom_{name}"] = str(p)
        else:
            f.confidence *= 0.5
            f.zoomed.append(f"zoom x{up * s:.1f}: not visible up close; confidence halved")

    # --- report --------------------------------------------------------------------------------------
    def report(self, approve=None) -> Path:
        fps = self.clip.fps / self.every
        keep = [f for f in self.findings if f.status != "dismissed"]
        for f in keep:
            choice = approve(f) if approve else "approve"
            f.set_status("reported" if choice == "approve" else "rejected", f"human decision: {choice}")
        lines = [f"# Video triage: `{self.path.name}`", "",
                 f"- {len(self.clip.frames)} frames analysed ({self.clip.size[0]}x{self.clip.size[1]} at "
                 f"{self.clip.fps:.0f} fps, every {self.every} frame(s)), {len(self.shots)} shot(s)",
                 f"- Golden recording: {'yes' if self.golden else 'no (golden comparisons off)'}",
                 f"- Findings: {len(keep)} kept, {len(self.findings) - len(keep)} dismissed", "",
                 "| id | glitch | time | shot | confidence | status | note |", "|---|---|---|---|---|---|---|"]
        for f in self.findings:
            lines.append(f"| {f.id} | `{f.detector}` | {f.timestamp(fps)} | {f.shot + 1} | {f.confidence:.2f} | "
                         f"{f.status} | {f.note} |")
        for f in keep:
            lines += ["", f"## {f.id}: {f.detector} at {f.timestamp(fps)}", "", f.note, ""]
            if f.zoomed:
                lines.append("Investigation: " + "; ".join(f.zoomed))
                lines.append("")
            for name, p in f.evidence.items():
                rel = Path(p).relative_to(self.out).as_posix()
                lines.append(f"**{name.replace('_', ' ')}**\n\n![{name}]({rel})\n")
        path = self.out / "REPORT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.out / "findings.json").write_text(json.dumps([f.__dict__ for f in self.findings], indent=2,
                                                           default=str), encoding="utf-8")
        sheet = ev.contact_sheet(self.clip.frames[:: max(1, len(self.clip.frames) // 48)], cols=8, thumb_w=120)
        cv.imwrite(str(self.out / "contact_sheet.jpg"), sheet)
        return path

    def run(self, approve=None) -> dict:
        t0 = time.time()
        self.trace.log("hunt_start", hunt=self.out.name, planner="video", head=self.path.name, baseline="-")
        self.scan()
        self.investigate()
        report = self.report(approve)
        out = {"video": str(self.path), "frames": len(self.clip.frames), "shots": len(self.shots),
               "findings": [{"id": f.id, "detector": f.detector, "time": f.timestamp(self.clip.fps / self.every),
                             "confidence": round(f.confidence, 2), "status": f.status} for f in self.findings],
               "report": str(report), "seconds": round(time.time() - t0, 1)}
        self.trace.log("hunt_end", ended="scan complete", steps=len(self.findings) + 1, captures=0,
                       seconds=out["seconds"])
        return out


def write_video(frames: list[np.ndarray], path: Path, fps: float = 30.0) -> Path:
    """MP4 (mp4v) from BGR frames; used by `record` and the tests."""
    h, w = frames[0].shape[:2]
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4v writer")
    for f in frames:
        writer.write(f)
    writer.release()
    return path
