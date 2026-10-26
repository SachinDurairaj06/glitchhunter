"""Benchmark of GlitchHunter's OpenCV 5 workload: run it once with pip OpenCV 5 and once with COOL.

The same script runs both builds on the same machine (e.g. an AWS Graviton instance):
  python -m glitchhunter bench --out bench/results/pip.json
  GH_OPENCV_DIST=cool python -m glitchhunter bench --out bench/results/cool.json   (COOL's cv2 on the path)
  python -m glitchhunter bench --compare bench/results/pip.json bench/results/cool.json

Frames are synthetic by default (deterministic 640x360 game-like scenes, so no browser is
needed on the server); --frames <dir> benchmarks real captured frames instead.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .cvinfo import opencv_info
from .textdnn import TextReader, models_available
from .vision import BUG_DETECTORS, DetectContext, run_detectors, ssim_map


def synthetic_frames(n: int = 24, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.zeros((360, 640, 3), np.uint8)
    base[:] = (60, 70, 85)
    tex = cv.resize(rng.uniform(0, 60, (45, 80)).astype(np.float32), (640, 360), interpolation=cv.INTER_CUBIC)
    base = np.clip(base + tex[..., None], 0, 255).astype(np.uint8)
    for x in range(0, 640, 64):
        cv.line(base, (x, 120), (x - 200 + x // 2, 359), (40, 45, 50), 2)
    cv.rectangle(base, (230, 140), (270, 185), (52, 112, 168), -1)
    cv.rectangle(base, (18, 18), (188, 78), (74, 42, 27), -1)
    cv.rectangle(base, (18, 18), (188, 78), (255, 168, 111), 3)
    cv.putText(base, "HP 100   PTS 0", (30, 56), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
    frames = []
    for t in range(n):
        f = base.copy()
        cv.circle(f, (320, 260 - 3 * t), 18, (217, 123, 61), -1)  # moving player
        if t % 2:
            cv.rectangle(f, (380, 170), (430, 190), (190, 200, 40), -1)  # flickering decal
        frames.append(f)
    return frames


def load_frames(directory: str) -> list[np.ndarray]:
    paths = sorted(Path(directory).glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNG frames in {directory}")
    return [cv.imread(str(p)) for p in paths]


def _time(fn, iterations: int) -> float:
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - t0) / iterations * 1000.0


def run_bench(frames: list[np.ndarray], iterations: int) -> dict:
    f0, f1 = frames[0], frames[1]
    gray = cv.cvtColor(f0, cv.COLOR_BGR2GRAY)
    mask = (gray > 100).astype(np.uint8) * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    acc, acc2 = np.zeros(gray.shape, np.float32), np.zeros(gray.shape, np.float32)
    gray1 = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    dis_engine = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(gray.shape[1], dtype=np.float32), np.arange(gray.shape[0], dtype=np.float32))
    gx, gy = gx + 0.5, gy + 0.25
    ops = {
        "cvtColor BGR2GRAY": lambda: cv.cvtColor(f0, cv.COLOR_BGR2GRAY),
        "cvtColor BGR2HSV": lambda: cv.cvtColor(f0, cv.COLOR_BGR2HSV),
        "cvtColor BGR2Lab": lambda: cv.cvtColor(f0, cv.COLOR_BGR2Lab),
        "inRange (magenta)": lambda: cv.inRange(f0, (138, 100, 60), (162, 255, 255)),
        "morphologyEx open 5x5": lambda: cv.morphologyEx(mask, cv.MORPH_OPEN, k),
        "connectedComponentsWithStats": lambda: cv.connectedComponentsWithStats(mask),
        "absdiff": lambda: cv.absdiff(f0, f1),
        "norm INF": lambda: cv.norm(f0, f1, cv.NORM_INF),
        "GaussianBlur 11x11": lambda: cv.GaussianBlur(gray, (11, 11), 1.5),
        "Canny": lambda: cv.Canny(gray, 60, 160),
        "findContours": lambda: cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE),
        "accumulate + accumulateSquare": lambda: (cv.accumulate(gray, acc), cv.accumulateSquare(gray, acc2)),
        "resize INTER_AREA 0.5x": lambda: cv.resize(f0, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA),
        "imencode PNG": lambda: cv.imencode(".png", f0),
        "SSIM map": lambda: ssim_map(f0, f1),
        "DIS optical flow (medium)": lambda: dis_engine.calc(gray, gray1, None),
        "remap (flow warp)": lambda: cv.remap(gray, gx, gy, cv.INTER_LINEAR),
        "medianBlur 3x3": lambda: cv.medianBlur(gray, 3),
        "Laplacian": lambda: cv.Laplacian(gray, cv.CV_32F),
        "calcHist + compareHist": lambda: cv.compareHist(cv.calcHist([f0], [0], None, [64], [0, 256]),
                                                        cv.calcHist([f1], [0], None, [64], [0, 256]),
                                                        cv.HISTCMP_BHATTACHARYYA),
    }
    results = {name: round(_time(fn, iterations), 4) for name, fn in ops.items()}

    if models_available():
        for engine in ("ENGINE_AUTO", "ENGINE_CLASSIC"):
            reader = TextReader(engine=getattr(cv.dnn, engine))
            results[f"DNN text detect+recognize ({engine.split('_')[1].lower()})"] = round(
                _time(lambda: reader.read(f0), max(3, iterations // 5)), 3)

    golden = frames
    suite_iters = max(3, iterations // 5)
    suite = round(_time(lambda: run_detectors(frames, [d for d in BUG_DETECTORS if d != "hud_text_overflow"],
                                              DetectContext(True, golden)), suite_iters), 2)
    return {"ops_ms": results, "detector_suite_ms_per_run": suite, "frames_per_run": len(frames)}


def compare(a_path: str, b_path: str) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    name_a, name_b = a["opencv"]["distribution"], b["opencv"]["distribution"]
    print(f"| op | {name_a} (ms) | {name_b} (ms) | speedup |\n|---|---|---|---|")
    for op, ms in a["ops_ms"].items():
        other = b["ops_ms"].get(op)
        if other:
            print(f"| {op} | {ms:.3f} | {other:.3f} | {ms / other:.2f}x |")
    sa, sb = a["detector_suite_ms_per_run"], b["detector_suite_ms_per_run"]
    print(f"| detector suite per run | {sa:.1f} | {sb:.1f} | {sa / sb:.2f}x |")
    if a.get("cost_per_1000_runs_usd") and b.get("cost_per_1000_runs_usd"):
        print(f"\ncost per 1000 runs: {name_a} ${a['cost_per_1000_runs_usd']:.4f} vs {name_b} ${b['cost_per_1000_runs_usd']:.4f}")
    return 0


def main(args) -> int:
    if getattr(args, "compare", None):
        return compare(*args.compare)
    frames = load_frames(args.frames) if getattr(args, "frames", None) else synthetic_frames()
    out = {"opencv": opencv_info(), "platform": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M"),
           **run_bench(frames, args.iterations)}
    price = getattr(args, "price_per_hour", None)
    if price:
        runs_per_hour = 3600_000.0 / out["detector_suite_ms_per_run"]
        out["instance_price_per_hour_usd"] = price
        out["cost_per_1000_runs_usd"] = round(price / runs_per_hour * 1000, 6)
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0
