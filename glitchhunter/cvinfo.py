"""OpenCV 5 runtime checks and build facts (used by /api/health and the benchmark)."""

from __future__ import annotations

import os
import platform

import cv2 as cv


def opencv_major() -> int:
    return int(cv.__version__.split(".")[0])


def require_opencv5() -> None:
    if opencv_major() < 5:
        raise RuntimeError(
            f"GlitchHunter needs OpenCV 5.x, found {cv.__version__}. "
            "Install the pinned build: pip install -r requirements.txt"
        )


def _build_line(build: str, key: str) -> str:
    for line in build.splitlines():
        if key in line:
            return line.split(":", 1)[-1].strip()
    return ""


def opencv_info() -> dict:
    build = cv.getBuildInformation()
    engines = [name for name in ("ENGINE_AUTO", "ENGINE_NEW", "ENGINE_CLASSIC", "ENGINE_ORT") if hasattr(cv.dnn, name)]
    return {
        "version": cv.__version__,
        "major": opencv_major(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "modules": _build_line(build, "To be built"),
        "cpu_baseline": _build_line(build, "Baseline"),
        "cpu_dispatch": _build_line(build, "Dispatched code generation"),
        "parallel": _build_line(build, "Parallel framework"),
        "hal": _build_line(build, "Custom HAL"),
        "kleidicv": "KleidiCV" in build or "kleidicv" in build,
        "dnn_engines": engines,
        "threads": cv.getNumThreads(),
        "optimized": cv.useOptimized(),
        # COOL (Cloud-Optimized OpenCV Library) ships as an AWS Marketplace build for Graviton.
        # Set GH_OPENCV_DIST=cool on that AMI so reports record which build ran the workload.
        "distribution": os.environ.get("GH_OPENCV_DIST", "pip"),
    }
