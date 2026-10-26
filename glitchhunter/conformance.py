"""Bridge conformance check: does a game build speak the GlitchHunter protocol, deterministically?

Run it while writing a bridge for an engine (Unity, Unreal, a custom engine), before writing
a test plan:

    python -m glitchhunter bridge-check -- MyGame.exe --gh-port {port} -screen-width {width} ...
    python -m glitchhunter bridge-check --repo path/to/game     # the launch command in its glitchhunter.json

Every protocol command is exercised and its answer checked. Determinism matters most: the same
seed and inputs must give the same frames, because replay, golden comparison and visual bisect
all rest on it, so that check also says how far apart two runs are and from which frame.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bridge import _Session, decode_png, launch


@dataclass
class Check:
    name: str
    status: str  # pass | info | warn | fail
    detail: str


def _play(s: _Session, frames: int, key: str | None = None) -> list[np.ndarray]:
    out = []
    for i in range(frames):
        if key and i == 1:
            s.call("key", code=key, down=True)
        if key and i == 4:
            s.call("key", code=key, down=False)
        s.call("advance", frames=1)
        out.append(decode_png(s.call("grab")))
    return out


def _digest(frames: list[np.ndarray]) -> list[str]:
    return [hashlib.sha1(f.tobytes()).hexdigest() for f in frames]


def _differs(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return any(x.shape != y.shape or not np.array_equal(x, y) for x, y in zip(a, b))


def run_checks(s: _Session, viewport: tuple[int, int], frames: int = 12, key: str = "Space") -> list[Check]:
    checks: list[Check] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append(Check(name, status, detail))

    hello = s.hello
    engine = hello.get("engine")
    add("hello", "pass" if hello.get("api") == 1 and engine else "warn",
        f"engine {engine or '(not reported)'}, api {hello.get('api', '(not reported)')}, viewport "
        f"{hello.get('viewport', '(not reported)')}")

    t0 = time.perf_counter()
    s.call("reset", seed=1, timeout=60)
    add("reset", "pass", f"fresh scene in {time.perf_counter() - t0:.2f}s")

    first = decode_png(s.call("grab"))
    size = (first.shape[1], first.shape[0])
    add("grab", "pass" if size == tuple(viewport) else "warn",
        f"PNG {size[0]}x{size[1]}" + ("" if size == tuple(viewport) else
                                       f" but {viewport[0]}x{viewport[1]} was asked for: honour --resolution / "
                                       "the screen size placeholders"))

    before = s.call("state").get("frame")
    reply = s.call("advance", frames=3)
    after = reply.get("frame", s.call("state").get("frame"))
    if before is None or after is None:
        add("advance", "warn", "no frame counter in the advance or state reply")
    else:
        add("advance", "pass" if after - before == 3 else "fail",
            f"advance 3 moved the frame counter {before} -> {after}"
            + ("" if after - before == 3 else ": it must run exactly the frames asked for, then pause"))

    s.call("reset", seed=1, timeout=60)
    t0 = time.perf_counter()
    run_a = _play(s, frames)
    per_frame = (time.perf_counter() - t0) / frames
    s.call("reset", seed=1, timeout=60)
    run_b = _play(s, frames)
    same = [x == y for x, y in zip(_digest(run_a), _digest(run_b))]
    if all(same):
        add("determinism", "pass", f"{frames} frames identical across two runs with seed 1 "
                                   f"({per_frame * 1000:.0f} ms per frame advanced and grabbed)")
    else:
        k = same.index(False)
        diff = np.abs(run_a[k].astype(np.int16) - run_b[k].astype(np.int16))
        status = "warn" if float(diff.mean()) < 0.5 else "fail"
        add("determinism", status,
            f"runs diverge from frame {k} (mean difference {diff.mean():.2f}, max {int(diff.max())} levels): seed "
            "every random generator on reset, step a fixed delta time, and read time from the frame counter, "
            "not the wall clock (dithering or temporal AA noise shows up as a small mean difference)")
    if not _differs(run_a[:1], run_a[-1:]):
        add("animation", "info", "first and last frames are identical: fine for a still scene, but check that "
                                 "advance renders new frames while the game is paused between commands")

    s.call("reset", seed=2, timeout=60)
    run_seed = _play(s, frames)
    if _differs(run_a, run_seed):
        add("seed", "pass", "seed 2 gives different frames than seed 1")
    else:
        add("seed", "info", "seed 2 gives the same frames as seed 1 (fine if the game has no randomness)")

    state = s.call("state")
    missing = [k for k in ("frame", "alive", "errors") if k not in state]
    add("state", "pass" if not missing else "warn",
        "frame, alive and errors reported" if not missing else f"state lacks {', '.join(missing)}")

    s.call("reset", seed=1, timeout=60)
    cam = state.get("camera") or {"pos": [0, 8, 15], "target": [0, 1, 0], "fov": 55}
    moved = {"pos": [float(cam["pos"][0]) + 2.0, float(cam["pos"][1]) + 1.0, float(cam["pos"][2])],
             "target": list(cam.get("target") or [0, 0, 0]), "fov": float(cam.get("fov") or 55)}
    s.call("set_camera", **moved)
    shifted = _play(s, 2)
    add("set_camera", "pass" if _differs(run_a[:2], shifted) else "warn",
        "moving the camera changes the picture" if _differs(run_a[:2], shifted) else
        "the picture did not change after set_camera: zoom and camera paths will not work")

    s.call("reset", seed=1, timeout=60)
    pressed = _play(s, 6, key=key)
    add("key", "pass", f"{key} down/up accepted; " + ("the picture reacted" if _differs(run_a[:6], pressed) else
                                                       "no visible reaction in 6 frames (fine if the key is unbound)"))

    hit = s.call("pick", x=viewport[0] / 2, y=viewport[1] / 2)
    if "hit" not in hit:
        add("pick", "warn", "pick reply has no 'hit' field: findings will not be named after objects")
    else:
        add("pick", "pass", f"centre of the screen: {hit.get('object') or 'nothing'}"
            + (f" at {hit['distance']:.1f} m" if hit.get("hit") and isinstance(hit.get("distance"), (int, float)) else ""))

    s.call("quit", timeout=10)
    try:
        s.proc.wait(timeout=10)
        add("quit", "pass", "process exited")
    except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired
        s.proc.kill()
        add("quit", "warn", "process still running 10 s after quit (it was killed)")
    return checks


def check_build(launch_cmd: list[str], viewport: tuple[int, int] = (640, 360), cwd: Path | None = None,
                frames: int = 12, key: str = "Space", ready_timeout: float = 90.0) -> list[Check]:
    log = Path(tempfile.mkstemp(prefix="gh-bridge-check-", suffix=".log")[1])
    cwd = Path(cwd or Path.cwd())
    try:
        s = launch(launch_cmd, cwd, {"dir": cwd, "width": viewport[0], "height": viewport[1], "seed": 1}, log,
                   ready_timeout, "the game")
    except Exception as exc:  # noqa: BLE001 - reported as a failed check
        return [Check("connect", "fail", f"{exc} (launch: {' '.join(launch_cmd)}; log: {log})")]
    checks = [Check("connect", "pass", f"connected back on the port given as {{port}} (log: {log})")]
    try:
        checks += run_checks(s, viewport, frames, key)
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("protocol", "fail", f"{exc} | {s.log_tail()}"))
        s.close()
    return checks


def launch_from_repo(repo: Path) -> tuple[list[str], tuple[int, int], Path]:
    manifest = json.loads((Path(repo) / "glitchhunter.json").read_text(encoding="utf-8"))
    driver = manifest.get("driver") or {}
    if driver.get("type") != "bridge":
        raise ValueError("glitchhunter.json has no bridge driver (web games are checked by `hunt` directly)")
    return list(driver["launch"]), tuple(manifest.get("viewport") or (640, 360)), Path(repo)


def format_checks(checks: list[Check]) -> str:
    mark = {"pass": "PASS", "info": "INFO", "warn": "WARN", "fail": "FAIL"}
    lines = [f"  {mark[c.status]}  {c.name:<12} {c.detail}" for c in checks]
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    verdict = ("ready for GlitchHunter" if not fails and not warns else
               "works, with warnings" if not fails else "not ready: fix the failures above")
    return "\n".join(lines + ["", f"{len(checks)} checks, {fails} failed, {warns} warnings: {verdict}"])
