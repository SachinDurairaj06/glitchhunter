"""The Lab: everything the agent's tools act on.

It owns the game-capture worker (one thread, because Playwright's sync API is
single-threaded), the run store, golden frames from the baseline build, OpenCV
inspection with saved evidence, and git operations on the game repo. The game's
test plan (scenarios, baseline, screen sizes) comes from its glitchhunter.json.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from . import evidence as ev
from .capture import BuildError, Capture, GameDriver, git
from .config import Settings
from .obs import metrics
from .project import Project, Scenario
from .vision import DETECTORS, DetectContext, Detection, expand, run_detectors

__all__ = ["BuildError", "Lab", "Run", "repeats_for", "zoom_camera"]


@dataclass
class Run:
    id: str
    sha: str
    scenario: str
    camera: dict | None
    seed: int
    seconds: float
    errors: list[dict]
    console_errors: list[str]
    dir: Path
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    viewport: tuple[int, int] | None = None  # None = the project's default screen size

    def summary(self) -> dict:
        return {"run_id": self.id, "commit": self.sha[:10], "scenario": self.scenario, "camera": self.camera,
                "seed": self.seed, "frames": len(self.frames), "seconds": round(self.seconds, 2),
                "viewport": list(self.viewport) if self.viewport else None,
                "game_errors": self.errors[:5], "sheet": str(self.dir / "sheet.jpg")}


def _camera_key(camera: dict | None) -> str:
    if not camera:
        return "default"
    return json.dumps({k: [round(v, 3) for v in camera[k]] if isinstance(camera[k], list) else camera[k]
                       for k in sorted(camera) if k in ("pos", "target", "fov")}, sort_keys=True)


def viewport_label(viewport) -> str:
    return f"{viewport[0]}x{viewport[1]}" if viewport else "default"


def make_driver(settings: Settings, project: Project):
    """The driver for the project's engine: a browser for web builds, the bridge for native ones."""
    kind = (project.driver or {}).get("type", "web")
    if kind == "bridge":
        from .bridge import BridgeDriver

        return BridgeDriver(settings.fixture_dir, settings.checkout_dir, project, project.viewport)
    if kind != "web":
        raise ValueError(f"unknown driver type {kind!r} in glitchhunter.json (use 'web' or 'bridge')")
    return GameDriver(settings.fixture_dir, settings.checkout_dir, headless=settings.headless, entry=project.entry,
                      viewport=project.viewport)


class DriverWorker:
    """Runs every driver call on one dedicated thread."""

    def __init__(self, settings: Settings, project: Project):
        self.settings = settings
        self.project = project
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="game-driver")
        self._driver: GameDriver | None = None

    def call(self, fn, timeout: float = 300.0):
        def task():
            if self._driver is None:
                self._driver = make_driver(self.settings, self.project)
            return fn(self._driver)

        return self._executor.submit(task).result(timeout=timeout)

    def close(self) -> None:
        def task():
            if self._driver is not None:
                self._driver.close()
                self._driver = None

        try:
            self._executor.submit(task).result(timeout=60)
        finally:
            self._executor.shutdown(wait=False)


class DriverPool:
    """Several capture workers (each with its own browser or game process) used in parallel.

    Captures are independent, so a survey of five scenarios or the regression check after a fix
    can run on two or three workers at once. GH_WORKERS sets the size (default 2).
    """

    def __init__(self, settings: Settings, project: Project, size: int):
        self.workers = [DriverWorker(settings, project) for _ in range(max(1, size))]
        self._free: "queue.Queue[DriverWorker]" = queue.Queue()
        for w in self.workers:
            self._free.put(w)

    @property
    def size(self) -> int:
        return len(self.workers)

    def call(self, fn, timeout: float = 300.0):
        worker = self._free.get()
        try:
            return worker.call(fn, timeout)
        finally:
            self._free.put(worker)

    def close(self) -> None:
        for w in self.workers:
            w.close()


class Lab:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repo = settings.fixture_dir
        if not (self.repo / ".git").exists():
            raise RuntimeError(f"no game repo at {self.repo}; run `python -m glitchhunter fixture` first")
        self.project = Project.load(self.repo)
        workers = int(os.environ.get("GH_WORKERS", "2"))
        self.worker = DriverPool(settings, self.project, workers)
        self._golden_locks: dict[tuple, threading.Lock] = {}
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._golden: dict[tuple, list[np.ndarray]] = {}
        self._lock = threading.Lock()
        self.keep_frames = os.environ.get("GH_KEEP_FRAMES", "key")  # all | key | none
        self.capture_count = 0
        self.prune_runs(int(os.environ.get("GH_MAX_RUNS", "400")))

    def prune_runs(self, max_runs: int) -> int:
        """Delete the oldest capture folders beyond max_runs (reports keep their own image copies)."""
        runs = sorted(p for p in self.settings.runs_dir.iterdir() if p.is_dir())
        stale = runs[:max(0, len(runs) - max_runs)]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
        return len(stale)

    def close(self) -> None:
        self.worker.close()

    # --- git -------------------------------------------------------------------------------------
    def resolve(self, rev: str) -> str:
        return git(self.repo, "rev-parse", "--verify", f"{rev}^{{commit}}")

    def head(self) -> str:
        return self.resolve(self.project.branch)

    def baseline(self) -> str:
        return self.resolve(self.project.baseline)

    def commits(self, limit: int = 50) -> list[dict]:
        out = git(self.repo, "log", "--first-parent", f"-n{limit}", "--format=%H%x09%P%x09%s%x09%D",
                  self.project.branch)
        rows = []
        for line in out.splitlines():
            sha, parents, subject, refs = (line.split("\t") + ["", "", ""])[:4]
            tags = [r.split("tag: ")[1] for r in refs.split(", ") if r.startswith("tag: ")]
            rows.append({"sha": sha, "short": sha[:10], "subject": subject, "tags": tags,
                         "merge": len(parents.split()) > 1})
        return rows

    def commits_between(self, good: str, bad: str) -> list[str]:
        """First-parent commits after good up to and including bad, oldest first."""
        out = git(self.repo, "rev-list", "--first-parent", "--reverse", f"{good}..{bad}")
        return [line for line in out.splitlines() if line]

    def parents(self, rev: str) -> list[str]:
        return git(self.repo, "rev-list", "--parents", "-n1", rev).split()[1:]

    def side_commits(self, merge: str) -> list[str]:
        """Commits a merge brought in from its second parent (the merged branch), oldest first."""
        parents = self.parents(merge)
        if len(parents) < 2:
            return []
        out = git(self.repo, "rev-list", "--first-parent", "--reverse", parents[1], f"^{parents[0]}")
        return [line for line in out.splitlines() if line]

    def subject(self, rev: str) -> str:
        return git(self.repo, "show", "-s", "--format=%s", rev)

    def delete_branch(self, branch: str) -> None:
        if not branch.startswith("glitchhunter/"):
            raise ValueError("only agent branches (glitchhunter/*) can be deleted")
        git(self.repo, "branch", "-D", branch)

    def show_commit(self, rev: str, max_lines: int = 120) -> dict:
        sha = self.resolve(rev)
        meta = git(self.repo, "show", "-s", "--format=%H%n%an%n%ad%n%s", sha).splitlines()
        parents = self.parents(sha)
        paths = list(self.project.patch_paths)
        if len(parents) > 1:  # a merge: what it changed relative to the mainline
            patch = git(self.repo, "diff", "--unified=2", parents[0], sha, "--", *paths)
            files = git(self.repo, "diff", "--name-only", parents[0], sha).splitlines()
        else:
            patch = git(self.repo, "show", "--format=", "--unified=2", sha, "--", *paths)
            files = git(self.repo, "show", "--format=", "--name-only", sha).splitlines()
        lines = patch.splitlines()
        return {"sha": meta[0], "author": meta[1], "date": meta[2], "subject": meta[3], "files": files,
                "merge": len(parents) > 1,
                "diff": "\n".join(lines[:max_lines]) + ("\n... (truncated)" if len(lines) > max_lines else "")}

    # --- capture -----------------------------------------------------------------------------------
    def scenario(self, name: str) -> Scenario:
        return self.project.scenario(name)

    def capture(self, rev: str, scenario: str, camera: dict | None = None, seed: int | None = None,
                keep: bool = True, purpose: str = "", viewport: tuple[int, int] | None = None) -> Run:
        """Play one scenario on one build. Raises BuildError when the build does not start."""
        sc = self.scenario(scenario)
        seed = self.settings.seed if seed is None else seed
        viewport = tuple(viewport) if viewport else None
        cap: Capture = self.worker.call(lambda d: d.run(rev, sc, camera, seed, viewport))
        with self._lock:
            self.capture_count += 1
        run_id = time.strftime("%m%d%H%M%S") + secrets.token_hex(2)
        run_dir = self.settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run = Run(run_id, cap.sha, scenario, camera, seed, cap.seconds, cap.errors, cap.console_errors,
                  run_dir, cap.frames, viewport)
        cv.imwrite(str(run_dir / "sheet.jpg"), ev.contact_sheet(cap.frames, cols=8, thumb_w=120),
                   [cv.IMWRITE_JPEG_QUALITY, 85])
        if self.keep_frames == "all" or (self.keep_frames == "key" and keep):
            for i, frame in enumerate(cap.frames):
                cv.imwrite(str(run_dir / f"f{i:03d}.png"), frame, [cv.IMWRITE_PNG_COMPRESSION, 1])
        (run_dir / "meta.json").write_text(json.dumps({**run.summary(), "purpose": purpose}, indent=2))
        metrics.emit("capture_seconds", cap.seconds, unit="Seconds", scenario=scenario)
        with self._lock:
            self._runs[run_id] = run
            while len(self._runs) > 12:
                self._runs.popitem(last=False)
        return run

    def get_run(self, run_id: str, need_frames: bool = True) -> Run:
        with self._lock:
            if run_id in self._runs:
                return self._runs[run_id]
        run_dir = self.settings.runs_dir / run_id
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            raise KeyError(f"unknown run {run_id!r}")
        meta = json.loads(meta_path.read_text())
        frames = [cv.imread(str(p)) for p in sorted(run_dir.glob("f*.png"))] if need_frames else []
        if need_frames and not frames:
            raise KeyError(f"frames for run {run_id} were not kept (GH_KEEP_FRAMES={self.keep_frames})")
        sha = self.resolve(meta["commit"])
        viewport = tuple(meta["viewport"]) if meta.get("viewport") else None
        return Run(run_id, sha, meta["scenario"], meta.get("camera"), meta.get("seed", 1),
                   meta.get("seconds", 0.0), meta.get("game_errors", []), [], run_dir, frames, viewport)

    def golden(self, scenario: str, camera: dict | None, viewport=None) -> list[np.ndarray]:
        key = (scenario, _camera_key(camera), viewport_label(viewport))
        with self._lock:
            lock = self._golden_locks.setdefault(key, threading.Lock())
        with lock:  # parallel captures of the same view wait for one golden capture
            if key not in self._golden:
                run = self.capture(self.project.baseline, scenario, camera, keep=False, purpose="golden",
                                   viewport=viewport)
                self._golden[key] = run.frames
        return self._golden[key]

    def capture_many(self, requests: list[dict]) -> list:
        """Run several captures in parallel on the worker pool; results (Run or the exception) in order.

        Each request holds Lab.capture keyword arguments: rev, scenario, camera, seed, keep, purpose,
        viewport.
        """
        if self.worker.size <= 1 or len(requests) <= 1:
            out = []
            for r in requests:
                try:
                    out.append(self.capture(**r))
                except Exception as exc:  # noqa: BLE001 - returned to the caller in order
                    out.append(exc)
            return out
        with ThreadPoolExecutor(max_workers=self.worker.size, thread_name_prefix="capture") as pool:
            futures = [pool.submit(self.capture, **r) for r in requests]
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as exc:  # noqa: BLE001
                    results.append(exc)
            return results

    def pick(self, run: Run, x: float, y: float, frame_index: int = 0) -> dict:
        sc = self.scenario(run.scenario)
        return self.worker.call(lambda d: d.pick(run.sha, sc, run.camera, frame_index, x, y, run.seed,
                                                 run.viewport))

    # --- inspection --------------------------------------------------------------------------------
    def context(self, run: Run, names: list[str], roi=None) -> DetectContext:
        sc = self.scenario(run.scenario)
        golden = None
        if any(DETECTORS[n].needs_golden or DETECTORS[n].golden_suppress for n in names):
            golden = self.golden(run.scenario, run.camera, run.viewport)
        return DetectContext(motion_expected=sc.motion_expected, golden=golden, roi=roi,
                             camera_static=sc.camera_static, fps=self.project.fps,
                             key=(run.scenario, _camera_key(run.camera), viewport_label(run.viewport)))

    def inspect(self, run: Run, detectors: list[str] | None = None,
                region: tuple[int, int, int, int] | None = None, tag: str = "") -> list[tuple[Detection, dict]]:
        """Run OpenCV detectors on a run; returns (detection, {evidence name: path}) pairs."""
        names = detectors or [n for n, s in DETECTORS.items() if s.severity != "info"]
        unknown = [n for n in names if n not in DETECTORS]
        if unknown:
            raise KeyError(f"unknown detectors {unknown}; choose from {sorted(DETECTORS)}")
        roi = expand(region, run.frames[0].shape) if region else None
        ctx = self.context(run, names, roi)
        t0 = time.perf_counter()
        dets = run_detectors(run.frames, names, ctx)
        metrics.emit("detect_seconds", time.perf_counter() - t0, unit="Seconds", detectors=len(names))
        out = []
        ev_dir = run.dir / "ev"
        for k, det in enumerate(dets):
            paths = {}
            if det.evidence:
                ev_dir.mkdir(exist_ok=True)
                for name, img in det.evidence.items():
                    path = ev_dir / f"{tag}{det.detector}-{k}-{name}.png"
                    cv.imwrite(str(path), img)
                    paths[name] = str(path)
            out.append((det, paths))
        return out

    # --- fixes -------------------------------------------------------------------------------------
    def make_fix(self, culprit: str, label: str, strategy: str = "revert", patch: str | None = None) -> dict:
        """Create branch glitchhunter/<label> = main + fix, in a throwaway worktree. Never touches main."""
        branch = f"glitchhunter/{re.sub(r'[^a-z0-9._-]+', '-', label.lower())}"
        wt = self.settings.worktree_dir / re.sub(r"[^a-z0-9]+", "", label.lower())[:16]
        if wt.exists():
            try:
                git(self.repo, "worktree", "remove", "--force", str(wt))
            except RuntimeError:
                shutil.rmtree(wt, ignore_errors=True)
        git(self.repo, "worktree", "prune")
        wt.parent.mkdir(parents=True, exist_ok=True)
        git(self.repo, "worktree", "add", "-q", "-B", branch, str(wt), self.project.branch)
        try:
            env_git = ["-c", "user.name=GlitchHunter", "-c", "user.email=glitchhunter@agent.invalid"]
            if strategy == "revert":
                sha = self.resolve(culprit)
                mainline = ["-m", "1"] if len(self.parents(sha)) > 1 else []  # a merge: undo the merged side
                git(wt, *env_git, "revert", "--no-edit", *mainline, sha)
            elif strategy == "patch":
                if not patch:
                    raise ValueError("strategy 'patch' needs a unified diff in `patch`")
                check_patch(patch, self.project)
                patch_file = wt / ".gh.patch"
                # newline="\n": on Windows text mode would write CRLF and git apply would reject it
                patch_file.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8",
                                      newline="\n")
                try:
                    git(wt, "apply", "--whitespace=nowarn", str(patch_file))
                finally:
                    patch_file.unlink(missing_ok=True)
                git(wt, "add", "-A")
                git(wt, *env_git, "commit", "-q", "-m", f"glitchhunter: fix for {label}")
            else:
                raise ValueError(f"unknown fix strategy {strategy!r}")
            sha = git(wt, "rev-parse", "HEAD")
            diff = git(self.repo, "show", "--format=%s", "--unified=2", sha)
        except Exception:
            git(self.repo, "worktree", "remove", "--force", str(wt))
            raise
        git(self.repo, "worktree", "remove", "--force", str(wt))
        return {"branch": branch, "sha": sha, "strategy": strategy, "diff": diff}


def check_patch(patch: str, project: Project) -> None:
    """Patches proposed by the language model may only touch the game files the project allows."""
    if len(patch) > 20_000:
        raise ValueError("patch too large")
    for line in patch.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].strip().split("\t")[0]
            if path == "/dev/null":
                continue
            path = re.sub(r"^[ab]/", "", path)
            if not project.allows_patch_path(path):
                raise ValueError(f"patch touches a file outside the game: {path}")


def zoom_camera(hit_point: list[float], camera_pos: list[float], factor: float, fov: float = 55.0) -> dict:
    """Move the camera along the line to the hit point so the region looks `factor` times bigger."""
    p = np.asarray(hit_point, np.float64)
    c = np.asarray(camera_pos, np.float64)
    factor = float(min(max(factor, 1.2), 8.0))
    new_pos = p + (c - p) / factor
    if new_pos[1] < 0.6:  # stay above the floor
        new_pos[1] = 0.6
    return {"pos": [round(float(v), 4) for v in new_pos], "target": [round(float(v), 4) for v in p], "fov": fov}


def repeats_for(repro_rate: float, miss_prob: float = 0.05, cap: int = 8) -> int:
    """Runs per commit so a flaky bug that shows up with probability p is missed < miss_prob of the time."""
    if repro_rate >= 0.999:
        return 1
    if repro_rate <= 0:
        return cap
    return int(min(cap, max(2, math.ceil(math.log(miss_prob) / math.log(1 - repro_rate)))))
