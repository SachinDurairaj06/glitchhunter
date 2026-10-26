"""Plays a game build headlessly and captures frames.

A build is any commit of the game repo. CheckoutCache exports the commit with
`git archive`, a local static server hosts it, and GameDriver drives the page
in headless Chromium (WebGL through SwiftShader, so frames are deterministic)
through the game's `window.__gh` test API: set the camera, press keys, advance
one fixed time step, grab the canvas, ray-cast a pixel.
"""

from __future__ import annotations

import base64
import functools
import io
import os
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2 as cv
import numpy as np

from .project import SCENARIOS, Scenario  # noqa: F401  (SCENARIOS: the built-in plan, kept for callers)

FRAME_W, FRAME_H = 640, 360

# Software WebGL keeps rendering identical across runs and machines without a GPU.
CHROMIUM_ARGS = [
    "--use-angle=swiftshader",
    "--use-gl=angle",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--force-color-profile=srgb",
    "--disable-features=CalculateNativeWinOcclusion",
]


class BuildError(RuntimeError):
    """The build never became playable (a script error before the test API was ready).

    Bisect treats such a commit as untestable and skips it, like `git bisect skip`.
    """


def git(repo: Path, *args: str) -> str:
    for attempt in range(3):
        result = subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=repo,
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        if result.stderr.strip():
            break  # git said what is wrong: retrying will not help
        time.sleep(1.0 + attempt)  # no message: git did not start (Windows out of desktop heap, a busy machine)
    raise RuntimeError(f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}")


class CheckoutCache:
    """Exports commits of the game repo into <root>/<sha[:10]>/ (one directory per commit)."""

    def __init__(self, repo: Path, root: Path, entry: str = "index.html"):
        self.repo = repo
        self.root = root
        self.entry = entry
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def resolve(self, rev: str) -> str:
        return git(self.repo, "rev-parse", "--verify", f"{rev}^{{commit}}")

    def ensure(self, rev: str) -> tuple[str, str]:
        """Return (full sha, directory name under root) for rev, exporting it on first use."""
        sha = self.resolve(rev)
        name = sha[:10]
        target = self.root / name
        with self._lock:
            if not (target / ".complete").exists():
                data = subprocess.run(["git", "-c", "core.longpaths=true", "archive", "--format=tar", sha],
                                      cwd=self.repo, capture_output=True, check=True).stdout
                tmp = self.root / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
                if tmp.exists():
                    _rmtree(tmp)
                with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                    tar.extractall(tmp, filter="data")
                (tmp / ".complete").write_text(sha)
                if target.exists():  # an export from before .complete markers, or a half-written one
                    _rmtree(target)
                try:
                    tmp.rename(target)
                except OSError:  # another process exported the same commit first
                    _rmtree(tmp)
                    if not (target / ".complete").exists():
                        raise
        return sha, name


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


class _QuietHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm"}

    def log_message(self, *args):  # keep test output readable
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class StaticServer:
    def __init__(self, root: Path):
        handler = functools.partial(_QuietHandler, directory=str(root))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@dataclass
class Capture:
    sha: str
    scenario: str
    camera: dict | None
    seed: int
    frames: list[np.ndarray]
    state: dict
    viewport: tuple[int, int] = (FRAME_W, FRAME_H)
    console_errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def errors(self) -> list[dict]:
        return list(self.state.get("errors", []))


def decode_data_url(url: str) -> np.ndarray:
    raw = base64.b64decode(url.split(",", 1)[1])
    img = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("could not decode frame from the canvas")
    return img


class GameDriver:
    """Headless Chromium session; not thread-safe (use it from one thread)."""

    def __init__(self, repo: Path, checkout_root: Path, *, headless: bool = True, entry: str = "index.html",
                 viewport: tuple[int, int] = (FRAME_W, FRAME_H)):
        from playwright.sync_api import sync_playwright

        self.entry = entry
        self.cache = CheckoutCache(repo, checkout_root, entry)
        self.server = StaticServer(checkout_root)
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless, args=CHROMIUM_ARGS)
        self.viewport = tuple(viewport)
        self.context = self.browser.new_context(viewport={"width": self.viewport[0], "height": self.viewport[1]},
                                                device_scale_factor=1)
        self.page = self.context.new_page()
        self._console: list[str] = []
        self.page.on("console", lambda m: self._console.append(m.text) if m.type == "error" else None)
        self.page.on("pageerror", lambda e: self._console.append(f"pageerror: {e}"))

    def close(self) -> None:
        for closer in (self.context.close, self.browser.close, self._pw.stop, self.server.close):
            try:
                closer()
            except Exception:
                pass

    def __enter__(self) -> "GameDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self, rev: str, camera: dict | None, seed: int, viewport: tuple[int, int] | None) -> str:
        sha, name = self.cache.ensure(rev)
        viewport = tuple(viewport or self.viewport)
        if viewport != tuple(self.page.viewport_size.values()):
            self.page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
        self._console = []
        self.page.goto(f"{self.server.base_url}/{name}/{self.entry}?test=1&seed={seed}"
                       f"&w={viewport[0]}&h={viewport[1]}", wait_until="load")
        self._wait_ready(sha)
        if camera:
            self.page.evaluate("(c) => window.__gh.setCamera(c)", camera)
        return sha

    def _wait_ready(self, sha: str, timeout: float = 30.0) -> None:
        """Wait for the test API; a script error with no API in sight means the build is broken."""
        t0 = time.perf_counter()
        error_seen = None
        while True:
            if self.page.evaluate("() => !!(window.__gh && window.__gh.ready === true)"):
                return
            errors = [e for e in self._console if e.startswith("pageerror")]
            if errors and error_seen is None:
                error_seen = time.perf_counter()
            if error_seen is not None and time.perf_counter() - error_seen > 1.5:
                if not self.page.evaluate("() => !!window.__gh"):
                    raise BuildError(f"build {sha[:10]} did not start: {errors[0][:300]}")
            if time.perf_counter() - t0 > timeout:
                raise BuildError(f"build {sha[:10]} did not become ready in {timeout:.0f}s"
                                 + (f": {errors[0][:300]}" if errors else ""))
            time.sleep(0.05)

    def _play(self, scenario: Scenario, frames: int, grab: bool) -> list[np.ndarray]:
        events: dict[int, list[tuple[str, str]]] = {}
        for index, kind, key in scenario.events:
            events.setdefault(index, []).append((kind, key))
        out = []
        for i in range(frames):
            for kind, key in events.get(i, ()):
                self.page.evaluate("([k, d]) => window.__gh.key(k, d)", [key, kind == "down"])
            cam = scenario.camera_at(i)
            if cam:
                self.page.evaluate("(c) => window.__gh.setCamera(c)", cam)
            if grab:
                url = self.page.evaluate("() => { window.__gh.advance(1); return window.__gh.grab(); }")
                out.append(decode_data_url(url))
            else:
                self.page.evaluate("() => window.__gh.advance(1)")
        return out

    def run(self, rev: str, scenario: Scenario, camera: dict | None = None, seed: int = 1,
            viewport: tuple[int, int] | None = None) -> Capture:
        t0 = time.perf_counter()
        sha = self._open(rev, camera or scenario.camera, seed, viewport)
        frames = self._play(scenario, scenario.frames, grab=True)
        state = self.page.evaluate("() => window.__gh.state()")
        return Capture(sha=sha, scenario=scenario.name, camera=camera, seed=seed, frames=frames, state=state,
                       viewport=tuple(viewport or self.viewport), console_errors=list(self._console),
                       seconds=time.perf_counter() - t0)

    def pick(self, rev: str, scenario: Scenario, camera: dict | None, frame_index: int,
             x: float, y: float, seed: int = 1, viewport: tuple[int, int] | None = None) -> dict:
        """Ray-cast through pixel (x, y) at a given frame of a scenario; returns the hit point."""
        self._open(rev, camera or scenario.camera, seed, viewport)
        self._play(scenario, max(0, frame_index) + 1, grab=False)
        hit = self.page.evaluate("([x, y]) => window.__gh.pick(x, y)", [x, y])
        hit["camera"] = self.page.evaluate("() => window.__gh.state().camera")
        return hit
