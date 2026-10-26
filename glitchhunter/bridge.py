"""Drives native game builds (Godot, Unity, Unreal, custom engines) over the bridge protocol.

The game ships a small bridge (bridges/godot, bridges/unity) that speaks the same commands as
the web test API: reset, set_camera, key, advance, grab, state, pick. GlitchHunter exports the
commit with git archive, optionally runs the project's build command, launches the game with
the launch command from glitchhunter.json, and talks to it over one TCP connection (one JSON
object per line). The game connects back to the port given on its command line.

    "driver": {
      "type": "bridge",
      "build": ["{godot}", "--headless", "--path", "{dir}", "--import"],
      "launch": ["{godot}", "--path", "{dir}", "--fixed-fps", "30", "--resolution", "{width}x{height}",
                 "--", "--gh-port", "{port}"],
      "ready_timeout": 60
    }

Placeholders: {dir} the exported commit, {port}, {width}, {height}, {seed}, and any tool path
from GH_TOOL_<NAME> environment variables or the built-in lookup ({godot}).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .capture import BuildError, Capture, CheckoutCache
from .project import Project, Scenario


def find_tool(name: str) -> str | None:
    """Path of an engine tool: GH_TOOL_<NAME>, then a local tools folder, then PATH."""
    env = os.environ.get(f"GH_TOOL_{name.upper()}")
    if env:
        return env
    if name == "godot":
        roots = [Path(os.environ.get("LOCALAPPDATA", "")) / "glitchhunter" / "tools" / "godot",
                 Path.home() / ".local" / "share" / "glitchhunter" / "tools" / "godot"]
        for root in roots:
            if root.is_dir():
                exes = sorted(root.glob("Godot_v*_win64.exe")) + sorted(root.glob("Godot_v*_linux*"))
                exes = [p for p in exes if "console" not in p.name]
                if exes:
                    return str(exes[-1])
        return shutil.which("godot") or shutil.which("godot4")
    return shutil.which(name)


def _expand(template: list[str], values: dict) -> list[str]:
    out = []
    for part in template:
        for key, value in values.items():
            part = part.replace("{" + key + "}", str(value))
        if "{" in part and "}" in part:
            name = part[part.index("{") + 1:part.index("}")]
            tool = find_tool(name)
            if tool is None:
                raise BuildError(f"tool '{name}' not found: set GH_TOOL_{name.upper()} to its path")
            part = part.replace("{" + name + "}", tool)
        out.append(part)
    return out


class _Session:
    """One running game process and its connection."""

    def __init__(self, proc: subprocess.Popen, conn: socket.socket, log_path: Path, key: tuple):
        self.proc = proc
        self.conn = conn
        self.file = conn.makefile("rwb")
        self.log_path = log_path
        self.key = key
        self.hello: dict = {}

    def call(self, cmd: str, timeout: float = 60.0, **args) -> dict:
        self.conn.settimeout(timeout)
        self.file.write((json.dumps({"cmd": cmd, **args}) + "\n").encode("utf-8"))
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise BuildError(f"game closed the connection during '{cmd}': {self.log_tail()}")
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise RuntimeError(f"bridge error on '{cmd}': {reply.get('error')}")
        return reply

    def log_tail(self, lines: int = 8) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return " | ".join(text[-lines:])
        except OSError:
            return ""

    def close(self) -> None:
        try:
            self.call("quit", timeout=5)
        except Exception:
            pass
        for closer in (self.file.close, self.conn.close):
            try:
                closer()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def launch(template: list[str], cwd: Path, values: dict, log_path: Path, ready_timeout: float, label: str,
           key: tuple = ()) -> _Session:
    """Start a game with a launch command template, wait for it to connect back, say hello."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    log = open(log_path, "wb")
    cmd = _expand(template, {**values, "port": port})
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + ready_timeout
    server.settimeout(0.5)
    try:
        while True:
            try:
                conn, _ = server.accept()
                break
            except socket.timeout:
                if proc.poll() is not None or time.time() > deadline:
                    proc.kill()
                    log.close()
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-400:]
                    raise BuildError(f"{label} did not connect to the bridge: {tail}")
    finally:
        server.close()
    log.close()
    session = _Session(proc, conn, log_path, key)
    session.hello = session.call("hello", timeout=30)
    return session


def decode_png(reply: dict) -> np.ndarray:
    img = cv.imdecode(np.frombuffer(base64.b64decode(reply["png"]), np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("bridge sent an undecodable frame")
    return img


class BridgeDriver:
    """Same interface as capture.GameDriver (run, pick, close), for native builds."""

    def __init__(self, repo: Path, checkout_root: Path, project: Project, viewport: tuple[int, int]):
        self.cache = CheckoutCache(repo, checkout_root, project.entry)
        self.root = checkout_root
        self.spec = dict(project.driver or {})
        self.viewport = tuple(viewport)
        self._session: _Session | None = None
        self._built: set[str] = set()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "BridgeDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _build(self, sha: str, folder: Path) -> None:
        if not self.spec.get("build") or sha in self._built or (folder / ".gh-built").exists():
            return
        cmd = _expand(self.spec["build"], {"dir": folder})
        result = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=float(
            self.spec.get("build_timeout", 600)), check=False)
        if result.returncode != 0:
            raise BuildError(f"build of {sha[:10]} failed: {(result.stdout + result.stderr)[-400:]}")
        (folder / ".gh-built").write_text(sha)
        self._built.add(sha)

    def _start(self, sha: str, folder: Path, viewport: tuple[int, int], seed: int) -> _Session:
        log_path = self.root / f".{sha[:10]}-{viewport[0]}x{viewport[1]}.log"
        return launch(self.spec["launch"], folder, {"dir": folder, "width": viewport[0], "height": viewport[1],
                                                    "seed": seed}, log_path, float(self.spec.get("ready_timeout", 60)),
                      f"build {sha[:10]}", (sha, viewport))

    def _open(self, rev: str, seed: int, viewport) -> tuple[str, _Session]:
        sha, name = self.cache.ensure(rev)
        folder = self.root / name
        viewport = tuple(viewport or self.viewport)
        if self._session is None or self._session.key != (sha, viewport) or self._session.proc.poll() is not None:
            self.close()
            self._build(sha, folder)
            self._session = self._start(sha, folder, viewport, seed)
            self._session.call("reset", seed=seed, timeout=60)
            self._check_log(sha)
        else:
            self._session.call("reset", seed=seed, timeout=60)
        return sha, self._session

    def _check_log(self, sha: str) -> None:
        """Engines often keep running when the game's own scripts fail to load: treat that as a broken build."""
        patterns = self.spec.get("error_patterns") or []
        if not patterns:
            return
        time.sleep(0.2)  # let the engine flush its log
        text = self._session.log_path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern in text:
                line = next((ln for ln in text.splitlines() if pattern in ln), pattern)
                self.close()
                raise BuildError(f"build {sha[:10]} reports a script error: {line.strip()[:300]}")

    def _play(self, s: _Session, scenario: Scenario, frames: int, camera: dict | None, grab: bool):
        if camera or scenario.camera:
            s.call("set_camera", **(camera or scenario.camera))
        events: dict[int, list] = {}
        for index, kind, key in scenario.events:
            events.setdefault(index, []).append((kind, key))
        out = []
        for i in range(frames):
            for kind, key in events.get(i, ()):
                s.call("key", code=key, down=kind == "down")
            cam = scenario.camera_at(i)
            if cam:
                s.call("set_camera", **cam)
            s.call("advance", frames=1)
            if grab:
                out.append(decode_png(s.call("grab")))
        return out

    def _retry(self, fn):
        """Run fn once more in a fresh game process if the game died under it (a driver hiccup, the
        machine out of GPU memory). A build that crashes again is reported as broken."""
        try:
            return fn()
        except (BuildError, OSError) as exc:
            if isinstance(exc, BuildError) and "closed the connection" not in str(exc):
                raise  # it never started or logged a script error: not a hiccup
            self.close()
            return fn()

    def run(self, rev: str, scenario: Scenario, camera: dict | None = None, seed: int = 1,
            viewport: tuple[int, int] | None = None) -> Capture:
        def once() -> Capture:
            t0 = time.perf_counter()
            sha, s = self._open(rev, seed, viewport)
            frames = self._play(s, scenario, scenario.frames, camera, grab=True)
            state = s.call("state")
            return Capture(sha=sha, scenario=scenario.name, camera=camera, seed=seed, frames=frames,
                           state={k: v for k, v in state.items() if k != "ok"},
                           viewport=tuple(viewport or self.viewport), seconds=time.perf_counter() - t0)

        return self._retry(once)

    def pick(self, rev: str, scenario: Scenario, camera: dict | None, frame_index: int, x: float, y: float,
             seed: int = 1, viewport: tuple[int, int] | None = None) -> dict:
        return self._retry(lambda: self._pick(rev, scenario, camera, frame_index, x, y, seed, viewport))

    def _pick(self, rev: str, scenario: Scenario, camera: dict | None, frame_index: int, x: float, y: float,
              seed: int, viewport: tuple[int, int] | None) -> dict:
        _, s = self._open(rev, seed, viewport)
        self._play(s, scenario, max(0, frame_index) + 1, camera, grab=False)
        hit = s.call("pick", x=x, y=y)
        hit.pop("ok", None)
        if "camera" not in hit:
            hit["camera"] = s.call("state").get("camera")
        return hit
