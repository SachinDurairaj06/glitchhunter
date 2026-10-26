"""Append-only JSONL trace: every decision, the evidence behind it, and what the tool returned.

This file is the audit log for a hunt and the proof that OpenCV output changes
the agent's next action: each `decision` names the observation it acted on.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np


def _default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class Trace:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.events: list[dict] = []
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def log(self, kind: str, **data) -> dict:
        event = {"t": round(time.time() - self._t0, 3), "kind": kind, **data}
        line = json.dumps(event, default=_default)
        with self._lock:
            self.events.append(json.loads(line))
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:
                pass
        return event

    @staticmethod
    def read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
