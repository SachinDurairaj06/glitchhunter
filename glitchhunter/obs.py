"""Observability: CloudWatch Embedded Metric Format (EMF) lines plus in-process aggregates.

Set GH_EMF=stdout in the container: the awslogs log driver ships stdout to
CloudWatch Logs, and CloudWatch turns EMF lines into metrics without any API
calls. GH_EMF=file:<path> appends to a file instead; unset keeps metrics in
memory only (they are still shown on /api/metrics).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time


class Metrics:
    def __init__(self, namespace: str | None = None):
        self.namespace = namespace or os.environ.get("GH_METRICS_NAMESPACE", "GlitchHunter")
        self.sink = os.environ.get("GH_EMF", "")
        self._lock = threading.Lock()
        self._agg: dict[str, dict] = {}

    def emit(self, name: str, value: float, unit: str = "Count", **dims) -> None:
        dims = {k: str(v) for k, v in dims.items()}
        with self._lock:
            agg = self._agg.setdefault(name, {"count": 0, "sum": 0.0, "max": float("-inf"), "unit": unit})
            agg["count"] += 1
            agg["sum"] += float(value)
            agg["max"] = max(agg["max"], float(value))
            agg["last"] = float(value)
        if not self.sink:
            return
        record = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": self.namespace,
                    "Dimensions": [["Service", *sorted(dims)]],
                    "Metrics": [{"Name": name, "Unit": unit}],
                }],
            },
            "Service": "glitchhunter",
            name: float(value),
            **dims,
        }
        line = json.dumps(record)
        with self._lock:
            if self.sink == "stdout":
                print(line, file=sys.stdout, flush=True)
            elif self.sink.startswith("file:"):
                with open(self.sink[5:], "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def snapshot(self) -> dict:
        with self._lock:
            return {k: {**v, "mean": v["sum"] / max(v["count"], 1)} for k, v in self._agg.items()}


metrics = Metrics()
