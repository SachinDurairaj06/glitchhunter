"""The perception -> decision -> action loop."""

from __future__ import annotations

import json
import secrets
import time
import traceback

from ..config import Settings
from ..lab import Lab
from ..obs import metrics
from .approvals import ApprovalBroker
from .planner import Action, RulePlanner, ToolResult
from .report import write_summary
from .state import Investigation
from .tools import TOOLS, HuntContext, ToolOutput
from .trace import Trace

MAX_CONSECUTIVE_ERRORS = 5


def make_planner(name: str, settings: Settings):
    if name == "bedrock":
        from .bedrock import BedrockPlanner

        return BedrockPlanner(settings)
    return RulePlanner()


class Hunt:
    def __init__(self, lab: Lab, settings: Settings, approvals: ApprovalBroker, planner=None,
                 hunt_id: str | None = None, label: str = ""):
        self.lab = lab
        self.settings = settings
        self.approvals = approvals
        self.id = hunt_id or time.strftime("h%m%d-%H%M%S-") + secrets.token_hex(2)
        self.label = label
        self.dir = settings.hunts_dir / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.dir / "trace.jsonl")
        self._planner_note = ""
        try:
            self.planner = planner or make_planner(settings.planner, settings)
        except Exception as exc:  # e.g. Bedrock client cannot be configured on this machine
            self.planner = RulePlanner()
            self._planner_note = f"{settings.planner} planner unavailable ({type(exc).__name__}: {exc}); using rule"
        self.inv = Investigation(head=lab.head(), baseline=lab.baseline(), project=lab.project)
        self.ctx = HuntContext(lab, self.inv, approvals, self.trace, self.id, self.dir)
        self.status = "created"
        self.outcome: dict = {}
        self.step = 0
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def call_tool(self, action: Action) -> ToolResult:
        spec = TOOLS.get(action.tool)
        if spec is None:
            return ToolResult(action.tool, action.args, False, {}, f"unknown tool {action.tool}",
                              error=f"unknown tool {action.tool!r}; available: {sorted(TOOLS)}")
        t0 = time.perf_counter()
        try:
            out: ToolOutput = spec.fn(self.ctx, dict(action.args or {}))
            result = ToolResult(action.tool, action.args, True, out.data, out.summary, out.images)
        except Exception as exc:  # tool errors go back to the planner instead of killing the hunt
            result = ToolResult(action.tool, action.args, False, {}, f"{action.tool} failed: {exc}",
                                error=f"{type(exc).__name__}: {exc}")
            self.trace.log("tool_error", step=self.step, tool=action.tool,
                           traceback=traceback.format_exc(limit=4))
        metrics.emit("tool_seconds", time.perf_counter() - t0, unit="Seconds", tool=action.tool)
        return result

    def run(self) -> dict:
        self.status = "running"
        t0 = time.time()
        self.trace.log("hunt_start", hunt=self.id, planner=getattr(self.planner, "name", "?"),
                       head=self.inv.head[:10], baseline=self.inv.baseline[:10],
                       approval_mode=self.approvals.mode, label=self.label, game=self.lab.project.title,
                       test_plan=self.lab.project.source, scenarios=list(self.lab.project.scenarios))
        if self._planner_note:
            self.trace.log("fallback", step=0, to="rule", reason=self._planner_note)
        from ..textdnn import models_available

        if not models_available():
            self.trace.log("warning", step=0,
                           error="OpenCV Zoo text models are missing, so hud_text_overflow is off; "
                                 "run `python -m glitchhunter fetch-models`")
        last: ToolResult | None = None
        errors_in_row = 0
        ended = "planner finished"
        while True:
            if self._stop:
                ended = "stopped by user"
                break
            if self.step >= self.settings.max_steps:
                ended = f"step budget ({self.settings.max_steps}) used up"
                break
            if time.time() - t0 > self.settings.max_seconds:
                ended = f"time budget ({self.settings.max_seconds:.0f}s) used up"
                break
            self.step += 1
            try:
                action = self.planner.next_action(self.inv, last)
            except Exception as exc:
                self.trace.log("planner_error", step=self.step, planner=getattr(self.planner, "name", "?"),
                               error=f"{type(exc).__name__}: {exc}")
                if not isinstance(self.planner, RulePlanner):
                    self.trace.log("fallback", step=self.step, to="rule",
                                   reason="LLM planner unavailable; continuing with the deterministic policy")
                    self.planner = RulePlanner()
                    continue
                ended = "planner crashed"
                break
            self.trace.log("decision", step=self.step, planner=action.planner, tool=action.tool,
                           args=action.args, reason=action.reason, evidence=action.evidence)
            if action.tool == "finish":
                ended = "planner finished: " + str(action.args.get("summary", ""))[:300]
                break
            result = self.call_tool(action)
            result.tool_use_id = getattr(action, "tool_use_id", None)
            self.trace.log("result", step=self.step, tool=result.tool, ok=result.ok, summary=result.summary,
                           error=result.error, images=result.images, data=_compact(result.data))
            last = result
            errors_in_row = 0 if result.ok else errors_in_row + 1
            if errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                ended = f"{errors_in_row} tool errors in a row"
                break
        seconds = time.time() - t0
        self.outcome = {
            "hunt": self.id, "game": self.lab.project.title, "planner": getattr(self.planner, "name", "?"),
            "ended": ended, "steps": self.step,
            "seconds": round(seconds, 1), "captures": self.lab.capture_count,
            "findings": [f.brief() for f in self.inv.ordered()],
        }
        self.trace.log("hunt_end", **{k: v for k, v in self.outcome.items() if k != "findings"})
        (self.dir / "findings.json").write_text(
            json.dumps([f.to_dict() for f in self.inv.ordered()], indent=2, default=str), encoding="utf-8")
        (self.dir / "outcome.json").write_text(json.dumps(self.outcome, indent=2, default=str), encoding="utf-8")
        write_summary(self.dir, self.id, self.inv, self.outcome)
        metrics.emit("hunt_seconds", seconds, unit="Seconds", planner=self.outcome["planner"])
        metrics.emit("findings_reported", sum(f.status == "reported" for f in self.inv.findings.values()))
        self.status = "done"
        if self.settings.s3_bucket:
            from ..store import upload_dir

            try:
                uri = upload_dir(self.dir, self.settings.s3_bucket, f"{self.settings.s3_prefix}/hunts/{self.id}")
                self.trace.log("uploaded", uri=uri)
                self.outcome["s3"] = uri
            except Exception as exc:
                self.trace.log("upload_error", error=str(exc))
        return self.outcome


def _compact(data: dict, limit: int = 1500) -> dict:
    text = json.dumps(data, default=str)
    if len(text) <= limit:
        return data
    out = {}
    for k, v in data.items():
        s = json.dumps(v, default=str)
        out[k] = v if len(s) < 400 else (s[:400] + "...")
    return out
