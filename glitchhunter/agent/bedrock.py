"""LLM planner: Claude on Amazon Bedrock picks the next tool call from the visual evidence.

Uses the Anthropic SDK's Bedrock Mantle client (Messages API on Bedrock, SigV4
auth from the normal AWS credential chain). The loop is manual because the
hunt loop owns tool execution, budgets, the human-approval gate and the
fallback to the rule planner; this class only turns state + last tool result
into the next tool call. History is append-only (thinking blocks stay valid).
"""

from __future__ import annotations

import base64
import json
import os
from collections import deque
from pathlib import Path

import cv2 as cv

from ..config import Settings
from ..vision import DETECTORS
from .planner import Action, ToolResult
from .state import Investigation
from .tools import TOOLS

DEFAULT_MODEL = "anthropic.claude-opus-5-5"
DEFAULT_FALLBACK_MODEL = "anthropic.claude-opus-4-8"  # used only if the main model declines a request
MAX_IMAGES_PER_RESULT = 2
MAX_IMAGE_SIDE = 800

SYSTEM_PROMPT = """You are GlitchHunter, a QA agent for a game kept in a git repository. \
You find visual bugs by capturing frames of the game's test scenarios and reading them with OpenCV 5 \
detectors, then you act on what the frames show: move the camera closer, replay, check other screen \
sizes, bisect commits, propose and verify fixes, and ask a human before anything is filed.

How to read the evidence:
- Each finding has a detector, a score (strength of the visual evidence, 0-1), a confidence (how far \
to trust it, 0-1), a screen region, and the object the game reports under that region.
- Low confidence on a small world-space region usually means the camera is too far away: zoom_in \
before deciding anything. Screen-space evidence (frozen or black frames, flashes, HUD text, color \
shifts, stutter) cannot be zoomed, and neither can scenarios that move the camera themselves.
- Detector families: classic single-frame checks; motion checks built on dense optical flow (an object \
vibrating, uneven motion, a one-frame pop, a character sliding without animating); comparisons with the \
golden build (a global tone change, shadow acne, lost texture detail, an object missing at the end); \
lighting checks against the golden build (surface relief shaded the wrong way by an inverted normal map, \
a shadow that is gone); pixel checks (NaN blocks, UI cut off by the screen edge).
- If the game supports several screen sizes, sweep_viewports checks the UI layout on each of them.
- replay shows whether a glitch reproduces and whether the evidence is identical on every run. Choose \
bisect repeats from that: 1 for deterministic glitches, more for flaky ones, enough that a bad commit \
is unlikely to look good by chance.
- bisect searches between the approved baseline build and the build under test, with the detector \
as the test. It skips commits that do not build and, when the first bad mainline commit is a merge, \
searches inside the merged branch. Two findings with the same culprit are one root cause. \
propose_fix reverts the culprit by default; a small patch is also possible.
- Always verify_fix before asking anyone. If a flaky glitch is still visible with the fix, the culprit \
may be wrong: bisect again with more repeats.
- request_human is required before file_report. If a human rejects, move on. Dismiss findings that do \
not reproduce.

Work through every finding until each one is reported, rejected, dismissed, a duplicate, or waiting \
for a human, then call finish with a short summary. Tool results include evidence images; look at them \
when the numbers are borderline."""


def _image_block(path: str) -> dict | None:
    img = cv.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".png", img)
    if not ok:
        return None
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(buf.tobytes()).decode("ascii")}}


def tool_definitions() -> list[dict]:
    return [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in TOOLS.values()]


def kickoff_text(inv: Investigation) -> str:
    detectors = "\n".join(f"- {name} ({spec.severity}{', zoomable' if spec.localized else ''}): {spec.description}"
                          for name, spec in DETECTORS.items() if spec.severity != "info")
    project = inv.project
    if project is None:
        from ..project import DEFAULT_PROJECT as project
    scenarios = "\n".join(f"- {s.name}: {s.description}" + ("" if s.zoomable else " (camera cannot be zoomed)")
                          for s in project.scenarios.values())
    sizes = (f"\n\nThe game also supports these screen sizes: {', '.join(f'{w}x{h}' for w, h in project.viewports)} "
             f"(default {project.viewport[0]}x{project.viewport[1]})." if project.viewports else "")
    return (f"Hunt for visual bugs in {project.title} at {project.branch} ({inv.head[:10]}). The approved baseline "
            f"is {project.baseline} ({inv.baseline[:10]}).\n\nScenarios (from the game's glitchhunter.json):\n"
            f"{scenarios}{sizes}\n\nOpenCV 5 detectors:\n{detectors}\n\nStart with a survey.")


class PlannerError(RuntimeError):
    pass


class BedrockPlanner:
    name = "bedrock"

    def __init__(self, settings: Settings, client=None):
        from anthropic import AnthropicBedrockMantle, BetaFallbackState, BetaRefusalFallbackMiddleware

        self.model = settings.bedrock_model_id or DEFAULT_MODEL
        self.effort = os.environ.get("GH_BEDROCK_EFFORT", "high")
        fallback = os.environ.get("GH_BEDROCK_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        self.client = client or AnthropicBedrockMantle(
            aws_region=settings.aws_region,
            middleware=[BetaRefusalFallbackMiddleware([{"model": fallback}])],
        )
        self.fallback_state = BetaFallbackState()  # one per conversation (one hunt)
        self.tools = tool_definitions()
        self.messages: list[dict] = []
        self.pending: deque = deque()
        self.results: list[dict] = []
        self.started = False
        self.nudges = 0
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        self._last_reason = ""

    # --- conversation ---------------------------------------------------------------------------
    def _tool_result_block(self, result: ToolResult) -> dict:
        payload = {"ok": result.ok, "summary": result.summary}
        if result.error:
            payload["error"] = result.error
        else:
            payload["data"] = result.data
        content: list[dict] = [{"type": "text", "text": json.dumps(payload, default=str)[:12000]}]
        for path in result.images[:MAX_IMAGES_PER_RESULT]:
            if path and Path(path).exists():
                block = _image_block(path)
                if block:
                    content.append(block)
        return {"type": "tool_result", "tool_use_id": result.tool_use_id, "content": content,
                "is_error": not result.ok}

    def _call(self):
        try:
            with self.fallback_state:
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=self.tools,
                    messages=self.messages,
                    thinking={"type": "adaptive", "display": "summarized"},
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},
                )
        except Exception as exc:  # auth, throttling, network: the hunt loop falls back to the rule planner
            raise PlannerError(f"Bedrock call failed: {type(exc).__name__}: {exc}") from exc
        self.usage["calls"] += 1
        usage = getattr(response, "usage", None)
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            self.usage[key] += int(getattr(usage, key, 0) or 0)
        if response.stop_reason == "refusal":
            raise PlannerError("the model declined the request (refusal)")
        return response

    @staticmethod
    def _reason(response) -> str:
        parts = []
        for block in response.content:
            if block.type == "thinking" and getattr(block, "thinking", ""):
                parts.append(block.thinking.strip())
            elif block.type == "text" and block.text.strip():
                parts.append(block.text.strip())
        return " ".join(parts)[:1200]

    def _action(self, block, reason: str, last: ToolResult | None) -> Action:
        evidence = {"model": self.model, "based_on": last.summary if last else "kickoff"}
        return Action(block.name, dict(block.input or {}), reason or f"model chose {block.name}",
                      evidence, planner="bedrock", tool_use_id=block.id)

    def next_action(self, inv: Investigation, last: ToolResult | None = None) -> Action:
        if last is not None and last.tool_use_id:
            self.results.append(self._tool_result_block(last))
        if self.pending:
            return self._action(self.pending.popleft(), self._last_reason, last)
        if not self.started:
            self.messages.append({"role": "user", "content": kickoff_text(inv)})
            self.started = True
        elif self.results:
            self.messages.append({"role": "user", "content": self.results})
            self.results = []
        response = self._call()
        self.messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        self._last_reason = self._reason(response)
        if uses:
            self.pending.extend(uses[1:])
            return self._action(uses[0], self._last_reason, last)
        if response.stop_reason == "max_tokens":
            raise PlannerError("response hit max_tokens without a tool call")
        still_open = inv.open_findings()
        if still_open and self.nudges < 2:
            self.nudges += 1
            ids = ", ".join(f"{f.id} ({f.status})" for f in still_open)
            self.messages.append({"role": "user", "content": f"These findings are still open: {ids}. "
                                  "Continue with tool calls, or call finish if you have decided to stop."})
            return self.next_action(inv, None)
        return Action("finish", {"summary": self._last_reason or "planner ended the hunt"},
                      self._last_reason or "model ended its turn", {"model": self.model}, planner="bedrock")
