"""Planners pick the next tool call from the investigation state.

RulePlanner is the deterministic policy (and the fallback when the LLM planner
is unavailable). Every action carries the reason and the visual evidence it
was based on, so the trace shows how OpenCV output steered the hunt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..lab import repeats_for
from .state import Finding, Investigation

ZOOM_BELOW = 0.6  # confidence under which localized evidence earns a closer look
DISMISS_BELOW = 0.3
STRONG_EVIDENCE = 0.8  # unreproduced but this confident: replay more before dismissing
MAX_REPLAYS = 12


@dataclass
class Action:
    tool: str
    args: dict
    reason: str
    evidence: dict = field(default_factory=dict)
    planner: str = "rule"
    tool_use_id: str | None = None  # set by the LLM planner so results go back to the right call


@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    data: dict
    summary: str
    images: list[str] = field(default_factory=list)
    error: str | None = None
    tool_use_id: str | None = None


def _ev(f: Finding, **extra) -> dict:
    return {"finding": f.id, "detector": f.detector, "score": round(f.score, 3),
            "confidence": round(f.confidence, 3), "area_px": f.area, **extra}


def bisect_repeats(f: Finding) -> int:
    """Captures per commit during bisect, from what replay showed.

    Identical frames on every replay mean the glitch is deterministic: one capture per
    commit is enough. Otherwise estimate the hit rate with a Beta(1,1) prior and pick
    enough repeats to miss a bad commit < 5% of the time.
    """
    r = f.repro or {}
    hits, runs = r.get("hits", 0), r.get("runs", 0)
    if r.get("deterministic") and runs and hits == runs:
        return 1
    estimate = (hits + 1) / (runs + 2) if runs else 0.5
    return repeats_for(estimate)


def _zoomable(inv: Investigation, f: Finding) -> bool:
    if inv.project is None:
        return True
    try:
        return inv.project.scenario(f.scenario).zoomable
    except KeyError:
        return False


def _baseline_name(inv: Investigation) -> str:
    return inv.project.baseline if inv.project is not None else "the baseline"


class RulePlanner:
    name = "rule"

    def __init__(self, soak_runs: int = 4, zoom: bool = True, sweep: bool = True):
        self.soak_runs = soak_runs
        self.zoom = zoom  # False = ablation without active perception
        self.sweep = sweep  # False = ablation without the screen-size sweep

    def next_action(self, inv: Investigation, last: ToolResult | None = None) -> Action:
        if not inv.surveyed:
            return Action("survey", {"commit": "main"},
                          "no evidence yet: capture every scenario at HEAD and run all OpenCV detectors")
        if self.sweep and not inv.swept and inv.project is not None and inv.project.viewports:
            sizes = ", ".join(f"{w}x{h}" for w, h in inv.project.viewports)
            return Action("sweep_viewports", {},
                          f"the game supports other screen sizes ({sizes}); UI anchored for one aspect ratio can run "
                          "off the screen on another, so replay the UI scenarios there and check the layout")
        for f in inv.ordered():
            action = self._for_finding(inv, f)
            if action:
                return action
        if not inv.soaked and self.soak_runs:
            return Action("soak", {"runs": self.soak_runs},
                          "all findings handled; replay every scenario once and soak the ones whose frames change "
                          "between runs, to catch intermittent glitches a single survey can miss")
        done = [f.id for f in inv.findings.values() if f.status == "reported"]
        waiting = [f.id for f in inv.findings.values() if f.status == "awaiting_approval"]
        return Action("finish", {"summary": f"reported {done}; waiting for a human on {waiting}"},
                      "no open findings left")

    def _for_finding(self, inv: Investigation, f: Finding) -> Action | None:
        s = f.status
        if s == "candidate":
            zoomable = self.zoom and f.localized and f.region and _zoomable(inv, f)
            if f.confidence < ZOOM_BELOW and zoomable and f.zooms < 2:
                factor = 3.0 if f.zooms == 0 else 2.0
                return Action("zoom_in", {"finding_id": f.id, "factor": factor},
                              f"{f.id} {f.detector}: confidence {f.confidence:.2f} < {ZOOM_BELOW} on a "
                              f"{f.area}px region; move the camera {factor:g}x closer before trusting it",
                              _ev(f))
            if f.confidence < DISMISS_BELOW:
                return Action("dismiss", {"finding_id": f.id,
                                          "reason": f"evidence stayed weak (confidence {f.confidence:.2f})"
                                                    + (f" after {f.zooms} zoom(s)" if f.zooms else "")},
                              f"{f.id}: confidence {f.confidence:.2f} < {DISMISS_BELOW}", _ev(f))
            n = 4 if f.detector in ("flash", "object_missing") else 3
            return Action("replay", {"finding_id": f.id, "n": n},
                          f"{f.id}: confidence {f.confidence:.2f} is high enough; replay {n}x to see if it "
                          "reproduces and whether frames are deterministic", _ev(f))
        if s in ("confirmed", "flaky"):
            repeats = bisect_repeats(f)
            why = ("identical evidence every replay -> deterministic, 1 capture per commit"
                   if repeats == 1 else
                   f"seen {f.repro['hits']}/{f.repro['runs']} with frames changing run to run -> "
                   f"{repeats} captures per commit keep the miss chance under 5%")
            return Action("bisect", {"finding_id": f.id, "repeats": repeats},
                          f"{f.id} reproduces; visual bisect from {_baseline_name(inv)}: {why}",
                          _ev(f, reproduction=f.repro))
        if s == "not_reproduced":
            runs = f.repro["runs"]
            if f.confidence >= STRONG_EVIDENCE and runs < MAX_REPLAYS:
                return Action("replay", {"finding_id": f.id, "n": 8},
                              f"{f.id}: the original evidence was strong (confidence {f.confidence:.2f}) but 0/{runs} "
                              "replays showed it; an intermittent glitch is likely, so replay 8 more times before "
                              "dismissing", _ev(f, reproduction=f.repro))
            return Action("dismiss", {"finding_id": f.id, "reason": f"did not reproduce in {runs} replays"},
                          f"{f.id}: 0/{runs} replays showed it", _ev(f, reproduction=f.repro))
        if s == "bisected":
            note = ""
            if (f.bisect or {}).get("via_merge"):
                note = f" (found inside merge {f.bisect['via_merge']})"
            return Action("propose_fix", {"finding_id": f.id, "strategy": "revert"},
                          f"{f.id}: first bad commit {f.culprit['sha'][:10]} '{f.culprit['subject']}'{note}; "
                          "revert it on a fix branch", _ev(f, culprit=f.culprit["sha"][:10]))
        if s == "fix_proposed":
            return Action("verify_fix", {"finding_id": f.id},
                          f"{f.id}: check the fix branch visually before asking anyone", _ev(f))
        if s == "fix_failed":
            flaky = not (f.repro or {}).get("deterministic", True)
            prev = (f.bisect or {}).get("repeats", 1)
            still_visible = (f.verification or {}).get("still_visible", 0) > 0
            if flaky and still_visible and f.bisect_attempts < 2 and prev < 8:
                repeats = min(8, max(prev * 2, prev + 2))
                return Action("bisect", {"finding_id": f.id, "repeats": repeats},
                              f"{f.id}: the fix did not remove a flaky glitch, so the culprit is probably wrong; "
                              f"bisect again with {repeats} captures per commit (was {prev})",
                              _ev(f, verification=f.verification))
            return Action("request_human", {
                "finding_id": f.id,
                "question": f"{f.id} ({f.detector}): the proposed fix did not pass ({f.verification['reason']}). "
                            "File the bug without a verified fix?",
                "options": ["file without fix", "reject"]},
                f"{f.id}: verification failed; a human decides", _ev(f, verification=f.verification))
        if s == "fix_verified":
            caveats = []
            b = f.bisect or {}
            if b.get("via_merge"):
                caveats.append(f"it arrived through merge {b['via_merge']} '{b.get('merge_subject')}'")
            if b.get("ambiguous_with"):
                caveats.append(f"commits {b['ambiguous_with']} do not build, so they could not be ruled out")
            return Action("request_human", {
                "finding_id": f.id,
                "question": f"{f.id} ({f.detector} on {f.object or f.scenario_label}): culprit "
                            f"{f.culprit['sha'][:10]} '{f.culprit['subject']}'"
                            + (f" ({'; '.join(caveats)})" if caveats else "")
                            + ". The revert removes the glitch with no new glitches. File the bug and keep the fix "
                              "branch?",
                "options": ["approve", "reject"]},
                f"{f.id}: fix verified visually; filing needs human approval", _ev(f, verification=f.verification))
        if s == "bisect_failed":
            return Action("request_human", {
                "finding_id": f.id,
                "question": f"{f.id} ({f.detector}): bisect could not isolate a commit "
                            f"({(f.bisect or {}).get('error')}). File it without a culprit?",
                "options": ["file without culprit", "reject"]},
                f"{f.id}: bisect failed; a human decides", _ev(f))
        if s == "approved":
            return Action("file_report", {"finding_id": f.id}, f"{f.id}: approved by {f.approval.get('by')}", _ev(f))
        return None
