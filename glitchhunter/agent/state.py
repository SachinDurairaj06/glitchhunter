"""Investigation state: the findings the agent is working on and what it knows about each."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..vision import DETECTORS

SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2, "info": 3}

# Statuses where the agent has nothing left to do for the finding.
TERMINAL = {"dismissed", "reported", "rejected", "duplicate", "awaiting_approval"}


@dataclass
class Finding:
    id: str
    detector: str
    scenario: str
    commit: str
    region: list[int] | None
    camera: dict | None
    score: float
    confidence: float
    note: str
    run_id: str
    status: str = "candidate"
    object: str | None = None
    frames: list[int] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    also_seen: list[str] = field(default_factory=list)
    zooms: int = 0
    repro: dict | None = None
    bisect: dict | None = None
    bisect_attempts: int = 0
    culprit: dict | None = None
    fix: dict | None = None
    verification: dict | None = None
    approval: dict | None = None
    report: str | None = None
    duplicate_of: str | None = None
    viewport: list[int] | None = None  # screen size the finding was seen at (None = the project default)
    history: list[dict] = field(default_factory=list)

    @property
    def severity(self) -> str:
        return DETECTORS[self.detector].severity

    @property
    def localized(self) -> bool:
        return DETECTORS[self.detector].localized

    @property
    def area(self) -> int:
        return int(self.region[2] * self.region[3]) if self.region else 0

    @property
    def scenario_label(self) -> str:
        return self.scenario if not self.viewport else f"{self.scenario}@{self.viewport[0]}x{self.viewport[1]}"

    def set_status(self, status: str, why: str) -> None:
        self.history.append({"t": round(time.time(), 2), "from": self.status, "to": status, "why": why})
        self.status = status

    def brief(self) -> dict:
        """What a planner needs to know, without image paths or long diffs."""
        out = {
            "id": self.id, "detector": self.detector, "severity": self.severity, "status": self.status,
            "scenario": self.scenario_label, "object": self.object, "score": round(self.score, 3),
            "confidence": round(self.confidence, 3), "region": self.region, "area_px": self.area,
            "camera": "zoomed" if self.zooms else "default", "zooms": self.zooms, "note": self.note,
            "also_seen_in": self.also_seen,
        }
        if self.repro:
            out["reproduction"] = self.repro
        if self.bisect:
            out["bisect"] = {k: self.bisect[k] for k in ("culprit", "culprit_subject", "repeats", "tested",
                                                         "via_merge", "skipped", "ambiguous_with", "error")
                             if self.bisect.get(k)}
        if self.fix:
            out["fix"] = {"branch": self.fix["branch"], "sha": self.fix["sha"][:10], "strategy": self.fix["strategy"]}
        if self.verification:
            out["verification"] = {k: self.verification[k] for k in ("passed", "reason") if k in self.verification}
        if self.approval:
            out["approval"] = {k: self.approval.get(k) for k in ("choice", "by", "note")}
        if self.duplicate_of:
            out["duplicate_of"] = self.duplicate_of
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity
        return d


class Investigation:
    def __init__(self, head: str, baseline: str, project=None):
        self.head = head
        self.baseline = baseline
        self.project = project  # the game's test plan (glitchhunter.json)
        self.findings: dict[str, Finding] = {}
        self.surveyed: dict[str, dict] = {}  # commit -> {finding id: detector}
        self.presence_cache: dict[str, dict[str, set[str]]] = {}  # commit -> detector -> scenarios
        self.scenario_hashes: dict[str, str] = {}  # scenario -> frame hash of its survey capture
        self.soaked = False
        self.swept = False
        self.started = time.time()

    def new_id(self) -> str:
        return f"F{len(self.findings) + 1}"

    def ordered(self) -> list[Finding]:
        return sorted(self.findings.values(),
                      key=lambda f: (SEVERITY_RANK.get(f.severity, 9), -f.score, f.id))

    def open_findings(self) -> list[Finding]:
        return [f for f in self.ordered() if f.status not in TERMINAL]

    def get(self, finding_id: str) -> Finding:
        if finding_id not in self.findings:
            raise KeyError(f"unknown finding {finding_id!r}; known: {sorted(self.findings)}")
        return self.findings[finding_id]

    def snapshot(self) -> dict:
        return {
            "head": self.head[:10], "baseline": self.baseline[:10],
            "surveyed": [c[:10] for c in self.surveyed], "soaked": self.soaked, "swept": self.swept,
            "findings": [f.brief() for f in self.ordered()],
        }
