"""Human approval gate.

The agent cannot file a bug report or keep a fix branch until a person (or an
explicit unattended policy) approves it. Modes:
  web           pending requests show on the dashboard; the hunt waits up to the timeout
  cli           ask on the terminal
  auto-approve  unattended policy (evaluation runs); recorded as such in the trace
  auto-reject   unattended policy that never lets the agent act
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class ApprovalRequest:
    id: str
    hunt_id: str
    finding_id: str
    question: str
    options: list[str]
    context: dict
    created: float = field(default_factory=time.time)
    choice: str | None = None
    note: str = ""
    by: str = ""
    decided: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ApprovalBroker:
    def __init__(self, mode: str = "auto-approve", timeout_s: float = 900.0):
        self.mode = mode
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._requests: dict[str, ApprovalRequest] = {}
        self._events: dict[str, threading.Event] = {}

    def request(self, hunt_id: str, finding_id: str, question: str, options: list[str],
                context: dict | None = None) -> ApprovalRequest:
        options = options or ["approve", "reject"]
        req = ApprovalRequest(secrets.token_hex(4), hunt_id, finding_id, question, options, context or {})
        with self._lock:
            self._requests[req.id] = req
        if self.mode == "auto-approve":
            return self._decide(req, options[0], "unattended run", "policy:auto-approve")
        if self.mode == "auto-reject":
            reject = next((o for o in options if "reject" in o.lower()), options[-1])
            return self._decide(req, reject, "unattended run", "policy:auto-reject")
        if self.mode == "cli":
            return self._ask_cli(req)
        event = threading.Event()
        with self._lock:
            self._events[req.id] = event
        event.wait(self.timeout_s)
        return req  # choice stays None when nobody answered in time

    def _ask_cli(self, req: ApprovalRequest) -> ApprovalRequest:
        print(f"\n[approval needed] {req.finding_id}: {req.question}")
        for i, opt in enumerate(req.options, 1):
            print(f"  {i}. {opt}")
        while True:
            raw = input("choice number (or text): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(req.options):
                return self._decide(req, req.options[int(raw) - 1], "", "cli")
            if raw in req.options:
                return self._decide(req, raw, "", "cli")

    def _decide(self, req: ApprovalRequest, choice: str, note: str, by: str) -> ApprovalRequest:
        with self._lock:
            req.choice, req.note, req.by, req.decided = choice, note, by, time.time()
            event = self._events.pop(req.id, None)
        if event:
            event.set()
        return req

    def decide(self, request_id: str, choice: str, note: str = "", by: str = "web") -> ApprovalRequest:
        with self._lock:
            req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"unknown approval request {request_id!r}")
        if req.choice is not None:
            raise ValueError("already decided")
        if choice not in req.options:
            raise ValueError(f"choice must be one of {req.options}")
        return self._decide(req, choice, note, by)

    def pending(self) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._requests.values() if r.choice is None]

    def all(self) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._requests.values()]
