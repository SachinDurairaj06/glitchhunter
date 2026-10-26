"""HTTP API + dashboard.

- /api/tools/{name}: the low-level OpenCV 5 tools (same set as the MCP server)
- /api/hunts: start / watch / stop agent hunts; /api/hunts/{id}/trace streams the decision trace
- /api/approvals: the human-in-the-loop queue (approve or reject what the agent wants to file)
- /files/...: evidence images and reports from the work directory
Set GH_API_TOKEN to require `Authorization: Bearer <token>` on every POST.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import service
from ..agent.approvals import ApprovalBroker
from ..agent.loop import Hunt
from ..agent.tools import TOOLS
from ..agent.trace import Trace
from ..config import Settings
from ..cvinfo import opencv_info
from ..obs import metrics

STATIC = Path(__file__).parent / "static"


class HuntRequest(BaseModel):
    planner: str | None = None
    label: str = ""
    max_steps: int | None = None


class Decision(BaseModel):
    choice: str
    note: str = ""


class GameChoice(BaseModel):
    game: str


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="GlitchHunter", version="0.1.0")
    approvals = ApprovalBroker(settings.approval_mode, settings.approval_timeout_s)
    hunts: dict[str, Hunt] = {}
    hunt_lock = threading.Lock()
    token = os.environ.get("GH_API_TOKEN", "")

    def require_token(request: Request) -> None:
        if token and request.headers.get("authorization", "") != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    def lab():
        return service.get_lab(settings)

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "style.css"):  # cache-bust so a reload always gets the current assets
            version = int((STATIC / name).stat().st_mtime)
            html = html.replace(f"/static/{name}", f"/static/{name}?v={version}")
        return html

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/health")
    def health():
        from ..textdnn import models_available

        try:
            project = lab().project
            game = {"title": project.title, "baseline": project.baseline, "scenarios": list(project.scenarios),
                    "viewports": [list(v) for v in project.viewports], "name": settings.game,
                    "driver": (project.driver or {}).get("type", "web"), "repo": bool(settings.repo)}
        except Exception as exc:  # no game repo yet
            game = {"title": None, "error": str(exc)}
        return {"status": "ok", "opencv": opencv_info(), "planner": settings.planner,
                "text_models": models_available(), "game": game,
                "approval_mode": approvals.mode, "auth": bool(token),
                "work_dir": str(settings.work_dir), "s3_bucket": settings.s3_bucket}

    @app.get("/api/metrics")
    def get_metrics():
        return metrics.snapshot()

    @app.get("/api/detectors")
    def detectors():
        from ..vision import DETECTORS

        families: dict[str, list] = {}
        for name, spec in DETECTORS.items():
            if spec.severity == "info":
                continue
            families.setdefault(spec.family, []).append(
                {"name": name, "description": spec.description, "severity": spec.severity,
                 "needs_golden": spec.needs_golden, "localized": spec.localized})
        return {"families": families, "count": sum(len(v) for v in families.values())}

    @app.get("/api/games")
    def games():
        from ..games import GAME_NAMES, get_game

        rows = []
        for name in GAME_NAMES:
            spec = get_game(name)
            built = (settings.work_dir / ("fixture" if name == "crypt" else f"fixture-{name}")).exists()
            rows.append({"name": name, "title": spec.title, "engine": spec.engine, "commits": len(spec.history),
                         "bugs": len(spec.bugs), "built": built})
        return {"games": rows, "current": None if settings.repo else settings.game}

    @app.post("/api/game", dependencies=[Depends(require_token)])
    def switch_game(choice: GameChoice):
        from ..fixture import build_fixture
        from ..games import GAME_NAMES

        if choice.game not in GAME_NAMES:
            raise HTTPException(404, f"unknown game {choice.game}")
        with hunt_lock:
            if any(h.status == "running" for h in hunts.values()):
                raise HTTPException(409, "a hunt is running; stop it before switching games")
            settings.game, settings.repo = choice.game, None
            if not settings.fixture_dir.exists():
                build_fixture(settings.fixture_dir, settings.fixture_manifest, choice.game)
            service.reset_lab(settings)
        return {"game": choice.game}

    @app.get("/api/eval")
    def evaluation():
        """The newest evaluation result per game (eval/results, written by `glitchhunter eval`)."""
        from ..config import REPO_ROOT

        newest: dict[str, dict] = {}
        for path in sorted((REPO_ROOT / "eval" / "results").glob("eval-*.json")):
            data = _read_json(path) or {}
            game = data.get("game")
            if game:
                newest[game] = data  # sorted by time stamp: later files win
        return {"results": [newest[g] for g in sorted(newest)]}

    @app.get("/api/repo")
    def repo():
        l = lab()
        return {"head": l.head(), "baseline": l.baseline(), "commits": l.commits(40)}

    # --- low-level tools (same as MCP) --------------------------------------------------------
    @app.get("/api/tools")
    def list_tools():
        return {"tools": sorted(service.TOOL_FUNCS), "agent_tools": [
            {"name": t.name, "description": t.description, "input_schema": t.schema} for t in TOOLS.values()]}

    @app.post("/api/tools/{name}", dependencies=[Depends(require_token)])
    def call_tool(name: str, body: dict | None = None):
        fn = service.TOOL_FUNCS.get(name)
        if fn is None:
            raise HTTPException(404, f"unknown tool {name}")
        try:
            return {"ok": True, "result": fn(**(body or {}))}
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}")

    # --- hunts ---------------------------------------------------------------------------------
    @app.post("/api/hunts", dependencies=[Depends(require_token)])
    def start_hunt(req: HuntRequest):
        with hunt_lock:
            if any(h.status == "running" for h in hunts.values()):
                raise HTTPException(409, "a hunt is already running")
            s = Settings(**{**settings.__dict__})
            if req.planner:
                s.planner = req.planner
            if req.max_steps:
                s.max_steps = req.max_steps
            hunt = Hunt(lab(), s, approvals, label=req.label)
            hunts[hunt.id] = hunt
            threading.Thread(target=hunt.run, name=f"hunt-{hunt.id}", daemon=True).start()
        return {"hunt": hunt.id, "dir": str(hunt.dir)}

    @app.get("/api/hunts")
    def list_hunts():
        rows = []
        if settings.hunts_dir.exists():
            for d in sorted(settings.hunts_dir.iterdir(), reverse=True)[:30]:
                live = hunts.get(d.name)
                outcome = _read_json(d / "outcome.json")
                found = _read_json(d / "findings.json") or []
                found = found if isinstance(found, list) else list(found.values())
                rows.append({"hunt": d.name, "status": live.status if live else ("done" if outcome else "running elsewhere"),
                             "ended": (outcome or {}).get("ended"), "steps": (outcome or {}).get("steps"),
                             "seconds": (outcome or {}).get("seconds"), "game": (outcome or {}).get("game"),
                             "reported": sum(1 for f in found if f.get("status") == "reported")})
        return rows

    @app.get("/api/hunts/{hunt_id}")
    def get_hunt(hunt_id: str):
        live = hunts.get(hunt_id)
        if live:
            return {"hunt": hunt_id, "status": live.status, "step": live.step,
                    "planner": getattr(live.planner, "name", "?"),
                    "findings": [f.to_dict() for f in live.inv.ordered()], "outcome": live.outcome}
        d = settings.hunts_dir / hunt_id
        if not d.exists():
            raise HTTPException(404, "unknown hunt")
        outcome = _read_json(d / "outcome.json")
        findings = _read_json(d / "findings.json") or []
        for f in findings:  # reports keep their own image copies; capture folders get pruned
            report_dir = d / "reports" / f.get("id", "")
            for name in list((f.get("evidence") or {})):
                copy = report_dir / f"{name}.png"
                if copy.exists():
                    f["evidence"][name] = str(copy)
        # A hunt started by another process (CLI, eval) writes findings only when it ends.
        return {"hunt": hunt_id, "status": "done" if outcome else "running elsewhere",
                "findings": findings, "outcome": outcome}

    @app.get("/api/hunts/{hunt_id}/trace")
    def get_trace(hunt_id: str, since: int = 0):
        live = hunts.get(hunt_id)
        events = list(live.trace.events) if live else Trace.read(settings.hunts_dir / hunt_id / "trace.jsonl")
        return {"events": events[since:], "next": len(events)}

    @app.post("/api/hunts/{hunt_id}/stop", dependencies=[Depends(require_token)])
    def stop_hunt(hunt_id: str):
        live = hunts.get(hunt_id)
        if not live:
            raise HTTPException(404, "hunt is not running")
        live.stop()
        return {"stopping": hunt_id}

    # --- human approvals ------------------------------------------------------------------------
    @app.get("/api/approvals")
    def pending_approvals():
        return {"pending": approvals.pending(), "mode": approvals.mode}

    @app.post("/api/approvals/{request_id}", dependencies=[Depends(require_token)])
    def decide(request_id: str, decision: Decision):
        try:
            req = approvals.decide(request_id, decision.choice, decision.note, by="web")
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return req.to_dict()

    # --- play any build in the browser ----------------------------------------------------------
    @app.get("/game/{rev}/{path:path}")
    def game(rev: str, path: str = ""):
        from ..capture import CheckoutCache

        cache = CheckoutCache(settings.fixture_dir, settings.checkout_dir)
        try:
            _, name = cache.ensure(rev)
        except (RuntimeError, OSError):
            raise HTTPException(404, f"unknown revision {rev}")
        root = (settings.checkout_dir / name).resolve()
        target = (root / (path or "index.html")).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    # --- files ----------------------------------------------------------------------------------
    @app.get("/files/{path:path}")
    def files(path: str):
        root = settings.work_dir.resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    @app.get("/api/file-url")
    def file_url(path: str):
        """Turn an absolute evidence path from a finding into a /files URL."""
        root = settings.work_dir.resolve()
        p = Path(path).resolve()
        if root not in p.parents:
            raise HTTPException(400, "path is outside the work directory")
        return {"url": "/files/" + p.relative_to(root).as_posix()}

    return app


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
