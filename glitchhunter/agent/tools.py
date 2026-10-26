"""The agent's tools. Each one wraps OpenCV 5 perception or a game/git action.

Planners (rule-based or Bedrock) choose these by name with JSON arguments; the
same specs are published over MCP and the HTTP API. Guardrails live here, not in
the planner: e.g. file_report refuses unless a human approved the finding.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import cv2 as cv

from .. import evidence as ev
from ..lab import BuildError, Lab, Run, viewport_label, zoom_camera
from ..vision import DETECTORS, Detection, iou
from .approvals import ApprovalBroker
from .report import write_report
from .state import Finding, Investigation
from .trace import Trace

# Re-checked on the whole frame: screen-wide evidence, or UI text that can run past any region box.
GLOBAL_DETECTORS = {"frozen_frame", "black_frame", "flash", "hud_text_overflow", "color_shift", "stutter",
                    "ui_offscreen"}
# Layout depends on the screen size: these run in the viewport sweep.
LAYOUT_DETECTORS = ["ui_offscreen", "hud_text_overflow"]


@dataclass
class ToolOutput:
    data: dict
    summary: str
    images: list[str] = field(default_factory=list)


@dataclass
class HuntContext:
    lab: Lab
    inv: Investigation
    approvals: ApprovalBroker
    trace: Trace
    hunt_id: str
    hunt_dir: Path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict
    fn: Callable[[HuntContext, dict], ToolOutput]


def _fid_schema(extra: dict | None = None, required: list[str] | None = None) -> dict:
    props = {"finding_id": {"type": "string", "description": "Finding id such as F2."}}
    props.update(extra or {})
    return {"type": "object", "properties": props, "required": ["finding_id", *(required or [])]}


# --- helpers ---------------------------------------------------------------------------------------

def _center(region) -> tuple[float, float]:
    x, y, w, h = region
    return x + w / 2.0, y + h / 2.0


def _viewport(f: Finding):
    return tuple(f.viewport) if f.viewport else None


def _scenario_label(scenario: str, viewport) -> str:
    return scenario if not viewport else f"{scenario}@{viewport_label(viewport)}"


def _pick_object(lab: Lab, run: Run, det: Detection) -> dict:
    """Ask the game what is under the detection: majority over the points the detector marked on
    the glitched thing, else over a few points in its region. An object that went missing is
    asked about in the golden build, the only one where it is still there."""
    if not det.region:
        return {}
    if det.pick_golden:
        run = replace(run, sha=lab.resolve(lab.project.baseline))
    x, y, w, h = det.region
    frame = det.frames[len(det.frames) // 2] if det.frames else 0
    points = list(det.points) or [(x + w * fx, y + h * fy, frame)
                                  for fx, fy in ((0.5, 0.5), (0.3, 0.4), (0.7, 0.4), (0.4, 0.7), (0.6, 0.7))]
    names, best = [], {}
    for px, py, t in points:
        hit = lab.pick(run, px, py, t)
        if hit.get("hit"):
            names.append(hit["object"])
            best.setdefault(hit["object"], hit)
    if not names:
        return {}
    name = Counter(names).most_common(1)[0][0]
    return best[name]


def _detect_on(ctx: HuntContext, run: Run, f: Finding, tag: str) -> list[tuple[Detection, dict]]:
    region = None if f.detector in GLOBAL_DETECTORS or not f.region else tuple(f.region)
    return ctx.lab.inspect(run, [f.detector], region, tag=tag)


def _frames_hash(run: Run) -> str:
    h = hashlib.sha1()
    for frame in run.frames:
        h.update(frame.tobytes())
    return h.hexdigest()


def _object_name(hit: dict) -> str | None:
    """Name of what the ray hit; coplanar surfaces are joined (e.g. 'floor+rune_decal')."""
    if not hit or not hit.get("hit"):
        return None
    coplanar = sorted(set(hit.get("coplanar") or []))
    return "+".join(coplanar) if len(coplanar) > 1 else hit.get("object")


def _register(ctx: HuntContext, det: Detection, paths: dict, run: Run, obj: dict) -> tuple[Finding, bool]:
    """Add a detection as a finding, or merge it into the finding for the same detector + object."""
    inv = ctx.inv
    spec = DETECTORS[det.detector]
    obj_name = _object_name(obj)
    viewport = list(run.viewport) if run.viewport else None
    label = _scenario_label(run.scenario, run.viewport)
    for f in inv.findings.values():
        if f.detector != det.detector or f.status in ("dismissed", "duplicate"):
            continue
        same = (not spec.localized) or (obj_name and obj_name == f.object) or (
            f.scenario == run.scenario and f.viewport == viewport and f.region and det.region
            and iou(f.region, det.region) > 0.2)
        if same:
            if label != f.scenario_label and label not in f.also_seen:
                f.also_seen.append(label)
            if f.zooms == 0 and det.score * (0.5 + det.confidence) > f.score * (0.5 + f.confidence):
                f.region = list(det.region) if det.region else None
                f.score, f.confidence, f.note = det.score, det.confidence, det.note
                f.frames, f.metrics, f.evidence = det.frames, det.to_dict()["metrics"], paths
                if label != f.scenario_label:
                    f.also_seen = [s for s in f.also_seen if s != label] + [f.scenario_label]
                    f.scenario, f.viewport = run.scenario, viewport
                f.run_id, f.camera = run.id, run.camera
            return f, False
    f = Finding(
        id=inv.new_id(), detector=det.detector, scenario=run.scenario, commit=run.sha,
        region=list(det.region) if det.region else None, camera=run.camera, score=det.score,
        confidence=det.confidence, note=det.note, run_id=run.id, object=obj_name, frames=det.frames,
        metrics=det.to_dict()["metrics"], evidence=paths, viewport=viewport,
    )
    if obj.get("coplanar") and len(obj["coplanar"]) > 1:
        f.metrics["coplanar_surfaces"] = obj["coplanar"]
    f.history.append({"t": 0, "from": None, "to": "candidate", "why": det.note})
    inv.findings[f.id] = f
    return f, True


def _best_image(paths: dict) -> list[str]:
    for key in ("closeup", "heatmap", "golden_vs_now", "annotated", "diff", "before_after", "frame", "flash_frame",
                "motion_plot", "tone_curve"):
        if key in paths:
            return [paths[key]]
    return list(paths.values())[:1]


def _inspect_and_register(ctx: HuntContext, run: Run, tag: str, detectors=None) -> tuple[list[str], list[str]]:
    created, merged = [], []
    for det, paths in ctx.lab.inspect(run, detectors, tag=tag):
        obj = _pick_object(ctx.lab, run, det) if (det.region and DETECTORS[det.detector].localized) else {}
        f, is_new = _register(ctx, det, paths, run, obj)
        (created if is_new else merged).append(f.id)
    return created, merged


# --- tools -----------------------------------------------------------------------------------------

def tool_survey(ctx: HuntContext, args: dict) -> ToolOutput:
    lab, inv = ctx.lab, ctx.inv
    sha = lab.resolve(args.get("commit") or lab.project.branch)
    created, merged, game_log = [], [], []
    runs = lab.capture_many([{"rev": sha, "scenario": s, "purpose": "survey"} for s in lab.project.scenarios])
    for scenario, run in zip(lab.project.scenarios, runs):
        if isinstance(run, Exception):
            raise run
        inv.scenario_hashes[scenario] = _frames_hash(run)
        for err in run.errors:
            item = {"scenario": scenario, **err}
            if item not in game_log:
                game_log.append(item)
        c, m = _inspect_and_register(ctx, run, "survey-")
        created += c
        merged += m
    inv.surveyed[sha] = {f.id: f.detector for f in inv.findings.values()}
    found = [inv.findings[i].brief() for i in dict.fromkeys(created + merged)]
    summary = (f"surveyed {len(lab.project.scenarios)} scenarios at {sha[:10]}: {len(created)} new finding(s) "
               + ", ".join(f"{f['id']}={f['detector']}(conf {f['confidence']:.2f})" for f in found))
    images = [p for i in created[:4] for p in _best_image(inv.findings[i].evidence)]
    return ToolOutput({"commit": sha[:10], "scenarios": list(lab.project.scenarios), "new_findings": created,
                       "findings": found, "game_log": game_log[:10]}, summary, images)


def tool_sweep_viewports(ctx: HuntContext, args: dict) -> ToolOutput:
    """Replay UI-heavy scenarios at the other screen sizes the game supports and check the layout."""
    lab, inv = ctx.lab, ctx.inv
    sha = lab.resolve(args.get("commit") or lab.project.branch)
    viewports = [tuple(v) for v in (args.get("viewports") or lab.project.viewports)]
    scenarios = args.get("scenarios") or list(lab.project.sweep_scenarios)
    created, checked = [], []
    for vp in viewports:
        for scenario in scenarios:
            run = lab.capture(sha, scenario, purpose="sweep", viewport=vp)
            checked.append(_scenario_label(scenario, vp))
            c, _ = _inspect_and_register(ctx, run, f"sweep{viewport_label(vp)}-", LAYOUT_DETECTORS)
            created += c
    inv.swept = True
    summary = (f"checked the layout at {', '.join(viewport_label(v) for v in viewports)} ({len(checked)} captures): "
               f"{len(created)} new finding(s) " + ", ".join(f"{i}={inv.findings[i].detector}" for i in created))
    return ToolOutput({"commit": sha[:10], "checked": checked, "new_findings": created,
                       "findings": [inv.findings[i].brief() for i in created]}, summary,
                      [p for i in created for p in _best_image(inv.findings[i].evidence)])


def tool_soak(ctx: HuntContext, args: dict) -> ToolOutput:
    """Repeat scenarios to catch intermittent glitches a single survey can miss.

    Without a scenario argument, every scenario is replayed once and compared with its survey
    capture; only scenarios whose frames differ (nondeterministic) are soaked further.
    """
    lab, inv = ctx.lab, ctx.inv
    runs = int(min(max(int(args.get("runs", 4)), 1), 10))
    sha = lab.resolve(args.get("commit") or lab.project.branch)
    created, nondeterministic, soaked = [], [], {}
    if args.get("scenario"):
        targets = {args["scenario"]: runs}
    else:
        targets = {}
        for scenario in lab.project.scenarios:
            run = lab.capture(sha, scenario, keep=False, purpose="soak-probe")
            c, _ = _inspect_and_register(ctx, run, "soak-probe-")
            created += c
            if inv.scenario_hashes.get(scenario) not in (None, _frames_hash(run)):
                nondeterministic.append(scenario)
                targets[scenario] = runs - 1
    for scenario, n in targets.items():
        soaked[scenario] = n
        for i in range(n):
            run = lab.capture(sha, scenario, keep=False, purpose="soak")
            c, _ = _inspect_and_register(ctx, run, f"soak{i}-")
            created += c
    inv.soaked = True
    detail = (f"nondeterministic scenarios {nondeterministic}" if not args.get("scenario")
              else f"scenario {args['scenario']}")
    summary = (f"soak at {sha[:10]}: {detail}; extra runs {soaked} -> {len(created)} new finding(s) {created}")
    return ToolOutput({"soaked": soaked, "nondeterministic": nondeterministic, "new_findings": created,
                       "findings": [inv.findings[i].brief() for i in created]}, summary,
                      [p for i in created for p in _best_image(inv.findings[i].evidence)])


def tool_zoom_in(ctx: HuntContext, args: dict) -> ToolOutput:
    lab, f = ctx.lab, ctx.inv.get(args["finding_id"])
    factor = float(args.get("factor", 3.0))
    if not f.localized or not f.region:
        raise ValueError(f"{f.id} ({f.detector}) is screen-space evidence; zooming the camera cannot help")
    if not lab.scenario(f.scenario).zoomable:
        raise ValueError(f"scenario {f.scenario} moves the camera itself; cannot zoom")
    if f.zooms >= 3:
        raise ValueError(f"{f.id} was already zoomed {f.zooms} times")
    run = lab.get_run(f.run_id, need_frames=False)  # aiming needs the camera, not the pixels
    cx, cy = _center(f.region)
    frame = f.frames[len(f.frames) // 2] if f.frames else 0
    hit = lab.pick(run, cx, cy, frame)
    if not hit.get("hit"):
        raise ValueError("ray cast through the region hit nothing; cannot aim the camera")
    camera = zoom_camera(hit["point"], hit["camera"]["pos"], factor, hit["camera"].get("fov", 55))
    new_run = lab.capture(run.sha, f.scenario, camera, purpose="zoom", viewport=_viewport(f))
    results = _detect_on_full(ctx, new_run, f, tag=f"{f.id}-zoom{f.zooms + 1}-")
    before = {"score": round(f.score, 3), "confidence": round(f.confidence, 3), "area_px": f.area,
              "region": f.region}
    images_before = _best_image(f.evidence)
    f.zooms += 1
    f.camera, f.run_id = camera, new_run.id
    if _object_name(hit):
        f.object = _object_name(hit)
    if hit.get("coplanar") and len(hit["coplanar"]) > 1:
        f.metrics["coplanar_surfaces"] = hit["coplanar"]
    if results:
        det, paths = results
        f.region = list(det.region) if det.region else f.region
        f.score, f.confidence, f.note = det.score, det.confidence, det.note
        f.frames, f.evidence = det.frames, {**paths, **{f"before_{k}": v for k, v in f.evidence.items()
                                                        if not k.startswith("before_")}}
        f.metrics.update(det.to_dict()["metrics"])
        verdict = "evidence got stronger" if det.confidence > before["confidence"] else "evidence did not improve"
    else:
        f.confidence *= 0.5
        f.region = None
        verdict = "glitch not visible up close; confidence halved"
    after = {"score": round(f.score, 3), "confidence": round(f.confidence, 3), "area_px": f.area,
             "region": f.region}
    summary = (f"{f.id} zoom x{factor:g} on '{_object_name(hit)}': confidence {before['confidence']:.2f} -> "
               f"{after['confidence']:.2f}, area {before['area_px']} -> {after['area_px']} px ({verdict})")
    return ToolOutput({"finding_id": f.id, "aimed_at": hit.get("object"), "coplanar": hit.get("coplanar"),
                       "camera": camera, "before": before, "after": after, "verdict": verdict}, summary,
                      _best_image(f.evidence) + images_before)


def _detect_on_full(ctx: HuntContext, run: Run, f: Finding, tag: str):
    """After a zoom the region moved; take the detection nearest the image center (the aim point)."""
    results = ctx.lab.inspect(run, [f.detector], None, tag=tag)
    if not results:
        return None
    h, w = run.frames[0].shape[:2]

    def covers_center(det) -> bool:
        x, y, rw, rh = det.region
        return x - 0.05 * w <= w / 2 <= x + rw + 0.05 * w and y - 0.05 * h <= h / 2 <= y + rh + 0.05 * h

    def dist(item):
        det = item[0]
        if not det.region:
            return 0.0
        cx, cy = _center(det.region)
        return ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5

    aimed = [item for item in results if item[0].region and covers_center(item[0])]
    if aimed:  # the largest region around the aim point is the glitch we zoomed in on
        return max(aimed, key=lambda item: item[0].region[2] * item[0].region[3])
    best = min(results, key=dist)
    if best[0].region and dist(best) > 0.45 * w:
        return None
    return best


def tool_replay(ctx: HuntContext, args: dict) -> ToolOutput:
    lab, f = ctx.lab, ctx.inv.get(args["finding_id"])
    n = int(min(max(int(args.get("n", 3)), 2), 10))
    hits, scores, hashes = 0, [], []
    runs = lab.capture_many([{"rev": f.commit, "scenario": f.scenario, "camera": f.camera, "keep": False,
                              "purpose": "replay", "viewport": _viewport(f)} for _ in range(n)])
    for i, run in enumerate(runs):
        if isinstance(run, Exception):
            raise run
        hashes.append(_frames_hash(run))
        found = _detect_on(ctx, run, f, tag=f"{f.id}-replay{i}-")
        present = bool(found)
        hits += present
        scores.append(round(max((d.score for d, _ in found), default=0.0), 3))
    # Replays with the same camera add up, so a second, longer replay of an intermittent glitch
    # refines the hit-rate estimate instead of replacing it.
    camera = _camera_key(f.camera)
    prev = f.repro if (f.repro and f.repro.get("camera") == camera and args.get("accumulate", True)) else None
    same_frames = len(set(hashes)) == 1 and (prev is None or prev.get("same_frames", False))
    if prev:
        hits_total, runs_total = prev["hits"] + hits, prev["runs"] + n
        scores = prev["scores"] + scores
    else:
        hits_total, runs_total = hits, n
    # Deterministic: identical frames every run, or (when something else in the build is random)
    # the detector measured exactly the same evidence every time.
    deterministic = same_frames or (hits_total == runs_total and max(scores) - min(scores) < 0.02)
    f.repro = {"runs": runs_total, "hits": hits_total, "rate": round(hits_total / runs_total, 3),
               "deterministic": deterministic, "scores": scores, "same_frames": same_frames, "camera": camera}
    if hits_total == runs_total:
        f.set_status("confirmed", f"reproduced {hits_total}/{runs_total}"
                     + ("" if deterministic else " (frames differ run to run)"))
    elif hits_total:
        f.set_status("flaky", f"reproduced {hits_total}/{runs_total}; frames differ run to run")
    else:
        f.set_status("not_reproduced", f"0/{runs_total} replays showed it")
    summary = (f"{f.id} replayed {n}x: seen {hits}/{n}" + (f" ({hits_total}/{runs_total} in total)" if prev else "")
               + ", " + ("identical frames every run (deterministic)" if same_frames else
                         "frames differ between runs but the evidence is identical (deterministic)" if deterministic
                         else "frames and evidence differ between runs (flaky)"))
    return ToolOutput({"finding_id": f.id, **{k: v for k, v in f.repro.items() if k != "camera"},
                       "status": f.status}, summary)


def _camera_key(camera: dict | None) -> str:
    return json.dumps(camera, sort_keys=True) if camera else "default"


def _first_bad(commits: list[str], oracle, probe_many=None, width: int = 1) -> tuple[int, list[int]]:
    """Search for the first bad commit; the last one is known bad, the one before the list good.

    oracle(sha) returns True (bad), False (good) or None (untestable: skipped, like `git bisect
    skip`). With probe_many(shas) -> verdicts and width > 1, each round tests `width` evenly spaced
    commits at once (one per capture worker), so a search over n commits takes about
    log_(width+1)(n) rounds instead of log_2(n). Returns the index of the first bad testable
    commit and the indices of skipped commits that could also be the culprit.
    """
    lo, hi = -1, len(commits) - 1
    skipped: set[int] = set()
    while hi - lo > 1:
        candidates = [i for i in range(lo + 1, hi) if i not in skipped]
        if not candidates:
            break
        if probe_many is not None and width > 1 and len(candidates) > 2:
            k = min(width, len(candidates))
            picks = sorted({candidates[min(len(candidates) - 1, (j + 1) * len(candidates) // (k + 1))]
                            for j in range(k)})
            verdicts = probe_many([commits[i] for i in picks])
            for i, v in zip(picks, verdicts):
                if v is None:
                    skipped.add(i)
            bad = [i for i, v in zip(picks, verdicts) if v]
            if bad:
                hi = min(hi, bad[0])
            good = [i for i, v in zip(picks, verdicts) if v is False and i < hi]
            if good:
                lo = max(lo, max(good))
            continue
        mid = (lo + hi) // 2
        mid = min(candidates, key=lambda i: (abs(i - mid), i))
        verdict = oracle(commits[mid])
        if verdict is None:
            skipped.add(mid)
        elif verdict:
            hi = mid
        else:
            lo = mid
    return hi, sorted(i for i in skipped if lo < i < hi)


def tool_bisect(ctx: HuntContext, args: dict) -> ToolOutput:
    lab, f, inv = ctx.lab, ctx.inv.get(args["finding_id"]), ctx.inv
    repeats = int(min(max(int(args.get("repeats", 1)), 1), 8))
    good = lab.resolve(args.get("good") or lab.project.baseline)
    bad = lab.resolve(args.get("bad") or f.commit)
    f.bisect_attempts += 1
    tested: dict[str, dict] = {}

    def oracle(sha: str) -> bool | None:
        if sha in tested and tested[sha]["bad"] is not None:
            return tested[sha]["bad"]
        for r in range(repeats):
            try:
                run = lab.capture(sha, f.scenario, f.camera, keep=False, purpose="bisect", viewport=_viewport(f))
            except BuildError as exc:
                tested[sha] = {"bad": None, "runs": r, "score": 0.0, "error": str(exc)[:200]}
                ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="skip (build broken)", runs=r)
                return None
            found = _detect_on(ctx, run, f, tag=f"{f.id}-bisect-")
            if found:
                tested[sha] = {"bad": True, "runs": r + 1, "score": round(found[0][0].score, 3)}
                ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="bad", runs=r + 1)
                return True
        tested[sha] = {"bad": False, "runs": repeats, "score": 0.0}
        ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="good", runs=repeats)
        return False

    def probe_many(shas: list[str]) -> list[bool | None]:
        """Several commits at once on the capture workers, `repeats` rounds at most."""
        verdict: dict[str, bool | None] = {sha: tested[sha]["bad"] for sha in shas
                                           if sha in tested and tested[sha]["bad"] is not None}
        runs_done = {sha: 0 for sha in shas}
        for r in range(repeats):
            pending = [sha for sha in shas if sha not in verdict]
            if not pending:
                break
            results = lab.capture_many([{"rev": sha, "scenario": f.scenario, "camera": f.camera, "keep": False,
                                         "purpose": "bisect", "viewport": _viewport(f)} for sha in pending])
            for sha, run in zip(pending, results):
                if isinstance(run, BuildError):
                    verdict[sha] = None
                    tested[sha] = {"bad": None, "runs": r, "score": 0.0, "error": str(run)[:200]}
                    ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="skip (build broken)", runs=r)
                    continue
                if isinstance(run, Exception):
                    raise run
                runs_done[sha] = r + 1
                found = _detect_on(ctx, run, f, tag=f"{f.id}-bisect-")
                if found:
                    verdict[sha] = True
                    tested[sha] = {"bad": True, "runs": r + 1, "score": round(found[0][0].score, 3)}
                    ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="bad", runs=r + 1)
        for sha in shas:
            if sha not in verdict:
                verdict[sha] = False
                tested[sha] = {"bad": False, "runs": repeats, "score": 0.0}
                ctx.trace.log("bisect_step", finding=f.id, commit=sha[:10], verdict="good", runs=repeats)
        return [verdict[sha] for sha in shas]

    width = lab.worker.size if hasattr(lab.worker, "size") else 1

    def fail(error: str, why: str) -> ToolOutput:
        f.set_status("bisect_failed", why)
        f.bisect = {"good": good[:10], "bad": bad[:10], "repeats": repeats, "error": error,
                    "steps": _steps(lab, tested)}
        return ToolOutput({"finding_id": f.id, "error": error, "status": f.status}, f"{f.id} bisect aborted: {why}")

    verdict = oracle(good)
    if verdict:
        return fail("baseline is bad", f"the baseline {lab.project.baseline} already shows {f.detector}")
    if verdict is None:
        return fail("baseline does not build", f"the baseline {lab.project.baseline} does not start")
    replayed_bad = bad == f.commit and (f.repro or {}).get("hits", 0) > 0
    if replayed_bad:
        # The replays already saw the glitch on this build; an intermittent glitch could hide again.
        tested[bad] = {"bad": True, "runs": 0, "score": max((f.repro or {}).get("scores") or [0.0]),
                       "known_from": "replay"}
        ctx.trace.log("bisect_step", finding=f.id, commit=bad[:10], verdict="bad (seen in replay)", runs=0)
    elif not oracle(bad):
        return fail("bad end did not reproduce", f"could not reproduce at {bad[:10]} in {repeats} run(s)")
    commits = lab.commits_between(good, bad)
    idx, ambiguous = _first_bad(commits, oracle, probe_many, width)
    culprit = commits[idx]
    merge = None
    if len(lab.parents(culprit)) > 1:
        # The first bad mainline commit is a merge: find the commit on the merged branch that did it.
        side = lab.side_commits(culprit)
        ctx.trace.log("bisect_step", finding=f.id, commit=culprit[:10],
                      verdict=f"merge: searching the {len(side)} merged commit(s)", runs=0)
        if side and oracle(side[-1]):
            s_idx, s_amb = _first_bad(side, oracle, probe_many, width)
            merge, culprit = culprit, side[s_idx]
            ambiguous_shas = [side[i] for i in s_amb]
        else:
            merge = culprit  # each side is fine alone: the merge itself (or the combination) did it
            ambiguous_shas = []
    else:
        ambiguous_shas = [commits[i] for i in ambiguous]
    info = lab.show_commit(culprit)
    f.culprit = info
    skipped = [sha[:10] for sha, v in tested.items() if v["bad"] is None]
    f.bisect = {"good": good[:10], "bad": bad[:10], "repeats": repeats, "culprit": culprit[:10],
                "culprit_subject": info["subject"], "steps": _steps(lab, tested), "tested": len(tested),
                "range": len(commits), "captures": sum(v["runs"] for v in tested.values()),
                "skipped": skipped, "ambiguous_with": [s[:10] for s in ambiguous_shas],
                "via_merge": merge[:10] if merge and merge != culprit else None,
                "merge_subject": lab.subject(merge) if merge and merge != culprit else None}
    primary = next((o for o in inv.findings.values() if o.id != f.id and o.culprit
                    and o.culprit["sha"] == info["sha"] and o.status != "duplicate"), None)
    if primary:
        f.duplicate_of = primary.id
        label = f.scenario_label
        if label not in primary.also_seen and label != primary.scenario_label:
            primary.also_seen.append(label)
        same = "same detector" if primary.detector == f.detector else f"{primary.detector} vs {f.detector}"
        f.set_status("duplicate", f"same root cause as {primary.id} ({same}, culprit {culprit[:10]})")
    else:
        f.set_status("bisected", f"first bad commit {culprit[:10]} '{info['subject']}'")
    extra = []
    if f.bisect["via_merge"]:
        extra.append(f"found inside merge {merge[:10]} '{f.bisect['merge_subject']}'")
    if skipped:
        extra.append(f"skipped {len(skipped)} broken build(s) {skipped}")
    if ambiguous_shas:
        extra.append(f"could also be skipped {[s[:10] for s in ambiguous_shas]}")
    summary = (f"{f.id} bisect over {len(commits)} commits with {repeats} run(s)/commit: "
               f"culprit {culprit[:10]} '{info['subject']}' after testing {len(tested)} builds"
               + (f"; {'; '.join(extra)}" if extra else "")
               + (f" (duplicate of {primary.id})" if primary else ""))
    return ToolOutput({"finding_id": f.id, **f.bisect, "status": f.status, "diff": info["diff"][:2500]}, summary)


def _steps(lab: Lab, tested: dict) -> list[dict]:
    return [{"commit": sha[:10], **v, "subject": lab.subject(sha)} for sha, v in tested.items()]


def tool_show_commit(ctx: HuntContext, args: dict) -> ToolOutput:
    info = ctx.lab.show_commit(args["sha"])
    return ToolOutput(info, f"commit {info['sha'][:10]} '{info['subject']}' ({len(info['files'])} files)")


def tool_propose_fix(ctx: HuntContext, args: dict) -> ToolOutput:
    f = ctx.inv.get(args["finding_id"])
    if not f.culprit:
        raise ValueError(f"{f.id} has no culprit commit yet; bisect first")
    strategy = args.get("strategy", "revert")
    fix = ctx.lab.make_fix(f.culprit["sha"], f"{ctx.hunt_id}-{f.id}-{f.detector}", strategy, args.get("patch"))
    f.fix = fix
    f.set_status("fix_proposed", f"{strategy} of {f.culprit['sha'][:10]} on branch {fix['branch']}")
    return ToolOutput({"finding_id": f.id, "branch": fix["branch"], "sha": fix["sha"][:10], "strategy": strategy,
                       "diff": fix["diff"][:2500]},
                      f"{f.id} fix proposed: {strategy} -> {fix['branch']} @ {fix['sha'][:10]}")


def _presence_by_detector(ctx: HuntContext, sha: str, scenarios=None) -> dict[str, set[str]]:
    """Which detectors fire in which scenarios at a commit (cached per commit for full surveys).

    A full check covers every scenario at the default screen size plus the layout detectors at
    the project's other screen sizes.
    """
    cache = ctx.inv.presence_cache
    if scenarios is None and sha in cache:
        return cache[sha]
    lab = ctx.lab
    out: dict[str, set[str]] = {}
    names = list(scenarios or lab.project.scenarios)
    jobs = [({"rev": sha, "scenario": s, "keep": False, "purpose": "verify"}, s, None) for s in names]
    if scenarios is None:
        jobs += [({"rev": sha, "scenario": s, "keep": False, "purpose": "verify", "viewport": tuple(vp)},
                  _scenario_label(s, vp), LAYOUT_DETECTORS)
                 for vp in lab.project.viewports for s in lab.project.sweep_scenarios]
    runs = lab.capture_many([j[0] for j in jobs])
    for (_, label, detectors), run in zip(jobs, runs):
        if isinstance(run, Exception):
            raise run
        for det, _ in lab.inspect(run, detectors, tag="verify-"):
            out.setdefault(det.detector, set()).add(label)
    if scenarios is None:
        cache[sha] = out
    return out


def _seen_at(ctx: HuntContext, sha: str, detector: str, labels, runs: int) -> str | None:
    """Look for a detector at a commit (up to `runs` captures per scenario); register what it finds."""
    for label in labels:
        scenario, _, vp = label.partition("@")
        viewport = tuple(int(v) for v in vp.split("x")) if vp else None
        for _ in range(runs):
            run = ctx.lab.capture(sha, scenario, keep=False, purpose="verify-recheck", viewport=viewport)
            for det, paths in ctx.lab.inspect(run, [detector], tag="recheck-"):
                obj = _pick_object(ctx.lab, run, det) if (det.region and DETECTORS[detector].localized) else {}
                finding, _ = _register(ctx, det, paths, run, obj)
                return finding.id
    return None


def tool_verify_fix(ctx: HuntContext, args: dict) -> ToolOutput:
    lab, f = ctx.lab, ctx.inv.get(args["finding_id"])
    if not f.fix:
        raise ValueError(f"{f.id} has no proposed fix")
    fix_sha = f.fix["sha"]
    repeats = max(1, int((f.bisect or {}).get("repeats", 1)))
    still, last_run = 0, None
    for _ in range(repeats):
        last_run = lab.capture(fix_sha, f.scenario, f.camera, keep=False, purpose="verify", viewport=_viewport(f))
        if _detect_on(ctx, last_run, f, tag=f"{f.id}-verify-"):
            still += 1
    # regression check: anything the fix build shows that the head build does not?
    after = _presence_by_detector(ctx, fix_sha)
    before = _presence_by_detector(ctx, f.commit)
    new = {d: sorted(s - before.get(d, set())) for d, s in after.items() if s - before.get(d, set())}
    # Fixing one glitch can reveal another one it was hiding (a black map view hides a broken
    # texture). Glitches that are already open findings are "unmasked", not regressions.
    known = {o.detector for o in ctx.inv.findings.values() if o.id != f.id and o.status != "dismissed"}
    regressions, unmasked, preexisting = {}, {}, {}
    for det_name, labels in new.items():
        if det_name == f.detector:
            continue
        if det_name in known:
            unmasked[det_name] = labels
            continue
        seen_at_head = _seen_at(ctx, f.commit, det_name, labels, runs=3)
        if seen_at_head:
            # Intermittent at HEAD too: not caused by the fix, but new to this hunt, so investigate it.
            preexisting[det_name] = seen_at_head
        else:
            regressions[det_name] = labels
    passed = still == 0 and not regressions
    reason = ("glitch gone and no new glitches" if passed else
              f"glitch still visible in {still}/{repeats} run(s)" if still else f"new glitches: {regressions}")
    if passed and unmasked:
        reason += f"; it unmasks already-known glitches {sorted(unmasked)}"
    if preexisting:
        reason += f"; also found intermittent glitches that HEAD has too: {sorted(preexisting)} (new findings)"
    f.verification = {"passed": passed, "reason": reason, "runs": repeats, "still_visible": still,
                      "regressions": regressions, "unmasked_known": unmasked,
                      "new_findings": sorted(preexisting.values()), "fix_sha": fix_sha[:10]}
    image = None
    if last_run is not None:
        head_run = lab.capture(f.commit, f.scenario, f.camera, keep=False, purpose="verify-before",
                               viewport=_viewport(f))
        idx = min(f.frames[-1] if f.frames else len(head_run.frames) // 2, len(head_run.frames) - 1)
        region = f.region if f.detector not in GLOBAL_DETECTORS else None
        before_img = ev.box(head_run.frames[idx], region, ev.RED)
        after_img = ev.box(last_run.frames[idx], region, ev.GREEN)
        side = ev.side_by_side([before_img, after_img], [f"before ({f.commit[:8]})", f"with fix ({fix_sha[:8]})"])
        path = last_run.dir / f"{f.id}-before-after.png"
        cv.imwrite(str(path), side)
        f.evidence["before_after_fix"] = str(path)
        image = str(path)
    f.set_status("fix_verified" if passed else "fix_failed", reason)
    return ToolOutput({"finding_id": f.id, **f.verification, "status": f.status},
                      f"{f.id} verify fix {fix_sha[:10]}: {'PASS' if passed else 'FAIL'} ({reason})",
                      [image] if image else [])


def tool_request_human(ctx: HuntContext, args: dict) -> ToolOutput:
    f = ctx.inv.get(args["finding_id"])
    options = list(args.get("options") or ["approve", "reject"])
    question = args.get("question") or f"File a bug for {f.id} ({f.detector}) and keep the fix branch?"
    context = {"finding": f.brief(), "evidence": f.evidence, "culprit": (f.culprit or {}).get("subject"),
               "fix_branch": (f.fix or {}).get("branch"), "verification": f.verification}
    f.set_status("awaiting_approval", "waiting for a human decision")
    req = ctx.approvals.request(ctx.hunt_id, f.id, question, options, context)
    f.approval = {"request_id": req.id, "question": question, "options": options, "choice": req.choice,
                  "by": req.by, "note": req.note}
    if req.choice is None:
        summary = f"{f.id}: no human decision within the timeout; left pending"
    elif req.choice == options[0] or any(k in req.choice.lower() for k in ("approve", "file", "yes")):
        f.set_status("approved", f"approved by {req.by}")
        summary = f"{f.id}: '{req.choice}' by {req.by}"
    else:
        f.set_status("rejected", f"'{req.choice}' by {req.by}")
        if f.fix:
            try:
                ctx.lab.delete_branch(f.fix["branch"])
                f.fix["deleted"] = True
            except Exception:
                pass
        summary = f"{f.id}: rejected by {req.by}; fix branch removed"
    return ToolOutput({"finding_id": f.id, "choice": req.choice, "by": req.by, "status": f.status}, summary)


def tool_file_report(ctx: HuntContext, args: dict) -> ToolOutput:
    f = ctx.inv.get(args["finding_id"])
    if f.status != "approved":
        raise PermissionError(f"{f.id} is '{f.status}': a human must approve before a report is filed")
    path = write_report(ctx.hunt_dir, ctx.hunt_id, f, ctx.lab)
    f.report = str(path)
    f.set_status("reported", f"report written to {path.name}")
    return ToolOutput({"finding_id": f.id, "report": str(path)}, f"{f.id}: bug report filed ({path.name})")


def tool_dismiss(ctx: HuntContext, args: dict) -> ToolOutput:
    f = ctx.inv.get(args["finding_id"])
    reason = args.get("reason") or "dismissed by planner"
    f.set_status("dismissed", reason)
    return ToolOutput({"finding_id": f.id, "status": f.status}, f"{f.id} dismissed: {reason}")


def tool_list_findings(ctx: HuntContext, args: dict) -> ToolOutput:
    snap = ctx.inv.snapshot()
    return ToolOutput(snap, f"{len(snap['findings'])} finding(s)")


def tool_look(ctx: HuntContext, args: dict) -> ToolOutput:
    f = ctx.inv.get(args["finding_id"])
    names = [n for n in f.evidence if not n.startswith("before_")][:3]
    return ToolOutput({"finding_id": f.id, "images": names}, f"{f.id}: showing {names}",
                      [f.evidence[n] for n in names])


def tool_finish(ctx: HuntContext, args: dict) -> ToolOutput:
    return ToolOutput({"summary": args.get("summary", "")}, "finish")


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in (
        ToolSpec("survey", "Capture every test scenario of the game's test plan at a commit and run all OpenCV 5 "
                 "glitch detectors. Registers candidate findings with score, confidence, screen region and the "
                 "object under it.",
                 {"type": "object", "properties": {"commit": {"type": "string", "description": "Commit or ref; default the branch head."}}},
                 tool_survey),
        ToolSpec("sweep_viewports", "Replay the UI scenarios at the other screen sizes the game supports (e.g. 4:3, "
                 "21:9) and check the layout (UI cut off by the screen edge, text overflowing its panel).",
                 {"type": "object", "properties": {"commit": {"type": "string"},
                                                   "viewports": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}},
                 tool_sweep_viewports),
        ToolSpec("soak", "Repeat scenarios to catch intermittent glitches. Without `scenario`, replays each scenario "
                 "once, finds the nondeterministic ones (frames differ from the survey) and soaks only those.",
                 {"type": "object", "properties": {"scenario": {"type": "string"},
                                                   "runs": {"type": "integer", "minimum": 1, "maximum": 10}}},
                 tool_soak),
        ToolSpec("zoom_in", "Active perception: ray-cast through the finding's region, move the game camera "
                 "closer to that point, recapture and re-run the detector. Use when confidence is low and the "
                 "evidence is a small world-space region.",
                 _fid_schema({"factor": {"type": "number", "minimum": 1.5, "maximum": 6}}), tool_zoom_in),
        ToolSpec("replay", "Re-run the finding's scenario n times; reports how often the glitch shows and whether "
                 "frames are identical across runs (deterministic) or not (flaky).",
                 _fid_schema({"n": {"type": "integer", "minimum": 2, "maximum": 10}}), tool_replay),
        ToolSpec("bisect", "Visual git bisect: binary-search commits between good (default: the approved baseline) "
                 "and the finding's commit, using the OpenCV detector as the test. repeats = runs per commit "
                 "(raise it for flaky glitches). Skips commits that do not build; when the first bad commit is "
                 "a merge, it searches inside the merged branch.",
                 _fid_schema({"good": {"type": "string"}, "repeats": {"type": "integer", "minimum": 1, "maximum": 8}}),
                 tool_bisect),
        ToolSpec("show_commit", "Show a commit's subject and diff.",
                 {"type": "object", "properties": {"sha": {"type": "string"}}, "required": ["sha"]}, tool_show_commit),
        ToolSpec("propose_fix", "Create branch glitchhunter/<id> = main + fix. strategy 'revert' reverts the "
                 "culprit; 'patch' applies a unified diff (game files only). main is never modified.",
                 _fid_schema({"strategy": {"type": "string", "enum": ["revert", "patch"]},
                              "patch": {"type": "string", "description": "Unified diff against main (patch only)."}}),
                 tool_propose_fix),
        ToolSpec("verify_fix", "Recapture the finding on the fix branch (glitch must be gone) and survey every "
                 "scenario and screen size to check the fix adds no new glitches.", _fid_schema(), tool_verify_fix),
        ToolSpec("request_human", "Ask a human to approve or reject (required before file_report). "
                 "Rejecting deletes the fix branch.",
                 _fid_schema({"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}),
                 tool_request_human),
        ToolSpec("file_report", "Write the bug report with evidence images, culprit, fix and verification. "
                 "Only allowed after a human approved the finding.", _fid_schema(), tool_file_report),
        ToolSpec("dismiss", "Close a finding as not a bug (e.g. did not reproduce).",
                 _fid_schema({"reason": {"type": "string"}}, ["reason"]), tool_dismiss),
        ToolSpec("list_findings", "Current findings and their status.", {"type": "object", "properties": {}},
                 tool_list_findings),
        ToolSpec("look", "Return the finding's evidence images so you can inspect them.", _fid_schema(), tool_look),
        ToolSpec("finish", "End the hunt with a short summary.",
                 {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                 tool_finish),
    )
}
