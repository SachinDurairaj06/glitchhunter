"""Evaluation against the planted bugs (ground truth next to the fixture repo), with ablations.

Modes
  full            agent with active perception (zoom), screen-size sweep, replay, visual bisect, fix + verify
  no_zoom         same agent, zoom disabled (ablation of active perception)
  no_sweep        same agent, no screen-size sweep (ablation of the layout check)
  fixed_pipeline  no agent: one survey of HEAD, report every detection (no culprit, no fix)
Plus a false-positive check: every scenario (and screen size) on the game's clean builds.

A finding counts for a bug when its culprit is the bug's commit (two symptoms of one cause are
one bug) or, without a culprit, when its detector is the bug's detector. Anything else that
was not dismissed is a false alarm.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .agent.approvals import ApprovalBroker
from .agent.loop import Hunt
from .agent.planner import RulePlanner
from .agent.tools import LAYOUT_DETECTORS
from .config import REPO_ROOT, load_settings
from .fixture import FixtureInfo, build_fixture
from .games import get_game
from .lab import BuildError, Lab
from .vision import BUG_DETECTORS


def match_bug(f: dict, info: FixtureInfo) -> str | None:
    culprit = (f.get("culprit") or {}).get("sha")
    if culprit:
        for name, bug in info.bugs.items():
            if culprit == bug["culprit_sha"] or culprit == bug.get("merge_sha"):
                return name
        return None
    for name, bug in info.bugs.items():
        if f["detector"] == bug["detector"]:
            return name
    return None


def score_findings(findings: list[dict], info: FixtureInfo) -> dict:
    """Match findings to planted bugs; count culprits, verified fixes and false alarms."""
    per_bug = {name: {"detected": False, "culprit_correct": False, "fix_verified": False, "reported": False}
               for name in info.bugs}
    false_alarms = []
    for f in findings:
        if f["status"] in ("dismissed", "duplicate"):
            continue
        bug = match_bug(f, info)
        if bug is None:
            false_alarms.append(f"{f['id']}:{f['detector']}")
            continue
        row = per_bug[bug]
        row["detected"] = True
        culprit = (f.get("culprit") or {}).get("sha")
        if culprit and culprit == info.bugs[bug]["culprit_sha"]:
            row["culprit_correct"] = True
        if (f.get("verification") or {}).get("passed"):
            row["fix_verified"] = True
        if f["status"] == "reported":
            row["reported"] = True
    n = len(info.bugs)
    detected = sum(r["detected"] for r in per_bug.values())
    return {
        "per_bug": per_bug,
        "recall": detected / n,
        "culprit_accuracy": sum(r["culprit_correct"] for r in per_bug.values()) / n,
        "fix_verified": sum(r["fix_verified"] for r in per_bug.values()) / n,
        "reported": sum(r["reported"] for r in per_bug.values()) / n,
        "false_alarms": len(false_alarms),
        "false_alarm_ids": false_alarms,
        "precision": detected / max(detected + len(false_alarms), 1),
    }


def run_hunt_mode(lab: Lab, settings, mode: str, label: str) -> dict:
    planner = RulePlanner(zoom=(mode != "no_zoom"), sweep=(mode != "no_sweep"))
    hunt = Hunt(lab, settings, ApprovalBroker("auto-approve"), planner=planner, label=label)
    captures_before = lab.capture_count
    outcome = hunt.run()
    findings = json.loads((hunt.dir / "findings.json").read_text(encoding="utf-8"))
    return {"hunt": hunt.id, "steps": outcome["steps"], "seconds": outcome["seconds"],
            "captures": lab.capture_count - captures_before, "findings": findings}


def run_fixed_pipeline(lab: Lab) -> dict:
    t0 = time.time()
    findings = []
    seen = set()
    for scenario in lab.project.scenarios:
        run = lab.capture(lab.project.branch, scenario, keep=False, purpose="fixed-pipeline")
        for det, _ in lab.inspect(run, BUG_DETECTORS, tag="fixed-"):
            if det.confidence < 0.5 or det.detector in seen:
                continue
            seen.add(det.detector)
            findings.append({"id": f"P{len(findings) + 1}", "detector": det.detector, "status": "reported",
                             "confidence": det.confidence})
    return {"steps": 1, "seconds": round(time.time() - t0, 1), "captures": len(lab.project.scenarios),
            "findings": findings}


def false_positive_check(lab: Lab, info: FixtureInfo, repeats: int) -> dict:
    rows = []
    for sha in info.clean:
        for _ in range(repeats):
            for scenario in lab.project.scenarios:
                try:
                    run = lab.capture(sha, scenario, keep=False, purpose="fp-check")
                except BuildError:
                    continue
                dets = [d.detector for d, _ in lab.inspect(run, BUG_DETECTORS, tag="fp-")]
                rows.append({"commit": sha[:10], "scenario": scenario, "detections": dets})
            for vp in lab.project.viewports:
                for scenario in lab.project.sweep_scenarios:
                    run = lab.capture(sha, scenario, keep=False, purpose="fp-check", viewport=tuple(vp))
                    dets = [d.detector for d, _ in lab.inspect(run, LAYOUT_DETECTORS, tag="fp-")]
                    rows.append({"commit": sha[:10], "scenario": f"{scenario}@{vp[0]}x{vp[1]}", "detections": dets})
    runs = len(rows)
    with_fp = sum(1 for r in rows if r["detections"])
    return {"runs": runs, "runs_with_false_alarm": with_fp, "false_alarm_rate": with_fp / max(runs, 1),
            "commits": [c[:10] for c in info.clean], "rows": rows}


def _mean(values):
    return round(statistics.mean(values), 3) if values else 0.0


def summarize(mode_runs: list[dict], info: FixtureInfo) -> dict:
    scores = [score_findings(r["findings"], info) for r in mode_runs]
    per_bug = {}
    for bug in info.bugs:
        per_bug[bug] = {k: _mean([float(s["per_bug"][bug][k]) for s in scores])
                        for k in ("detected", "culprit_correct", "fix_verified", "reported")}
    return {
        "runs": len(mode_runs),
        "recall": _mean([s["recall"] for s in scores]),
        "precision": _mean([s["precision"] for s in scores]),
        "culprit_accuracy": _mean([s["culprit_accuracy"] for s in scores]),
        "fix_verified": _mean([s["fix_verified"] for s in scores]),
        "false_alarms": _mean([s["false_alarms"] for s in scores]),
        "false_alarm_ids": [s["false_alarm_ids"] for s in scores],
        "seconds": _mean([r["seconds"] for r in mode_runs]),
        "captures": _mean([r["captures"] for r in mode_runs]),
        "steps": _mean([r["steps"] for r in mode_runs]),
        "per_bug": per_bug,
        "hunts": [r.get("hunt") for r in mode_runs],
    }


def write_markdown(result: dict, path: Path) -> None:
    spec = get_game(result["game"])
    modes = result["modes"]
    n_flaky = sum(1 for b in spec.bugs.values() if b.flaky)
    history = (f"{len(spec.history)}-commit history"
               + (" with a merged feature branch and a commit that does not build"
                  if spec.notes.get("untestable_subjects") else ""))
    lines = [
        f"# GlitchHunter evaluation: {spec.title}", "",
        f"Run {result['started']} on {result['machine']} with OpenCV {result['opencv']} "
        f"({result['repeats']} repeat(s) per mode). Ground truth: {len(spec.bugs)} bugs planted in a {history} "
        f"({len(spec.bugs) - n_flaky} deterministic, {n_flaky} intermittent).", "",
        "| mode | recall | precision | culprit found | fix verified | false alarms / run | captures | time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in modes.items():
        lines.append(f"| {name} | {m['recall']:.0%} | {m['precision']:.0%} | {m['culprit_accuracy']:.0%} | "
                     f"{m['fix_verified']:.0%} | {m['false_alarms']:.1f} | {m['captures']:.0f} | {m['seconds']:.0f} |")
    lines += ["", "## Per bug (fraction of runs)", "",
              "| bug | detector | " + " | ".join(f"{m} detected" for m in modes) + " | full: culprit | full: fix verified |",
              "|---|---|" + "---|" * (len(modes) + 2)]
    for bug, b in spec.bugs.items():
        cells = [f"{modes[m]['per_bug'][bug]['detected']:.0%}" for m in modes]
        full = modes.get("full", {}).get("per_bug", {}).get(bug, {})
        kind = "intermittent" if b.flaky else "deterministic"
        where = f", only at {b.viewport[0]}x{b.viewport[1]}" if b.viewport else ""
        lines.append(f"| {b.title} ({kind}{where}) | `{b.detector}` | " + " | ".join(cells)
                     + f" | {full.get('culprit_correct', 0):.0%} | {full.get('fix_verified', 0):.0%} |")
    fp = result["false_positive_check"]
    lines += ["", "## False alarms on clean builds", "",
              f"{fp['runs_with_false_alarm']} of {fp['runs']} captures of clean builds ({', '.join(fp['commits'])}; "
              f"every scenario and screen size) produced a detection ({fp['false_alarm_rate']:.1%}).", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(results_dir: Path, out: Path) -> None:
    """eval/RESULTS.md: the newest result for each game."""
    newest: dict[str, dict] = {}
    for p in sorted(results_dir.glob("eval-*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        newest[data.get("game", "crypt")] = data
    lines = ["# GlitchHunter evaluation", "",
             "Newest run per test game (`python -m glitchhunter eval --game <name>`).", ""]
    for game, data in newest.items():
        spec = get_game(game)
        lines += [f"## {spec.title}", "", f"[details](RESULTS_{game}.md)", "",
                  "| mode | recall | culprit found | fix verified | false alarms / run | time (s) |",
                  "|---|---|---|---|---|---|"]
        for name, m in data["modes"].items():
            lines.append(f"| {name} | {m['recall']:.0%} | {m['culprit_accuracy']:.0%} | {m['fix_verified']:.0%} | "
                         f"{m['false_alarms']:.1f} | {m['seconds']:.0f} |")
        fp = data.get("false_positive_check", {})
        if fp:
            lines += ["", f"Clean builds: {fp['runs_with_false_alarm']} of {fp['runs']} captures with a detection.", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args) -> int:
    from .cvinfo import opencv_info

    game = getattr(args, "game", None) or "crypt"
    settings = load_settings(work_dir=getattr(args, "work_dir", None), max_steps=400, game=game)
    settings.max_seconds = max(settings.max_seconds, 5400.0)
    info = build_fixture(settings.fixture_dir, settings.fixture_manifest, game)
    lab = Lab(settings)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    result = {"game": game, "started": time.strftime("%Y-%m-%d %H:%M"), "repeats": args.repeats,
              "opencv": opencv_info()["version"], "machine": opencv_info()["machine"], "modes": {}, "raw": {}}
    try:
        for mode in modes:
            runs = []
            for i in range(args.repeats):
                print(f"[eval] {game} {mode} run {i + 1}/{args.repeats}", flush=True)
                if mode == "fixed_pipeline":
                    runs.append(run_fixed_pipeline(lab))
                else:
                    runs.append(run_hunt_mode(lab, settings, mode, f"eval-{game}-{mode}-{i + 1}"))
                s = score_findings(runs[-1]["findings"], info)
                print(f"[eval]   -> {s['recall']:.0%} recall, {s['culprit_accuracy']:.0%} culprit, "
                      f"{s['fix_verified']:.0%} fixed, false alarms {s['false_alarm_ids']}, {runs[-1]['seconds']}s",
                      flush=True)
            result["modes"][mode] = summarize(runs, info)
            result["raw"][mode] = runs
        print("[eval] false-positive check on clean builds", flush=True)
        result["false_positive_check"] = false_positive_check(lab, info, repeats=max(1, min(args.repeats, 2)))
    finally:
        lab.close()
    out_dir = Path(args.out) if args.out else REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    (out_dir / f"eval-{game}-{stamp}.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "raw"}, indent=2, default=str), encoding="utf-8")
    write_markdown(result, out_dir.parent / f"RESULTS_{game}.md")
    write_index(out_dir, out_dir.parent / "RESULTS.md")
    print((out_dir.parent / f"RESULTS_{game}.md").read_text(encoding="utf-8"))
    return 0
