"""Bug reports (Markdown + JSON) with the evidence images copied next to them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..vision import DETECTORS

TITLES = {
    "missing_texture": "Missing texture placeholder on {obj}",
    "zfight_flicker": "Z-fighting flicker on {obj}",
    "frozen_frame": "Game freezes during '{scenario}'",
    "black_frame": "Black screen in '{scenario}'",
    "flash": "Full-screen flashes (photosensitivity risk)",
    "geometry_explosion": "Exploded / stretched geometry on {obj}",
    "hud_text_overflow": "HUD text overflows its panel",
    "color_shift": "Whole picture changed tone or color in '{scenario}'",
    "shadow_acne": "Shadow acne (stripe/moire pattern) on {obj}",
    "detail_loss": "Blurry / low-resolution texture on {obj}",
    "object_missing": "{obj} missing at the end of '{scenario}'",
    "shading_inverted": "Inverted normal map on {obj} (relief lit from the wrong side)",
    "shadow_missing": "Missing shadow on {obj}",
    "physics_jitter": "{obj} vibrates while it should be at rest",
    "stutter": "Uneven motion (stutter) in '{scenario}'",
    "pop_in": "Visible pop (LOD or asset switch) on {obj}",
    "frozen_animation": "{obj} slides without animating",
    "nan_pixels": "Black NaN pixels over {obj}",
    "ui_offscreen": "UI cut off by the screen edge in '{scenario}'",
}


def title_for(f) -> str:
    title = TITLES.get(f.detector, f.detector).format(obj=f.object or "an object", scenario=f.scenario_label)
    return title[0].upper() + title[1:]


def _copy_images(f, dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, src in f.evidence.items():
        src_path = Path(src)
        if src_path.exists():
            target = dest / f"{name}.png"
            shutil.copy2(src_path, target)
            out[name] = f"{dest.name}/{target.name}"
    return out


def write_report(hunt_dir: Path, hunt_id: str, f, lab) -> Path:
    reports = hunt_dir / "reports"
    images = _copy_images(f, reports / f.id)
    sc = lab.project.scenarios.get(f.scenario)
    lines = [
        f"# {f.id}: {title_for(f)}",
        "",
        f"- **Severity:** {f.severity}",
        f"- **Detector:** `{f.detector}`: {DETECTORS[f.detector].description}",
        f"- **Seen in:** {', '.join([f.scenario_label, *f.also_seen])}",
        f"- **Build:** `{f.commit[:10]}` (hunt `{hunt_id}`)",
        f"- **Evidence:** {f.note}",
    ]
    if f.metrics.get("coplanar_surfaces"):
        lines.append(f"- **Coplanar surfaces at the hit point:** {', '.join(f.metrics['coplanar_surfaces'])}")
    size = f"&w={f.viewport[0]}&h={f.viewport[1]}" if f.viewport else ""
    lines += ["", "## Steps to reproduce", "",
              f"1. Build the game at the commit above and open `{lab.project.entry}?test=1&seed=1{size}`"
              + (f" (a {f.viewport[0]}x{f.viewport[1]} screen)." if f.viewport else "."),
              f"2. Scenario `{f.scenario}`: {sc.description if sc else ''}"]
    if f.camera:
        lines.append(f"3. Camera: position {f.camera['pos']}, looking at {f.camera['target']} "
                     f"(the agent zoomed in {f.zooms}x).")
    if f.repro:
        lines += ["", f"Reproduced **{f.repro['hits']}/{f.repro['runs']}** replays; frames were "
                      f"{'identical every run (deterministic)' if f.repro['deterministic'] else 'different between runs (flaky)'}."]
    lines += ["", "## Visual evidence", ""]
    for name, rel in images.items():
        lines.append(f"**{name.replace('_', ' ')}**\n\n![{name}]({rel})\n")
    if f.bisect and f.culprit:
        lines += ["## Culprit (visual git bisect)", "",
                  f"Searched {f.bisect.get('range')} commits between `{f.bisect['good']}` (good) and "
                  f"`{f.bisect['bad']}` (bad) with {f.bisect['repeats']} capture(s) per commit; "
                  f"{f.bisect.get('tested')} builds tested.", ""]
        if f.bisect.get("via_merge"):
            lines += [f"The first bad mainline commit was the merge `{f.bisect['via_merge']}` "
                      f"({f.bisect.get('merge_subject')}); bisect then searched the merged branch.", ""]
        if f.bisect.get("skipped"):
            lines += [f"Builds that did not start were skipped: {', '.join(f'`{s}`' for s in f.bisect['skipped'])}."
                      + (f" The culprit could also be {', '.join(f'`{s}`' for s in f.bisect['ambiguous_with'])}."
                         if f.bisect.get("ambiguous_with") else ""), ""]
        lines += ["| commit | subject | verdict | captures |", "|---|---|---|---|"]
        for step in f.bisect.get("steps", []):
            verdict = "skip (does not build)" if step["bad"] is None else ("bad" if step["bad"] else "good")
            lines.append(f"| `{step['commit']}` | {step['subject']} | {verdict} | {step['runs']} |")
        lines += ["", f"**First bad commit:** `{f.culprit['sha'][:10]}` {f.culprit['subject']}", "",
                  "```diff", f.culprit["diff"], "```", ""]
    if f.fix:
        lines += ["## Proposed fix", "", f"Branch `{f.fix['branch']}` at `{f.fix['sha'][:10]}` "
                  f"({f.fix['strategy']}).", "", "```diff", f.fix["diff"][:4000], "```", ""]
    if f.verification:
        v = f.verification
        lines += ["## Verification", "",
                  f"**{'PASS' if v['passed'] else 'FAIL'}**: {v['reason']} "
                  f"({v['runs']} targeted capture(s) on the fix build plus a survey of every scenario).", ""]
    if f.approval:
        a = f.approval
        lines += ["## Human decision", "", f"Question: {a['question']}", "",
                  f"Decision: **{a.get('choice')}** by `{a.get('by')}`" + (f", note: {a['note']}" if a.get("note") else ""), ""]
    lines += ["## Status history", ""]
    for h in f.history:
        lines.append(f"- {h['from']} -> **{h['to']}**: {h['why']}")
    path = reports / f"{f.id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / f"{f.id}.json").write_text(json.dumps(f.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def write_summary(hunt_dir: Path, hunt_id: str, inv, outcome: dict) -> Path:
    rows = ["| id | glitch | severity | status | culprit | fix | report |", "|---|---|---|---|---|---|---|"]
    for f in inv.ordered():
        culprit = f"`{f.culprit['sha'][:10]}` {f.culprit['subject']}" if f.culprit else ""
        fix = f"`{f.fix['branch']}`" if f.fix and not f.fix.get("deleted") else ""
        report = f"[{f.id}.md](reports/{f.id}.md)" if f.report else ""
        rows.append(f"| {f.id} | {title_for(f)} | {f.severity} | {f.status} | {culprit} | {fix} | {report} |")
    text = "\n".join([
        f"# GlitchHunter hunt `{hunt_id}`", "",
        f"- Planner: {outcome.get('planner')}", f"- Head: `{inv.head[:10]}`  baseline: `{inv.baseline[:10]}`",
        f"- Steps: {outcome.get('steps')}  captures: {outcome.get('captures')}  "
        f"time: {outcome.get('seconds', 0):.0f}s", f"- Ended because: {outcome.get('ended')}", "",
        *rows, "", "Full decision trace: [trace.jsonl](trace.jsonl)", ""])
    path = hunt_dir / "SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path
