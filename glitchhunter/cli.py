"""Command line: python -m glitchhunter <command>."""

from __future__ import annotations

import argparse
import json
import sys


def _progress(event: dict) -> None:
    kind = event["kind"]
    if kind == "decision":
        print(f"[{event['t']:7.1f}s] DECIDE {event['tool']:<13} {event['reason']}", flush=True)
    elif kind == "result":
        mark = "ok " if event["ok"] else "ERR"
        print(f"[{event['t']:7.1f}s]   {mark}  {event['summary']}", flush=True)
    elif kind == "bisect_step":
        print(f"[{event['t']:7.1f}s]        bisect {event['commit']}: {event['verdict']} ({event['runs']} capture(s))",
              flush=True)
    elif kind in ("planner_error", "fallback", "tool_error", "warning"):
        print(f"[{event['t']:7.1f}s]   !!  {kind}: {event.get('error') or event.get('reason') or ''}", flush=True)


def cmd_fixture(args) -> int:
    from .config import load_settings
    from .fixture import build_fixture

    s = load_settings(work_dir=args.work_dir, game=args.game)
    info = build_fixture(s.fixture_dir, s.fixture_manifest, s.game)
    print(f"game repo ({s.game}): {s.fixture_dir}")
    for c in info.commits:
        branch = f"  ({c['branch']})" if c.get("branch", "main") != "main" else ""
        tag = f"  [{c['tag']}]" if c["tag"] else ""
        print(f"  {c['index']:2d} {c['sha'][:10]} {c['subject']}{tag}{branch}")
    print(f"ground truth (not visible to the agent): {s.fixture_manifest}")
    return 0


def cmd_hunt(args) -> int:
    from .agent.approvals import ApprovalBroker
    from .agent.loop import Hunt
    from .config import load_settings
    from .lab import Lab

    s = load_settings(work_dir=args.work_dir, planner=args.planner, approval_mode=args.approval,
                      max_steps=args.max_steps, game=args.game, repo=args.repo)
    lab = Lab(s)
    try:
        hunt = Hunt(lab, s, ApprovalBroker(s.approval_mode, s.approval_timeout_s), label=args.label or "")
        if not args.quiet:
            hunt.trace.subscribe(_progress)
        print(f"hunt {hunt.id}: game={lab.project.title} ({lab.project.source}) planner={s.planner} "
              f"approvals={s.approval_mode} output={hunt.dir}", flush=True)
        outcome = hunt.run()
    finally:
        lab.close()
    print(json.dumps({k: v for k, v in outcome.items() if k != "findings"}, indent=2))
    for f in outcome["findings"]:
        culprit = (f.get("bisect") or {}).get("culprit", "")
        print(f"  {f['id']:4} {f['detector']:<18} {f['status']:<18} {f.get('object') or f['scenario']:<20} {culprit}")
    print(f"summary: {hunt.dir / 'SUMMARY.md'}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api.app import create_app
    from .config import load_settings

    s = load_settings(work_dir=args.work_dir, approval_mode="web", game=args.game, repo=args.repo)
    uvicorn.run(create_app(s), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import build_server

    build_server().run(transport=args.transport)
    return 0


def cmd_eval(args) -> int:
    from .evaluate import main as eval_main

    return eval_main(args)


def cmd_bench(args) -> int:
    from .bench import main as bench_main

    return bench_main(args)


def cmd_info(args) -> int:
    from .cvinfo import opencv_info

    print(json.dumps(opencv_info(), indent=2))
    return 0


def cmd_bedrock_check(args) -> int:
    """One tiny request to Claude on Amazon Bedrock, to confirm credentials, region and model access."""
    import time

    from .agent.bedrock import DEFAULT_MODEL
    from .config import load_settings

    s = load_settings()
    model = args.model or s.bedrock_model_id or DEFAULT_MODEL
    print(f"region={s.aws_region} model={model}")
    try:
        from anthropic import AnthropicBedrockMantle

        client = AnthropicBedrockMantle(aws_region=s.aws_region)
        t0 = time.perf_counter()
        msg = client.messages.create(model=model, max_tokens=64,
                                     messages=[{"role": "user", "content": "Reply with the single word OK."}])
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Check: AWS credentials (aws sts get-caller-identity), AWS_REGION, and model access for this "
              "model in the Amazon Bedrock console. The agent still runs with --planner rule.")
        return 1
    text = next((b.text for b in msg.content if b.type == "text"), "")
    print(f"OK in {time.perf_counter() - t0:.1f}s: stop_reason={msg.stop_reason} reply={text!r} "
          f"input_tokens={msg.usage.input_tokens} output_tokens={msg.usage.output_tokens}")
    return 0


def cmd_record(args) -> int:
    """Record scenarios of a build to an MP4 (for demos, or as input to scan-video)."""
    from pathlib import Path

    from .config import load_settings
    from .lab import Lab
    from .video import write_video

    s = load_settings(work_dir=args.work_dir, game=args.game, repo=args.repo)
    lab = Lab(s)
    try:
        names = [n for n in (args.scenarios or "").split(",") if n] or list(lab.project.scenarios)
        frames = []
        for name in names:
            run = lab.capture(args.commit, name, keep=False, purpose="record",
                              viewport=tuple(int(v) for v in args.viewport.split("x")) if args.viewport else None)
            frames += run.frames
            print(f"{name}: {len(run.frames)} frames")
    finally:
        lab.close()
    out = write_video(frames, Path(args.out), lab.project.fps)
    print(f"wrote {out} ({len(frames)} frames)")
    return 0


def cmd_scan_video(args) -> int:
    """Find glitches in a gameplay recording from any engine."""
    from pathlib import Path

    from .video import VideoTriage

    out = Path(args.out) if args.out else Path(args.video).with_suffix("").parent / (Path(args.video).stem + "-triage")
    triage = VideoTriage(Path(args.video), out, golden=Path(args.golden) if args.golden else None,
                         every=args.every, window=args.window, max_frames=args.max_frames)
    result = triage.run()
    print(json.dumps(result, indent=2))
    return 0


def cmd_bridge_check(args) -> int:
    """Check that a native build speaks the bridge protocol, deterministically."""
    from pathlib import Path

    from .conformance import check_build, format_checks, launch_from_repo

    launch_cmd = [a for a in args.launch if a != "--"]
    if args.repo:
        launch_cmd, viewport, cwd = launch_from_repo(Path(args.repo))
    elif launch_cmd:
        viewport, cwd = (640, 360), Path(args.cwd) if args.cwd else Path.cwd()
    else:
        print("give --repo PATH (a project with a glitchhunter.json) or the launch command after --, "
              "with {port} where the game takes GlitchHunter's port", file=sys.stderr)
        return 2
    if args.viewport:
        viewport = tuple(int(v) for v in args.viewport.split("x"))
    print(f"launching: {' '.join(launch_cmd)}  ({viewport[0]}x{viewport[1]})")
    checks = check_build(launch_cmd, viewport, cwd, args.frames, args.key, args.timeout)
    print(format_checks(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def cmd_init(args) -> int:
    """Add GlitchHunter to a game project: test plan + the engine's bridge."""
    from pathlib import Path

    from .scaffold import init_project

    for line in init_project(Path(args.dir), args.engine, name=args.name, baseline=args.baseline, force=args.force):
        print(line)
    return 0


def cmd_aws_doctor(args) -> int:
    from .awsdoctor import main as doctor_main

    return doctor_main(args)


def cmd_fetch_models(args) -> int:
    from .textdnn import fetch_models

    for path in fetch_models():
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glitchhunter", description=__doc__)
    parser.add_argument("--work-dir", default=None, help="state directory (default: GH_WORK_DIR or <repo>/.work)")
    sub = parser.add_subparsers(dest="command", required=True)

    def game_args(p, repo: bool = True):
        from .games import GAME_NAMES

        p.add_argument("--game", choices=list(GAME_NAMES), default=None,
                       help="built-in test game (default: GH_GAME or crypt)")
        if repo:
            p.add_argument("--repo", default=None, help="hunt in any game repo with a glitchhunter.json instead")

    p = sub.add_parser("fixture", help="(re)build a test game's git history with planted bugs")
    game_args(p, repo=False)
    p.set_defaults(fn=cmd_fixture)

    p = sub.add_parser("hunt", help="run the agent against the game repo")
    game_args(p)
    p.add_argument("--planner", choices=["rule", "bedrock"], default=None)
    p.add_argument("--approval", choices=["auto-approve", "auto-reject", "cli", "web"], default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--label", default="")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_hunt)

    p = sub.add_parser("serve", help="web dashboard + HTTP tool API")
    game_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("mcp", help="expose the OpenCV tools over the Model Context Protocol")
    p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    p.set_defaults(fn=cmd_mcp)

    p = sub.add_parser("eval", help="score the agent against the planted bugs (plus ablations)")
    game_args(p, repo=False)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--modes", default="full,no_zoom,fixed_pipeline",
                   help="comma-separated: full, no_zoom, no_sweep, fixed_pipeline")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("bench", help="time the OpenCV workload (pip OpenCV 5 vs COOL on Graviton)")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--frames", default=None, help="directory of captured PNG frames (default: synthetic)")
    p.add_argument("--price-per-hour", type=float, default=None, help="instance price, for cost per 1000 runs")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two benchmark files")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_bench)

    sub.add_parser("info", help="OpenCV build information").set_defaults(fn=cmd_info)
    p = sub.add_parser("bedrock-check", help="send one tiny request to Claude on Amazon Bedrock")
    p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_bedrock_check)
    sub.add_parser("fetch-models", help="download the OpenCV Zoo text models").set_defaults(fn=cmd_fetch_models)

    p = sub.add_parser("aws-doctor", help="check AWS credentials, Bedrock access, S3 and CloudWatch step by step")
    p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_aws_doctor)

    from .awsdeploy import add_parser as add_deploy_parser

    add_deploy_parser(sub)

    p = sub.add_parser("record", help="record scenarios of a build to an MP4")
    game_args(p)
    p.add_argument("--commit", default="main")
    p.add_argument("--scenarios", default="", help="comma-separated (default: all)")
    p.add_argument("--viewport", default=None, help="e.g. 480x360")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("scan-video", help="find glitches in a gameplay recording from any engine")
    p.add_argument("video")
    p.add_argument("--golden", default=None, help="frame-aligned recording of the same run on an approved build")
    p.add_argument("--every", type=int, default=1, help="analyse every n-th frame")
    p.add_argument("--window", type=int, default=48, help="frames per analysis window")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_scan_video)

    from .scaffold import ENGINES

    p = sub.add_parser("init", help="add GlitchHunter to a game project (any engine): test plan + bridge")
    p.add_argument("--engine", choices=list(ENGINES), required=True)
    p.add_argument("--dir", default=".", help="the game project folder")
    p.add_argument("--name", default=None)
    p.add_argument("--baseline", default="v1.0", help="tag of the approved build")
    p.add_argument("--force", action="store_true", help="replace files that exist")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("bridge-check", help="check a native build's bridge: every command, and determinism")
    p.add_argument("--repo", default=None, help="use the launch command from this project's glitchhunter.json")
    p.add_argument("--cwd", default=None, help="working directory for the launch command ({dir})")
    p.add_argument("--viewport", default=None, help="e.g. 640x360")
    p.add_argument("--frames", type=int, default=12, help="frames per determinism run")
    p.add_argument("--key", default="Space", help="key to press in the input check (KeyboardEvent.code)")
    p.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the game to connect")
    p.add_argument("launch", nargs=argparse.REMAINDER, help="-- launch command with {port}, {width}, {height}")
    p.set_defaults(fn=cmd_bridge_check)

    args = parser.parse_args(argv)
    if args.command not in ("info", "aws-doctor", "deploy"):
        from .cvinfo import require_opencv5

        require_opencv5()
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
