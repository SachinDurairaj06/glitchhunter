# OpenCV tools over MCP (for developer agents)

`python -m glitchhunter mcp` starts a Model Context Protocol server (stdio by default,
`--transport streamable-http` for remote clients). It exposes the same OpenCV 5 workload the
autonomous agent uses, so a coding agent can check what a change does to the picture before
it opens a pull request.

| Tool | Arguments | Returns |
|---|---|---|
| `opencv_info` | none | OpenCV version, DNN engines, CPU features, scenarios, detectors |
| `list_commits` | `limit` | Commits of the game repo, newest first |
| `show_commit` | `sha` | Subject, files, diff |
| `capture` | `commit`, `scenario`, optional `camera` `{pos, target, fov}`, `seed` | `run_id`, frame count, game errors, contact sheet path |
| `detect` | `run_id`, optional `detectors`, optional `region` `[x, y, w, h]` | Detections (score, confidence, region, metrics) and evidence image paths |
| `pick` | `run_id`, `x`, `y`, `frame` | Object under the pixel, 3D hit point, coplanar surfaces |
| `compare_to_golden` | `run_id` | SSIM structural diff and exploded-geometry check against `release-1.0` |
| `bisect` | `detector`, `scenario`, optional `good`, `bad`, `camera`, `region`, `repeats` | Culprit commit and the builds tested |

## Add it to Claude Code

From the repository root:

```bash
claude mcp add glitchhunter -- python -m glitchhunter mcp
```

## Example session (recorded with the MCP Python client)

```text
list_tools -> opencv_info, list_commits, show_commit, capture, detect, pick, compare_to_golden, bisect
capture {"commit": "main", "scenario": "map_view"}
  -> {"run_id": "10230314310a7f", "commit": "c7a252ba80", "scenario": "map_view", "frames": 10, "seconds": 0.49}
detect {"run_id": "10230314310a7f"}
  -> black_frame        score 0.95  "10/10 frames are >85% black; only the UI (if anything) is still drawn"
  -> hud_text_overflow  score 1.00  "UI text ['dscoretoa'] extends 86px beyond its panel"
```

A developer agent might continue with `bisect {"detector": "black_frame", "scenario": "map_view"}`,
which returns `extend map view draw distance` with its diff, and then fix the clip planes in its own
working copy.

## HTTP equivalent

Every MCP tool is also available at `POST /api/tools/<name>` with the same JSON arguments, which is
useful for CI jobs:

```bash
curl -s -X POST http://127.0.0.1:8000/api/tools/capture -H "Content-Type: application/json" -d '{"commit": "main", "scenario": "overview"}'
```
