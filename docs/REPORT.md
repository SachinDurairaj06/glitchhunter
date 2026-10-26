# GlitchHunter: technical report

OpenCV AI Competition 2026, Agentic Vision path.

## 1. Problem and users

Visual bugs (missing textures, flickering surfaces, frozen frames, broken meshes, clipped UI,
shadow acne, inverted normal maps, missing shadows, physics objects falling through floors,
characters sliding in a T-pose, LOD pops, NaN pixels from shaders) are among the most common
defects in games, and they are found mostly by people watching the screen. Large studios staff QA
teams; small and indie teams often cannot. Existing automation either compares screenshots
against reference images (engine graphics test frameworks) or asks a vision model whether a frame
"looks wrong"; neither tells a developer *which change* broke the build or whether a fix works.

**Users:** game developers and QA leads on small teams, whatever engine they use, and developer
agents (Claude Code, Kiro, any MCP client) that need to see what a code change did to the picture.

**Goal:** an agent that finds visual bugs in a build by looking at it, confirms them, finds the
commit that introduced each one, proposes a fix, checks the fix visually, and asks a human before
filing.

## 2. What the system does

One hunt is a perception-decision-action loop (diagrams in [ARCHITECTURE.md](ARCHITECTURE.md)):

1. **Perceive.** Play the scenarios of the game's test plan (`glitchhunter.json`) on the build under
   test, capture the frames, run 19 OpenCV 5 detectors in five families, compare with the
   approved build, and ask the game what object is under each finding.
2. **Decide.** A planner reads the findings (score, confidence, screen region, object, family) and
   picks the next tool call.
3. **Act.** Zoom the game camera in, check other screen sizes, replay, bisect the git history with
   a detector as the test (skipping commits that do not build, descending into merged branches),
   revert the culprit on a branch, verify the fix visually, ask a human, file a report.

The OpenCV output decides the next action at every step; section 6 shows it in real traces.

## 3. Architecture

| Layer | Implementation |
|---|---|
| Games under test | Three games with git histories that plant 28 bugs at known commits; the ground truth is stored outside the repos. **Crypt Runner** (three.js, 17 commits, 7 bugs). **Ironworks** (three.js with a skinned robot, shadows, bloom and custom post-processing passes, physics with substeps, LOD, instancing, a follow camera; 33 commits with a feature branch and a merge, 10 bugs). **Hangar 9** (Godot 4.7, Vulkan Forward+: ACES, glow, fog, directional and spot shadows, rigid bodies with CCD, an AnimationPlayer walk cycle, visibility-range LOD, normal mapping, CPU particles, custom culling bounds, a spatial shader, Control anchors; 28 commits with a merge and a commit that does not build, 11 bugs). |
| Capture | One interface, two drivers. Web: Playwright drives headless Chromium with SwiftShader WebGL (deterministic frames) through a small test API (`window.__gh`, available as a drop-in module). Native: the game connects back over TCP and answers nine JSON commands (bridges for Godot, tested, and Unity, reference). A pool of drivers captures in parallel. |
| Perception | OpenCV 5.0.0: 19 detectors in five families (classic, motion, compare, lighting, pixels), golden frames from the approved build, DIS optical flow, SSIM, DNN text reading (section 4). Detectors run in parallel threads. |
| Agent | Hunt loop with step and time budgets, an append-only JSONL trace, a human approval broker and Markdown/JSON reports. Two planners: a deterministic rule policy, and Claude Opus 5.5 on Amazon Bedrock using tool calls, which falls back to the rule policy on any Bedrock error. |
| Interfaces | FastAPI dashboard and HTTP tool API; an MCP server exposing 8 OpenCV tools to developer agents; a CLI with `init` and `bridge-check` for onboarding a new engine, and `scan-video` for recordings from any engine. |
| AWS | Graviton EC2 (arm64) running the container; S3 for hunt artifacts; Amazon Bedrock for the planner; CloudWatch Logs plus EMF metrics; IAM least privilege; SSM instead of SSH. Template: `deploy/cloudformation.yaml` (passes `cfn-lint`). `aws-doctor` checks each piece step by step. |

## 4. OpenCV 5 implementation

All perception uses OpenCV 5.0.0 (`opencv-python-headless==5.0.0.93`). `glitchhunter/cvinfo.py`
refuses to run on 4.x.

| Family | Detector | OpenCV 5 functions | Idea |
|---|---|---|---|
| classic | `missing_texture` | `cvtColor` (HSV), `inRange`, `morphologyEx`, `connectedComponentsWithStats` | Magenta cells interleaved with black cells, or flat saturated magenta (Unity's error shader). A purple prop is neither checkered nor pure. |
| classic | `zfight_flicker` | frame stacks, temporal std and step, morphological opening, DIS optical flow | Surfaces that toggle between frames; flow tells a vibrating object (a pixel or more of real motion) from a toggle. Small regions get low confidence on purpose, so the agent zooms in. |
| classic | `frozen_frame`, `black_frame`, `flash` | `norm`, gray thresholds, luma median/MAD | Hangs, black screens, full-screen flashes counted against WCAG 2.3.1. |
| classic | `geometry_explosion` | SSIM map, CIE Lab ΔE, `findContours`, `convexHull`, `Canny`, `HoughLinesP` | A spiky change versus the golden build with new outlines; changes a surface explains (relit, inverted relief, recoloured) do not count. |
| classic | `hud_text_overflow` | **OpenCV 5 DNN** `TextDetectionModel_DB` (PP-OCRv3) + CRNN, `ENGINE_AUTO` | UI text whose box runs past its panel. |
| compare | `color_shift` | Lab histograms (`calcHist`, `compareHist`) on unchanged pixels, per-tile consistency, power-law fit | A global tone or colour change; names the kind (too dark, washed out, desaturated, tint) and fits gamma (2.2 means linear output on an sRGB display). |
| compare | `shadow_acne` | `medianBlur` residual normalized by local brightness, structure tensor | Surfaces gaining a fine stripe pattern. |
| compare | `detail_loss` | contrast-normalized `Laplacian` energy, SSIM to a blurred golden frame | A texture stuck at a low mip. |
| compare | `object_missing` | end-state Lab ΔE + SSIM, `GaussianBlur` at the object's scale | An object that stands out from its surroundings in the golden build and not in this one, whose area changed average colour (a blurred painted line does not), and that is not a lighting change. Named from the golden build. |
| lighting | `shading_inverted` | difference of Gaussians normalized by brightness, windowed NCC | Relief detail that keeps its contrast but anti-correlates with the golden build: bumps read as dents (a normal map in the other engine's green-channel convention). |
| lighting | `shadow_missing` | brightness ratio after tone matching, NCC texture test, Lab chroma | An area the golden build has in shadow is now lit, with the same texture and colour and no shadow edge. |
| motion | `physics_jitter` | DIS flow per pixel: lag-1 autocorrelation, flips, coherence | An object vibrating while it should rest. |
| motion | `stutter` | DIS flow magnitude series | Uneven motion (frame pacing, a camera ticking at 20 Hz). |
| motion | `pop_in` | flow-compensated residual (`remap` with DIS flow); differences to the golden run | Content that switches in one frame; under a moving camera a one-frame step in the difference to the golden run must confirm it (flow fails at every occlusion edge). |
| motion | `frozen_animation` | DIS flow in tracked blobs minus their rigid motion | A character that moves but no longer articulates. |
| pixels | `nan_pixels` | dead-black components in lit surroundings | NaN/Inf from a shader. |
| pixels | `ui_offscreen` | panel edges + DNN text touching the border | UI cut off at another screen size. |

**Tone normalization.** A grade or gamma change touches nearly the whole picture; a local glitch
does not. Before any local comparison, per-channel curves fitted on quantile bins of co-located
pixels, then a robust 3x4 colour matrix (least squares with outlier rejection, undoing saturation
grades that curves cannot), map the build into the golden build's tones, but only when the
difference covers most tiles and the mapping clearly helps. Without it, a desaturated build
reported a "missing" painted line and a "missing object" where only a shadow was gone.

**One DIS engine per image size and thread.** OpenCV's DIS optical flow adapts its patch size to
the first small image it sees and keeps that setting, so an engine shared between 40x40 crops and
640x360 frames gave different answers depending on what had run before; a calibration run
reproduced a false pop only after an unrelated z-fighting check. Each image size and thread now
gets its own engine: answers are stable, and the flow-heavy detectors became twice as fast.

**New DNN engine.** OpenCV 5's graph engine (`ENGINE_AUTO`, which falls back to `ENGINE_CLASSIC`
if a layer is unsupported) runs both OpenCV Zoo text models. On the development machine (x86_64,
12 threads), PP-OCRv3 detection on a 640x360 frame took 43 ms with the new engine versus 137 ms
with the classic engine (`bench/results/`; indicative).

## 5. Agent design

**Tools** (JSON-schema specs shared by both planners, MCP and HTTP): `survey`, `sweep_viewports`,
`soak`, `zoom_in`, `replay`, `bisect`, `show_commit`, `propose_fix`, `verify_fix`,
`request_human`, `file_report`, `dismiss`, `list_findings`, `look`, `finish`.

**Active perception (`zoom_in`).** The agent ray-casts through the finding (detectors mark points
on the glitched thing itself, so a thin walking figure is named, not the floor behind it), moves
the camera along the line to the hit point by a factor of 2-3, recaptures and re-runs the
detector. Screen-space evidence and scenarios that move the camera are not zoomed.

**Screen sizes (`sweep_viewports`).** UI scenarios are replayed at every size the test plan lists
(4:3, 21:9): a panel anchored for 16:9 runs off a 4:3 screen.

**Reproduction and determinism (`replay`).** Identical frames, or identical evidence, on every
replay means the glitch is deterministic: the bisect uses one capture per commit. Otherwise the
hit rate is estimated with a Beta(1,1) prior, p = (k+1)/(N+2), and the bisect uses r = ceil(log 0.05
/ log(1-p)) captures per commit (at most 8). Every re-check sees the whole frame, as the survey
did (an object that is gone covers a small part of the frame but most of a crop), and keeps what
overlaps the finding.

**Visual git bisect (`bisect`).** Search over the first-parent commits between the approved build
and the build under test, with the finding's detector as the test. Commits that do not build are
skipped (`git bisect skip` semantics); if the culprit could be one of them, the report says so.
When the first bad mainline commit is a merge, the search continues inside the merged branch, and
the fix reverts with `-m 1` if the culprit is the merge itself. With k capture workers each round
tests k evenly spaced commits at once. Findings that bisect to the same culprit are one root
cause: they are filed once.

**Fix and verification.** `propose_fix` reverts the culprit on `glitchhunter/<hunt>-<finding>` in
a throwaway worktree; `main` is never touched. The Bedrock planner may also propose a unified-diff
patch, which is rejected unless it touches only files the test plan allows. `verify_fix`
recaptures the finding on the fix build and re-surveys every scenario and screen size. A glitch
that appears only on the fix build is a regression, unless it matches another open finding (the
fix unmasked it) or shows on the original build too (an intermittent bug the survey missed: a new
finding).

**Failure handling.** Tool exceptions go back to the planner as errors; five in a row stop the
hunt. Step and time budgets are enforced. Any Bedrock failure switches to the rule planner,
recorded in the trace. Builds that do not start, never connect, or log script errors are skipped,
not fatal. Strong evidence that does not reproduce gets 8 more replays before it is dismissed. A
flaky glitch whose fix fails is bisected again with twice the captures. A bisect whose ends do not
behave is escalated to a human.

**Human control.** `file_report` refuses unless a person approved the finding. On the dashboard
each request shows the before/after evidence and takes an optional note. Rejecting deletes the
agent's fix branch. Unattended runs must choose a policy explicitly, and every decision records
who made it.

**Observability.** `trace.jsonl` logs every decision with its reason and evidence, every tool
result, and every bisect step. CloudWatch EMF metrics cover capture, detector, tool and hunt time
and findings reported. Each hunt writes `SUMMARY.md`, `findings.json` and per-finding reports,
and uploads them to S3 when configured.

**Bedrock planner.** It uses the Anthropic Python SDK's Bedrock Mantle client with model
`anthropic.claude-opus-5-5`, adaptive thinking with summarized reasoning (the summary becomes the
decision's `reason` in the trace), and prompt caching. Tool results carry up to two evidence
images, so the model sees the frames as well as the numbers.

## 6. Evidence that vision changes the next action

From the Hangar 9 (Godot) hunt in this repository's evaluation runs:

```text
112.5 RESULT survey      16 new findings: color_shift, shadow_acne, detail_loss, object_missing,
                         nan_pixels, shading_inverted, shadow_missing, frozen_animation, pop_in ...
117.6 RESULT sweep       checked the layout at 480x360: F17=ui_offscreen
124.7 RESULT replay      F1 replayed 3x: seen 3/3, identical frames every run (deterministic)
124.7 DECIDE bisect      F1 ... deterministic, 1 capture per commit
148.8 RESULT bisect      F1 culprit 812271b1d3 'grading: muted look for the dusk mission' after
                         testing 7 builds; skipped 1 broken build(s)
286.2 RESULT bisect      F6 culprit 93f6b24702 'heat haze: n*sqrt(n) instead of pow' (duplicate of F11)
```

From the Crypt Runner sample hunt (dashboard run, web approvals): a z-fighting decal at 121 px
gets confidence 0.10, so the agent moves the camera 3x closer (0.33), then 2x closer (0.99), then
replays and bisects with one capture per commit; a random flash seen in 1 of 4 replays is
bisected with 8 captures per commit to keep the miss chance under 5%.

## 7. Evaluation

<!-- eval:start -->
**Method.** `python -m glitchhunter eval --game <game>` rebuilds a game's history, runs the agent
in several modes against its planted bugs, and captures every scenario of the clean builds to
count false alarms. Ground truth comes from the fixture manifest, which the agent never reads.

- `full`: the agent with zoom, viewport sweep, replay, bisect, fix and verification.
- `no_zoom`: the same agent without active perception.
- `no_sweep`: the same agent without the screen-size sweep.
- `fixed_pipeline`: no agent; one survey of HEAD that reports every detection with confidence ≥ 0.5.

A planted bug counts as found when a reported finding bisects to its commit (or, without a
bisect, when its detector reports it). Results: [eval/RESULTS.md](../eval/RESULTS.md).

**Calibration.** Every detector is also run on clips captured at each bug's commit, at the commit
before it, and on the clean builds of all three games (`scratch` calibration, 150+ clips). It
found most of the robustness work in this report: tone normalization, the DIS engine reuse bug,
golden-run confirmation for pops under a moving camera, the purple-sign false positive.
<!-- eval:end -->

## 8. Limitations

- **Our games, our bugs.** The three games and their 28 bugs were written for this project. The
  detectors were calibrated on them, which is also the evaluation set. Expect lower numbers on
  unseen games. Nothing has been measured on commercial games or public glitch datasets.
- **Unity and Unreal not run.** The Unity bridge and build script follow the protocol the Godot
  bridge is tested with, but they have not been run in a Unity project (no Unity license here);
  Unreal has a protocol description and a reference engine, not a bridge.
- **Determinism is required for bisect.** Engines and games that cannot step a fixed time per frame
  or seed their randomness can still be surveyed, replayed and triaged, but bisect needs more
  captures per commit; `bridge-check` measures this up front.
- **Bedrock planner not run live.** No AWS credentials were available during development. The
  planner is covered by tests with a scripted client; the rule planner produced every result here.
- **AWS stack not deployed yet.** The Dockerfile and CloudFormation template have not been built or
  deployed from the development machine (the template passes `cfn-lint`). COOL was not measured.
- **Flaky bugs can be missed.** If an intermittent glitch never shows during the survey and soak
  runs, it is never investigated.
- **One culprit per finding.** A glitch caused by two interacting commits is attributed to the
  later one. Revert is a blunt fix; the patch path exists but was not evaluated.

## 9. Responsible use

- **Data.** The agent sees only frames of the game under test and its git history. No personal data
  is captured. Bedrock receives game frames and detector numbers only.
- **Human approval.** Nothing is filed and no fix is kept without a person's decision (or an
  explicit, logged policy for unattended evaluation runs). The agent cannot modify `main`; fix
  branches are namespaced and deleted on rejection; LLM-proposed patches may touch only game files.
- **Accessibility.** The flash detector counts full-screen flashes against the WCAG 2.3.1
  three-flashes-per-second threshold, because such flashes can trigger seizures.
- **Least privilege on AWS.** The instance role can only write `glitchhunter/*` in its own bucket,
  call Bedrock, and write its own log group. IMDSv2, no SSH, one CIDR for the dashboard, a bearer
  token on every POST, artifacts expire after 30 days.
- **Honest reporting.** Evaluation results come from the shipped harness and include failures.

## 10. Reproduce

```bash
python -m glitchhunter fixture --game hangar
```

```bash
python -m glitchhunter hunt --game hangar --approval auto-approve
```

```bash
python -m glitchhunter eval --game ironworks --repeats 1
```

```bash
python -m pytest
```
