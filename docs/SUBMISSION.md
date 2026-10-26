# Devpost submission draft

Copy each section into the matching Devpost field. Replace anything in <angle brackets> before you submit.

## Project name

GlitchHunter

## Tagline

An OpenCV 5 agent that finds visual bugs in a game build from any engine, bisects them to the commit that caused them, and proposes a verified fix for a human to approve.

## Featured paths

Agentic Vision with OpenCV 5. (Best Use of COOL only if you run the COOL benchmark on Graviton: see `deploy/README.md`.)

## Inspiration

Visual bugs (missing textures, flickering surfaces, frozen frames, exploded meshes, clipped UI, shadow acne, inverted normal maps, missing shadows, physics objects falling through floors, characters sliding in a T-pose, LOD pops, NaN pixels from shaders) are among the most common defects in games, and people find them by watching the screen. Small teams can't afford that. Screenshot comparison tools flag differences but don't explain them, and asking a vision model "does this frame look wrong?" doesn't say which change broke the build.

## What it does

GlitchHunter plays the scenarios of a game's test plan, on any engine (web games in headless Chromium, Godot natively over a small bridge, Unity and Unreal through the same protocol, or just a recording), and reads every frame with 19 OpenCV 5 detectors. Then it investigates like a QA engineer, and each step is chosen from what OpenCV measured:

- **Small, uncertain evidence: move closer.** It ray-casts into the scene, moves the game camera 2-3x closer, recaptures and measures again. A 121-pixel flicker at confidence 0.10 became a 900-pixel z-fighting region at 0.99.
- **Other screen sizes.** UI that fits 16:9 is replayed at 4:3 and 21:9, where an anchored panel runs off the screen.
- **Confirmed: replay.** Identical evidence on every replay means one capture per commit is enough; a random flash seen 1 time in 4 gets 8 captures per commit.
- **Find the cause: visual git bisect.** The OpenCV detector is the pass/fail test. Commits that don't build are skipped, merged branches are searched inside, several commits are probed at once on parallel workers, and findings with the same culprit are one root cause.
- **Fix and check.** It reverts the culprit on a branch, recaptures, and re-surveys every scenario and screen size; it recognizes known bugs that a fix unmasks.
- **Ask.** Nothing is filed until a person approves on the dashboard; rejecting deletes the agent's branch. Every decision and the evidence behind it is logged.

## How we built it

- **OpenCV 5.0.0**, five detector families: classic (HSV segmentation and morphology for missing textures and Unity's error magenta; temporal statistics for z-fighting; frame differencing; luma statistics against WCAG 2.3.1; SSIM, Lab ΔE, contour solidity, Canny and Hough for exploded geometry; the **new OpenCV 5 DNN engine** running PP-OCRv3 + CRNN for UI text); motion (**DIS optical flow**: lag-1 autocorrelation for physics jitter, flow series for stutter, flow-compensated residuals with `remap` for LOD and culling pops, articulation inside tracked blobs for T-posing characters); comparison with the approved build (Lab histograms and a gamma fit for colour shifts, contrast-normalized detail for shadow acne and blurry textures, end-state analysis for objects that fell through the floor); lighting (windowed NCC of band-pass detail for inverted normal maps, brightness ratio with texture and colour tests for missing shadows); pixels (NaN blocks, UI cut by the screen edge). A tone normalization (quantile tone curves plus a robust colour matrix) keeps a global grade from looking like local damage.
- **Agent**: a perception-decision-action loop with 15 tools, a deterministic rule planner, and a Claude Opus 5.5 planner on Amazon Bedrock that uses tool calls and sees evidence images, with fallback to the rule planner on any Bedrock error. Budgets, an append-only trace, and a human approval gate are enforced in the tool layer.
- **Any engine**: one capture interface with a web driver and a native bridge driver (TCP, nine JSON commands). A Godot autoload (tested), a Unity component and batch build script (type-checked against Unity's API with the C# version Unity uses), a drop-in web test API module, a Python reference engine, plus `init` to scaffold a project and `bridge-check` to grade a bridge, determinism first. Video triage covers engines with no integration at all.
- **Interfaces**: FastAPI dashboard with live trace, findings and Approve/Reject; an MCP server exposing 8 OpenCV tools to developer agents such as Claude Code; a CLI.
- **AWS**: CloudFormation stack with a Graviton EC2 instance running the container, S3 for traces, reports and evidence, Amazon Bedrock for the planner, CloudWatch logs plus EMF metrics, a least-privilege IAM role, IMDSv2, and SSM instead of SSH; `aws-doctor` checks each piece.
- **Test bed**: three games with 28 planted bugs at known commits: Crypt Runner (three.js), Ironworks (three.js with a skinned robot, bloom, custom post passes, physics, LOD, a follow camera, and a merged branch) and Hangar 9 (Godot 4 with shadows, CCD physics, an animation player, LOD, normal maps, particles, custom culling bounds, and a commit that doesn't build). The ground truth is kept outside the repos.

## Challenges we ran into

- Several bugs at once: a desaturated grade made a painted line look like a missing object, and a lost shadow looked like one too, until a tone normalization and a "same surface, different light" test separated them.
- OpenCV's DIS optical flow adapts its patch size to the first small image it sees and keeps it, so one engine shared between crops and full frames answered differently depending on what ran before. One engine per image size fixed it and made the flow detectors twice as fast.
- Pops under a moving camera: flow fails at every occlusion edge, so a pop must also be a one-frame step in the difference to the approved build's identical camera path.
- Intermittent bugs that hide during replays, bisects or verification; fixes that unmask other bugs.
- Making native engines deterministic enough to bisect: fixed time steps, seeded particles, input that only lands on advanced frames.

## Accomplishments that we're proud of

- <paste the headline numbers from eval/RESULTS.md per game: recall, culprit accuracy, fix verified, false alarms>
- The same agent and detectors on a web engine and a native Godot build, with lighting bugs (inverted normal maps, missing shadows) found by comparing pixels with the approved build.
- The ablations show the agentic part matters: without zooming the z-fight is lost, without the screen-size sweep the cut-off HUD is lost, and a fixed pipeline finds bugs but no causes or fixes.
- A complete, auditable trace for every decision, and a working human-in-the-loop path (approve and reject).

## What we learned

Tuning detectors matters less than letting the agent gather more evidence when it's unsure, and comparing with the approved build is what makes hard glitches (lighting, physics, animation) visible at all. Determinism is the contract that makes visual bisect possible on any engine.

## What's next

- Run the Unity bridge in a real project and write the Unreal subsystem.
- Run the COOL benchmark on Graviton and the capture workers on arm64.
- Let the LLM planner write targeted patches instead of full reverts (the patch path is implemented and tested).

## Built with

opencv, opencv-dnn, python, three.js, godot, unity, playwright, chromium, fastapi, model-context-protocol, amazon-bedrock, claude, amazon-ec2, aws-graviton, amazon-s3, amazon-cloudwatch, aws-cloudformation, docker

## Links

- Repository: <your repo URL>
- Live endpoint: <DashboardUrl from the CloudFormation outputs>
- Video (5 min max): <YouTube/Vimeo link>, script in `docs/VIDEO_SCRIPT.md`
- Technical report: `docs/REPORT.md`
