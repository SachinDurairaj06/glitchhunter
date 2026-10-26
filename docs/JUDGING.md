# Where to find the evidence for each judging criterion

## Agentic Vision Award

| Criterion | Where to look |
|---|---|
| **Substantive OpenCV 5 + agent (30%)** | 19 detectors in five families in [`glitchhunter/vision/`](../glitchhunter/vision): HSV segmentation, temporal statistics, morphology, connected components, SSIM, Lab ΔE and histograms, contours/convex hulls, Canny/Hough, **DIS optical flow** (jitter, stutter, pops, frozen animation), flow-compensated residuals with `remap`, windowed NCC of band-pass detail (inverted normal maps), tone curves and a robust colour matrix (tone normalization), and OpenCV 5 DNN text detection + recognition with the new engine ([`textdnn.py`](../glitchhunter/textdnn.py)). Each detector is an agent tool input; [ARCHITECTURE.md](ARCHITECTURE.md#where-each-decision-comes-from) maps each measurement to the action it triggers. |
| **Orchestration & autonomy (25%)** | Perception-decision-action loop in [`agent/loop.py`](../glitchhunter/agent/loop.py); 15 tools in [`agent/tools.py`](../glitchhunter/agent/tools.py); rule policy in [`agent/planner.py`](../glitchhunter/agent/planner.py); Claude-on-Bedrock planner in [`agent/bedrock.py`](../glitchhunter/agent/bedrock.py). Active perception (camera zoom by ray-casting), screen-size sweep, replay and determinism checks, bisect repeats from the reproduction rate, bisect that skips broken builds and descends into merged branches, root-cause deduplication, re-bisect when a flaky fix fails, unmasked-vs-regression reasoning. |
| **Task effectiveness (20%)** | [eval/RESULTS.md](../eval/RESULTS.md): three games, 28 planted bugs, recall, culprit accuracy, verified fixes, false alarms on clean builds, and ablations (no zoom, no sweep, fixed pipeline). |
| **Failure handling & human control (15%)** | Approval gate enforced in the tool layer (`file_report` refuses without approval; rejecting deletes the fix branch): [`agent/approvals.py`](../glitchhunter/agent/approvals.py), dashboard **Needs your decision** panel. Budgets, consecutive-error stop, Bedrock-to-rule fallback, commits that do not build skipped and reported, flaky-bug handling, escalation when bisect ends misbehave: [REPORT.md section 5](REPORT.md#5-agent-design). A real rejection is in [sample-hunt/SUMMARY.md](sample-hunt/SUMMARY.md). |
| **User experience (10%)** | Dashboard (live trace, finding cards with evidence, approvals): `python -m glitchhunter serve`. Bug reports with images. Onboarding any engine: `init` and `bridge-check` ([ENGINES.md](ENGINES.md)). |

Required extras: agent workflow diagram ([ARCHITECTURE.md](ARCHITECTURE.md#agent-workflow-perception---decision---action)); traces where OpenCV output changes the next action ([REPORT.md section 6](REPORT.md#6-evidence-that-vision-changes-the-next-action)); evaluation of task success, failure handling, observability and human control ([REPORT.md sections 5 and 7](REPORT.md)).

## Overall rubric

| Criterion | Where to look |
|---|---|
| Technical execution (30%) | Code above; 110 tests ([`tests/`](../tests)), 11 of which drive a real headless browser; a Godot 4 game driven natively over the bridge; [REPORT.md](REPORT.md) |
| Innovation (20%) | Visual git bisect with an OpenCV detector as the test (skip-aware, merge-aware, k commits per round); camera re-aiming by ray-cast for low-confidence evidence; capture counts per commit from the observed reproduction rate; lighting checks (inverted normal maps, missing shadows) against the approved build; telling unmasked bugs from regressions |
| Real-world impact (20%) | Works with any engine: web, Godot (tested), Unity and Unreal through one documented protocol, any recording through video triage ([ENGINES.md](ENGINES.md)); developer agents via MCP ([MCP.md](MCP.md)); photosensitivity (WCAG 2.3.1) flash detection |
| User experience (10%) | Dashboard, reports, `init` / `bridge-check` onboarding |
| Documentation (10%) | [README](../README.md), [REPORT.md](REPORT.md), [ARCHITECTURE.md](ARCHITECTURE.md), [ENGINES.md](ENGINES.md), [AWS_SETUP.md](AWS_SETUP.md), [deploy/README.md](../deploy/README.md), [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) |
| Responsible cloud delivery (10%) | [deploy/cloudformation.yaml](../deploy/cloudformation.yaml): Graviton EC2, S3 (encrypted, private, 30-day expiry), least-privilege IAM, IMDSv2, SSM instead of SSH, CIDR-restricted port, bearer token, CloudWatch logs + EMF metrics; `aws-doctor`; [REPORT.md section 9](REPORT.md#9-responsible-use) |
