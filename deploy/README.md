# Deploying GlitchHunter on AWS

The stack in [`cloudformation.yaml`](cloudformation.yaml) creates:

| Resource | Purpose |
|---|---|
| EC2 Graviton instance (arm64, default `c7g.xlarge`) | Runs the container: OpenCV 5 detectors, headless Chromium captures, the agent loop, the dashboard and HTTP/MCP tool API |
| S3 bucket (SSE, public access blocked, 30-day expiry) | Hunt artifacts: `trace.jsonl`, reports, evidence images (`GH_S3_BUCKET`) |
| IAM instance role | `bedrock-mantle:CreateInference` for Claude on Amazon Bedrock, S3 writes under `glitchhunter/`, CloudWatch Logs, SSM Session Manager (no SSH) |
| CloudWatch log group | Container stdout through the `awslogs` driver; EMF lines become metrics in the `GlitchHunter` namespace (capture time, detector time, tool time, hunt time, findings reported) |
| Security group | Port 8000 open only to `AllowedCidr` |

## Before you deploy (needs your AWS account)

1. Push this repository somewhere the instance can `git clone` (for example a GitHub repo) and note its HTTPS URL.
2. In the Amazon Bedrock console, check model access for **Claude Opus 5.5** (`anthropic.claude-opus-5-5`)
   in your region. If it is not available, deploy with `Planner=rule`, or set `BedrockModelId` to
   another model you can use (for example `anthropic.claude-opus-4-8`). If the Bedrock call fails at
   run time, the agent logs it and continues with the deterministic rule planner.
3. Pick a long random API token (16+ characters).

## Deploy

```bash
aws cloudformation deploy \
  --stack-name glitchhunter \
  --template-file deploy/cloudformation.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      RepoUrl=https://github.com/<you>/glitchhunter.git \
      AllowedCidr=<your.ip.address>/32 \
      ApiToken=<long-random-token> \
      Planner=bedrock
```

```bash
aws cloudformation describe-stacks --stack-name glitchhunter --query "Stacks[0].Outputs"
```

Open `DashboardUrl?token=<ApiToken>`. The first start builds the image and the game history (5–10 minutes).
Start a hunt from the dashboard. Approvals wait for you in the **Needs your decision** panel.

Shell access (no SSH key needed):

```bash
aws ssm start-session --target <InstanceId>
```

## Without CloudFormation (any Docker host)

```bash
docker build -f deploy/Dockerfile -t glitchhunter .
```

```bash
docker run -p 8000:8000 --shm-size 1g -e GH_PLANNER=rule -e GH_API_TOKEN=<token> glitchhunter
```

## Best Use of COOL: benchmark on Graviton

COOL (Cloud-Optimized OpenCV Library) is sold on AWS Marketplace as a Graviton AMI. GlitchHunter's
OpenCV workload is benchmarked natively on that AMI, so COOL's own `cv2` runs the work:

1. Subscribe to the COOL listing that ships an **OpenCV 5** build, and launch it on a Graviton instance.
2. On that instance, clone this repo, then create a venv that can see COOL's system `cv2` and install
   everything except the pip OpenCV wheel:

   ```bash
   python3 -m venv --system-site-packages .venv-cool
   ```

   ```bash
   grep -v opencv requirements.txt | .venv-cool/bin/pip install -r /dev/stdin
   ```

3. Confirm which OpenCV build the venv sees (it should report 5.x and the COOL/KleidiCV build flags):

   ```bash
   .venv-cool/bin/python -m glitchhunter info
   ```

4. Benchmark COOL, then the stock pip OpenCV 5 wheel in a second venv on the same instance:

   ```bash
   GH_OPENCV_DIST=cool .venv-cool/bin/python -m glitchhunter bench --price-per-hour <instance $/h> --out bench/results/cool.json
   ```

   ```bash
   python3 -m venv .venv-pip && .venv-pip/bin/pip install -r requirements.txt
   ```

   ```bash
   GH_OPENCV_DIST=pip .venv-pip/bin/python -m glitchhunter bench --price-per-hour <instance $/h> --out bench/results/pip.json
   ```

5. Compare per-op latency, detector-suite time per run, and cost per 1000 runs:

   ```bash
   .venv-pip/bin/python -m glitchhunter bench --compare bench/results/pip.json bench/results/cool.json
   ```

To run hunts on the COOL build, start the dashboard from the COOL venv. Headless Chromium on arm64
also needs Playwright's browser: `.venv-cool/bin/python -m playwright install --with-deps chromium`.

## Notes and limits

- The Dockerfile and the stack have not been built or deployed yet: the development machine had no
  Docker and no AWS credentials. The template passes `cfn-lint`. Expect to fix small issues on the
  first deploy, and watch the first boot in CloudWatch Logs.
- Headless WebGL on linux/arm64 uses Chromium's SwiftShader fallback (`--enable-unsafe-swiftshader`).
  It has been verified on x86_64 Windows during development, not yet on arm64. The CI workflow runs
  the end-to-end tests on a GitHub arm64 runner, so pushing the repository answers this before you
  deploy. If WebGL fails on arm64, run the capture worker on an x86 instance and keep the OpenCV
  analysis on Graviton.
- The dashboard has no user accounts. Keep `AllowedCidr` narrow and always set `ApiToken`.
- Delete the stack when you are done: `aws cloudformation delete-stack --stack-name glitchhunter`
  (empty the bucket first).
