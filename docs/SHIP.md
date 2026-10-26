# Shipping checklist (deadline: Oct 26, 11:59 PM PT = Oct 27, 12:29 PM IST)

Everything that could be done without your accounts is done: code, tests, evaluation, docs,
dashboard, deployment template and a one-command deploy. These steps need you (your GitHub,
AWS and Devpost logins). Run the commands from the `glitchhunter` folder; on Windows use
`.venv\Scripts\python` (or the venv you already have) for `python`.

## 1. Publish the repository (5 minutes)

The code is committed locally (`git log`). The AWS instance clones the repository and the
judges need a link, so it must be public:

```bash
gh repo create glitchhunter --public --source . --remote origin --push
```

(Or create an empty repo on github.com and `git remote add origin <url>` then `git push -u origin main`.)
CI starts on push: tests on Linux x86_64 and arm64, a short hunt, the benchmark, the Docker image
on arm64 and the Unity bridge type check.

## 2. AWS credentials and Claude on Bedrock (10 minutes)

1. Create an access key for your IAM user (IAM console > Users > Security credentials), then:

   ```bash
   aws configure
   ```

   (No AWS CLI installed? `python -m pip install awscli` puts one in the venv; run it as `python -m awscli configure`.)
2. Amazon Bedrock console > Model access: enable **Claude Opus 5.5** in your region (us-east-1 by
   default). If it is not offered, use any Claude model you can enable and pass `--model <id>` below.
3. Check everything at once:

   ```bash
   python -m glitchhunter aws-doctor
   ```

   It should end with `ready for --planner bedrock`.

## 3. Deploy on AWS Graviton (15 minutes, mostly waiting)

```bash
python -m glitchhunter deploy --repo-url https://github.com/<you>/glitchhunter.git
```

It checks credentials and that the repo is public, opens the dashboard to your IP only,
generates the API token, creates the CloudFormation stack (Graviton EC2, S3, IAM role for
Bedrock, CloudWatch logs + metrics), waits, and prints the dashboard link with the token. The
instance then builds the container and the test game (5-10 minutes). Cost: a c7g.xlarge is about
$0.15 per hour; delete everything afterwards with `python -m glitchhunter deploy --destroy`.

Open the printed `DashboardUrl?token=...`, press **Start hunt** with the planner set to
**Claude on Amazon Bedrock**, and approve findings in **Needs your decision**. That run (Claude
planning on Bedrock, artifacts in S3, metrics in CloudWatch) is your AWS evidence: take
screenshots of the dashboard, the S3 bucket and the CloudWatch log group.

## 4. Demo video (under 5 minutes)

Script: [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md). Record locally with
`python -m glitchhunter serve --game hangar` (Godot) and the deployed dashboard. Lead with the 3D
and physics bugs (pallet through the pad, T-posing robot, vanishing truck, upside-down wall
relief, missing shadow), then the bisect strip and culprit diff, then approve/reject, then the
scoreboard. Upload to YouTube (unlisted is fine).

## 5. Devpost

Copy the fields from [SUBMISSION.md](SUBMISSION.md) (numbers already filled in from
`eval/RESULTS.md`), add the repository link, the video link and the dashboard URL, choose the
Agentic Vision path, and submit before the deadline.

## If something fails

| Symptom | Fix |
|---|---|
| `deploy` says the repo cannot be cloned | the repository is still private, or the URL has a typo |
| `aws-doctor` FAIL on Bedrock | model access not granted yet in that region: enable it, or `--model` another Claude |
| Stack stuck on the instance | CloudWatch log group `/glitchhunter/...` shows the boot; the first build takes a while |
| Dashboard does not open | your IP changed: redeploy (it updates the allowed CIDR) |
| No time for AWS | the local dashboard and the evaluation still show the whole agent; say in the submission that the stack is ready (`deploy/`) and what blocked the live run |
