# Connecting GlitchHunter to AWS

Everything below needs your AWS account; nothing here has been run yet. Each step ends with a
command that tells you whether it worked.

## 1. Credentials on your machine

Pick one:

- **IAM Identity Center (SSO)**, recommended:

  ```bash
  aws configure sso
  ```

  then `aws sso login --profile <profile>` and set `AWS_PROFILE=<profile>`.

- **Access key** for an IAM user: `aws configure` (key, secret, region).

Set the region where Claude is enabled for your account:

```bash
export AWS_REGION=us-east-1
```

(PowerShell: `$env:AWS_REGION = "us-east-1"`.)

## 2. Model access in Amazon Bedrock

In the Amazon Bedrock console, open **Model access** and request access to **Claude Opus 5.5**
(`anthropic.claude-opus-5-5`). If it is not offered in your region, use another Claude model you
can access and set `GH_BEDROCK_MODEL_ID` to its id.

## 3. Check everything at once

```bash
python -m glitchhunter aws-doctor
```

It checks, in order: region, credentials (STS), the Anthropic models Bedrock lists, one tiny
Claude call through Bedrock, S3 (only if `GH_S3_BUCKET` is set: writes and deletes one test object),
and CloudWatch Logs. Every failed check prints what to fix. When it ends with
`ready for --planner bedrock`, run a hunt with Claude planning:

```bash
python -m glitchhunter hunt --game ironworks --planner bedrock --approval cli
```

If a Bedrock call fails during a hunt, the trace records it and the hunt continues with the
rule planner, so a hunt never stops because of the model.

## 4. Deploy the dashboard on AWS (Graviton)

The quickest way, once the repository is public on GitHub:

```bash
python -m glitchhunter deploy --repo-url https://github.com/<you>/glitchhunter.git
```

It checks the credentials and that the repository can be cloned, restricts the dashboard to your
public IP, generates the API token, creates or updates the stack, waits for it and prints the
dashboard link (`--destroy` deletes it again). The manual equivalent, from
[deploy/README.md](../deploy/README.md):

1. Push the repository to GitHub (the instance clones it).
2. Deploy the stack:

   ```bash
   aws cloudformation deploy --stack-name glitchhunter --template-file deploy/cloudformation.yaml --capabilities CAPABILITY_IAM --parameter-overrides RepoUrl=https://github.com/<you>/glitchhunter.git AllowedCidr=<your.ip>/32 ApiToken=<long-random-token> Planner=bedrock Game=ironworks
   ```

3. Read the outputs (dashboard URL, bucket, instance id):

   ```bash
   aws cloudformation describe-stacks --stack-name glitchhunter --query "Stacks[0].Outputs"
   ```

4. Open `DashboardUrl?token=<ApiToken>`; the first boot builds the image and both test games
   (5-10 minutes; watch the log group in CloudWatch).

The stack gives the instance role exactly what the app uses: `bedrock-mantle:CreateInference`,
S3 writes under `glitchhunter/` in its own bucket, CloudWatch Logs, and SSM Session Manager
(`aws ssm start-session --target <InstanceId>`, no SSH).

## 5. Optional: COOL on Graviton

The benchmark procedure for the Cloud-Optimized OpenCV Library (AWS Marketplace) is in
[deploy/README.md](../deploy/README.md#best-use-of-cool-benchmark-on-graviton).

## Clean up

```bash
aws cloudformation delete-stack --stack-name glitchhunter
```

Empty the artifact bucket first if it holds hunt reports you no longer need.
