"""Checks that this machine can use the AWS services GlitchHunter needs, one at a time.

Run `python -m glitchhunter aws-doctor` after configuring credentials (aws configure,
aws sso login, or an instance role). Every check prints PASS, WARN or FAIL with what to do
next; nothing is created except one small test object in GH_S3_BUCKET (deleted again).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .config import load_settings


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str
    fix: str = ""


def _client(service: str, region: str):
    import boto3

    return boto3.client(service, region_name=region)


def check_credentials(region: str) -> Check:
    try:
        ident = _client("sts", region).get_caller_identity()
    except Exception as exc:
        return Check("credentials", "FAIL", f"{type(exc).__name__}: {exc}",
                     "Configure credentials: `aws configure` (access key), `aws sso login --profile <p>` "
                     "(then set AWS_PROFILE), or run on an EC2 instance with the stack's instance role.")
    return Check("credentials", "PASS", f"account {ident['Account']} as {ident['Arn']}")


def check_region(region: str) -> Check:
    source = "AWS_REGION" if os.environ.get("AWS_REGION") else (
        "AWS_DEFAULT_REGION" if os.environ.get("AWS_DEFAULT_REGION") else "default")
    note = "" if source != "default" else " (set AWS_REGION to choose another)"
    return Check("region", "PASS" if source != "default" else "WARN", f"{region} from {source}{note}",
                 "" if source != "default" else "export AWS_REGION=us-east-1 (or the region where Bedrock "
                 "has the Claude model enabled for your account)")


def check_bedrock_models(region: str, model: str) -> Check:
    try:
        models = _client("bedrock", region).list_foundation_models(byProvider="Anthropic")["modelSummaries"]
    except Exception as exc:
        return Check("bedrock model list", "WARN", f"{type(exc).__name__}: {exc}",
                     "Needs bedrock:ListFoundationModels (optional; the inference check below is what matters).")
    ids = sorted(m["modelId"] for m in models)
    found = any(model.split(":")[0] in i for i in ids)
    return Check("bedrock model list", "PASS" if found else "WARN",
                 f"{len(ids)} Anthropic models in {region}" + ("" if found else f"; {model} not listed"),
                 "" if found else "Pick a listed model with GH_BEDROCK_MODEL_ID, or request access to it in the "
                                  "Amazon Bedrock console (Model access).")


def check_bedrock_inference(region: str, model: str) -> Check:
    try:
        from anthropic import AnthropicBedrockMantle

        client = AnthropicBedrockMantle(aws_region=region)
        t0 = time.perf_counter()
        msg = client.messages.create(model=model, max_tokens=16,
                                     messages=[{"role": "user", "content": "Reply with the single word OK."}])
    except Exception as exc:
        return Check("bedrock inference", "FAIL", f"{type(exc).__name__}: {exc}",
                     "Enable model access for this model in the Amazon Bedrock console, check the region, and "
                     "allow bedrock-mantle:CreateInference (the CloudFormation role in deploy/ has it). The agent "
                     "still runs with --planner rule.")
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return Check("bedrock inference", "PASS",
                 f"{model} answered {text!r} in {time.perf_counter() - t0:.1f}s ({msg.usage.input_tokens} in / "
                 f"{msg.usage.output_tokens} out tokens)")


def check_s3(region: str, bucket: str | None) -> Check:
    if not bucket:
        return Check("s3 artifacts", "SKIP", "GH_S3_BUCKET is not set: hunt artifacts stay on local disk",
                     "Set GH_S3_BUCKET to the stack's ArtifactBucket output to upload reports and evidence.")
    s3 = _client("s3", region)
    key = f"glitchhunter/doctor/{int(time.time())}.txt"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=b"glitchhunter aws-doctor", ServerSideEncryption="AES256")
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:
        return Check("s3 artifacts", "FAIL", f"{type(exc).__name__}: {exc}",
                     "Check the bucket name and that the role allows s3:PutObject and s3:DeleteObject on it.")
    return Check("s3 artifacts", "PASS", f"wrote and deleted s3://{bucket}/{key}")


def check_cloudwatch(region: str) -> Check:
    group = os.environ.get("GH_LOG_GROUP", "/glitchhunter")
    try:
        groups = _client("logs", region).describe_log_groups(logGroupNamePrefix=group)["logGroups"]
    except Exception as exc:
        return Check("cloudwatch logs", "WARN", f"{type(exc).__name__}: {exc}",
                     "Optional: metrics go to CloudWatch as EMF lines in the container log (GH_EMF=stdout).")
    if not groups:
        return Check("cloudwatch logs", "WARN", f"no log group {group} yet",
                     "Created by the CloudFormation stack; locally, metrics print to stdout with GH_EMF=stdout.")
    return Check("cloudwatch logs", "PASS", f"log group {groups[0]['logGroupName']} exists")


def run(model: str | None = None) -> list[Check]:
    s = load_settings()
    from .agent.bedrock import DEFAULT_MODEL

    model = model or s.bedrock_model_id or DEFAULT_MODEL
    try:
        import boto3  # noqa: F401
    except ImportError:
        return [Check("boto3", "FAIL", "boto3 is not installed", "pip install -r requirements.txt")]
    checks = [check_region(s.aws_region), check_credentials(s.aws_region)]
    if checks[-1].status == "FAIL":
        return checks
    checks += [check_bedrock_models(s.aws_region, model), check_bedrock_inference(s.aws_region, model),
               check_s3(s.aws_region, s.s3_bucket), check_cloudwatch(s.aws_region)]
    return checks


def main(args) -> int:
    checks = run(getattr(args, "model", None))
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:4}] {c.name:<{width}}  {c.detail}")
        if c.fix and c.status != "PASS":
            print(f"{'':{width + 9}}-> {c.fix}")
    failed = [c for c in checks if c.status == "FAIL"]
    print("\nready for --planner bedrock" if not failed else f"\n{len(failed)} check(s) failed")
    return 1 if failed else 0
