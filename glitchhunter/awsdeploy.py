"""One-command AWS deployment of the dashboard stack (deploy/cloudformation.yaml) with boto3.

    python -m glitchhunter deploy --repo-url https://github.com/<you>/glitchhunter.git
    python -m glitchhunter deploy --destroy

It checks the credentials, checks that the instance will be able to clone the repository,
opens the dashboard to this machine's public IP only, generates the API token, creates or
updates the stack, waits for it, and prints the dashboard link. The token and the outputs are
saved in the work directory (never in the repository).
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from urllib.request import urlopen

from .config import REPO_ROOT, load_settings

TEMPLATE = REPO_ROOT / "deploy" / "cloudformation.yaml"


def stack_parameters(repo_url: str, cidr: str, token: str, planner: str = "bedrock", game: str = "ironworks",
                     instance_type: str = "c7g.xlarge", model: str | None = None, ref: str = "main") -> list[dict]:
    params = {"RepoUrl": repo_url, "RepoRef": ref, "AllowedCidr": cidr, "ApiToken": token, "Planner": planner,
              "Game": game, "InstanceType": instance_type}
    if model:
        params["BedrockModelId"] = model
    return [{"ParameterKey": k, "ParameterValue": v} for k, v in params.items()]


def public_cidr() -> str:
    ip = urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()
    return f"{ip}/32"


def repo_is_cloneable(url: str) -> bool:
    """The instance clones anonymously: the repository must be public (git ls-remote without credentials)."""
    env = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_ASKPASS": "echo"}
    result = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", url, "HEAD"], capture_output=True,
                            text=True, timeout=60, env={**os.environ, **env}, check=False)
    return result.returncode == 0


def _say(msg: str) -> None:
    print(msg, flush=True)


def deploy(args) -> int:
    import boto3

    region = args.region
    stack = args.stack
    session = boto3.Session(region_name=region)
    try:
        ident = session.client("sts").get_caller_identity()
    except Exception as exc:  # noqa: BLE001
        _say(f"FAIL credentials: {type(exc).__name__}: {exc}")
        _say("  Configure them first: `aws configure` (an access key) or `aws sso login`; see docs/AWS_SETUP.md.")
        return 1
    _say(f"PASS credentials: account {ident['Account']} ({ident['Arn']}), region {region}")
    cf = session.client("cloudformation")
    state_file = load_settings().work_dir / f"aws-{stack}.json"

    if args.destroy:
        cf.delete_stack(StackName=stack)
        _say(f"deleting stack {stack} (empty its artifact bucket first if deletion stops on it)...")
        cf.get_waiter("stack_delete_complete").wait(StackName=stack)
        _say("deleted")
        return 0

    if not args.repo_url:
        _say("FAIL give --repo-url https://github.com/<you>/glitchhunter.git (the instance clones it)")
        return 2
    if not repo_is_cloneable(args.repo_url):
        _say(f"FAIL {args.repo_url} cannot be cloned anonymously: push the repository and make it public first")
        return 1
    _say(f"PASS repository {args.repo_url} is public")
    cidr = args.cidr or public_cidr()
    token = args.token or secrets.token_urlsafe(24)
    params = stack_parameters(args.repo_url, cidr, token, args.planner, args.game, args.instance_type, args.model,
                              args.ref)
    body = TEMPLATE.read_text(encoding="utf-8")
    cf.validate_template(TemplateBody=body)
    _say(f"PASS template is valid; the dashboard will only answer {cidr}")
    try:
        cf.describe_stacks(StackName=stack)
        exists = True
    except cf.exceptions.ClientError:
        exists = False
    t0 = time.time()
    if exists:
        try:
            cf.update_stack(StackName=stack, TemplateBody=body, Parameters=params, Capabilities=["CAPABILITY_IAM"])
        except cf.exceptions.ClientError as exc:
            if "No updates are to be performed" not in str(exc):
                raise
            _say("stack is already up to date")
        else:
            _say(f"updating stack {stack} (a few minutes)...")
            cf.get_waiter("stack_update_complete").wait(StackName=stack, WaiterConfig={"Delay": 15, "MaxAttempts": 120})
    else:
        cf.create_stack(StackName=stack, TemplateBody=body, Parameters=params, Capabilities=["CAPABILITY_IAM"],
                        Tags=[{"Key": "project", "Value": "glitchhunter"}])
        _say(f"creating stack {stack} (a few minutes)...")
        cf.get_waiter("stack_create_complete").wait(StackName=stack, WaiterConfig={"Delay": 15, "MaxAttempts": 120})
    outputs = {o["OutputKey"]: o["OutputValue"]
               for o in cf.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])}
    url = outputs.get("DashboardUrl", "")
    state_file.write_text(json.dumps({"stack": stack, "region": region, "token": token, "outputs": outputs}, indent=2),
                          encoding="utf-8")
    _say(f"done in {round(time.time() - t0)}s")
    for k, v in outputs.items():
        _say(f"  {k}: {v}")
    _say(f"Dashboard: {url}?token={token}")
    _say("The instance builds the container and the game on first boot (5-10 minutes); watch the log group.")
    _say(f"Token and outputs saved to {state_file}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("deploy", help="deploy the dashboard stack to AWS (Graviton, S3, Bedrock, CloudWatch)")
    p.add_argument("--repo-url", default=None, help="public HTTPS git URL the instance clones")
    p.add_argument("--ref", default="main")
    p.add_argument("--region", default=None, help="default: AWS_REGION or us-east-1")
    p.add_argument("--stack", default="glitchhunter")
    p.add_argument("--planner", choices=["bedrock", "rule"], default="bedrock")
    p.add_argument("--game", choices=["ironworks", "crypt"], default="ironworks")
    p.add_argument("--instance-type", default="c7g.xlarge")
    p.add_argument("--model", default=None, help="Bedrock model id (default: the template's)")
    p.add_argument("--cidr", default=None, help="who may open the dashboard (default: this machine's public IP /32)")
    p.add_argument("--token", default=None, help="API token (default: generated)")
    p.add_argument("--destroy", action="store_true", help="delete the stack")
    p.set_defaults(fn=_run)


def _run(args) -> int:
    args.region = args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return deploy(args)


__all__ = ["add_parser", "deploy", "public_cidr", "repo_is_cloneable", "stack_parameters"]
