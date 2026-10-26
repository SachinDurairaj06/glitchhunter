"""The Bedrock planner inside a real hunt, with a stand-in model that follows the rule policy.

Checks the message plumbing end to end: every tool_use gets exactly one tool_result in the
next user turn, results carry evidence images, and the hunt runs real tools on real frames.
"""

from types import SimpleNamespace as NS

import pytest

from glitchhunter.agent.approvals import ApprovalBroker
from glitchhunter.agent.bedrock import BedrockPlanner
from glitchhunter.agent.loop import Hunt
from glitchhunter.agent.planner import RulePlanner
from glitchhunter.config import load_settings
from glitchhunter.fixture import build_fixture

pytestmark = pytest.mark.slow


class RuleModel:
    """Answers each request with the tool call the rule planner would make."""

    def __init__(self):
        self.inv = None
        self.rule = RulePlanner(soak_runs=0)
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        messages = list(kwargs["messages"])
        self.requests.append(messages)
        _check_pairing(messages)
        action = self.rule.next_action(self.inv)
        block = NS(type="tool_use", id=f"tu{len(self.requests)}", name=action.tool, input=action.args)
        thinking = NS(type="thinking", thinking=f"(stand-in) {action.reason}")
        usage = NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0)
        return NS(content=[thinking, block], stop_reason="tool_use", usage=usage, model="stand-in")


def _check_pairing(messages):
    """Every assistant tool_use must be answered by a tool_result in the following user message."""
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        uses = [b.id for b in msg["content"] if getattr(b, "type", None) == "tool_use"]
        if not uses or i + 1 >= len(messages):
            continue
        answer = messages[i + 1]
        assert answer["role"] == "user"
        ids = [b["tool_use_id"] for b in answer["content"] if b.get("type") == "tool_result"]
        assert ids == uses, f"tool results {ids} do not match tool uses {uses}"


def test_bedrock_planner_drives_a_real_hunt(tmp_path):
    settings = load_settings(work_dir=tmp_path, max_steps=6, game="crypt")
    build_fixture(settings.fixture_dir, settings.fixture_manifest, "crypt")
    from glitchhunter.lab import Lab

    lab = Lab(settings)
    try:
        model = RuleModel()
        planner = BedrockPlanner(settings, client=model)
        hunt = Hunt(lab, settings, ApprovalBroker("auto-approve"), planner=planner)
        model.inv = hunt.inv
        outcome = hunt.run()
    finally:
        lab.close()
    assert outcome["planner"] == "bedrock" and outcome["steps"] == 6
    decisions = [e for e in hunt.trace.events if e["kind"] == "decision"]
    assert decisions[0]["tool"] == "survey" and decisions[0]["reason"].startswith("(stand-in)")
    results = [e for e in hunt.trace.events if e["kind"] == "result"]
    assert all(r["ok"] for r in results)
    # the survey result went back to the model with evidence images attached
    second_request = model.requests[1]
    tool_result = second_request[-1]["content"][0]
    assert any(c["type"] == "image" for c in tool_result["content"])
