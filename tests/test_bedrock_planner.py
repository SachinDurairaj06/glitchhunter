"""BedrockPlanner against a scripted fake client: no AWS calls."""

from types import SimpleNamespace as NS

import cv2 as cv
import numpy as np
import pytest

from glitchhunter.agent.bedrock import BedrockPlanner, PlannerError
from glitchhunter.agent.planner import ToolResult
from glitchhunter.agent.state import Finding, Investigation
from glitchhunter.config import load_settings


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the planner keeps appending
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.beta = NS(messages=FakeMessages(responses))


def tool_use(id_, name, args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop="tool_use"):
    usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0)
    return NS(content=list(blocks), stop_reason=stop, usage=usage, model="anthropic.claude-opus-5-5")


def make(responses, tmp_path):
    settings = load_settings(work_dir=tmp_path)
    return BedrockPlanner(settings, client=FakeClient(responses))


def test_maps_tool_use_to_actions_and_returns_results(tmp_path):
    thinking = NS(type="thinking", thinking="Survey first, then look at the weakest evidence.")
    planner = make([
        reply(thinking, tool_use("t1", "survey", {"commit": "main"})),
        reply(tool_use("t2", "zoom_in", {"finding_id": "F2", "factor": 3})),
    ], tmp_path)
    inv = Investigation("h" * 40, "b" * 40)
    a1 = planner.next_action(inv)
    assert (a1.tool, a1.tool_use_id, a1.planner) == ("survey", "t1", "bedrock")
    assert "Survey first" in a1.reason

    img = tmp_path / "ev.png"
    cv.imwrite(str(img), np.zeros((40, 60, 3), np.uint8))
    last = ToolResult("survey", {}, True, {"new_findings": ["F2"]}, "1 new finding", [str(img)], tool_use_id="t1")
    a2 = planner.next_action(inv, last)
    assert a2.tool == "zoom_in" and a2.args["finding_id"] == "F2"

    sent = planner.client.beta.messages.calls[1]
    result_msg = sent["messages"][-1]
    assert result_msg["role"] == "user"
    block = result_msg["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1" and block["is_error"] is False
    assert [c["type"] for c in block["content"]] == ["text", "image"]
    # request shape: adaptive thinking, explicit effort, no sampling params, Bedrock model id
    assert sent["model"] == "anthropic.claude-opus-5-5"
    assert sent["thinking"]["type"] == "adaptive" and "effort" in sent["output_config"]
    assert "temperature" not in sent and "tool_choice" not in sent


def test_parallel_tool_uses_are_queued_and_answered_together(tmp_path):
    planner = make([
        reply(tool_use("a", "replay", {"finding_id": "F1"}), tool_use("b", "replay", {"finding_id": "F3"})),
        reply(NS(type="text", text="done"), stop="end_turn"),
    ], tmp_path)
    inv = Investigation("h" * 40, "b" * 40)
    assert planner.next_action(inv).tool_use_id == "a"
    second = planner.next_action(inv, ToolResult("replay", {}, True, {}, "ok", tool_use_id="a"))
    assert second.tool_use_id == "b"
    final = planner.next_action(inv, ToolResult("replay", {}, False, {}, "boom", error="x", tool_use_id="b"))
    assert final.tool == "finish"
    results = planner.client.beta.messages.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True


def test_nudges_when_model_stops_with_open_findings(tmp_path):
    planner = make([
        reply(NS(type="text", text="I think we're done."), stop="end_turn"),
        reply(tool_use("c", "replay", {"finding_id": "F1"})),
    ], tmp_path)
    inv = Investigation("h" * 40, "b" * 40)
    inv.findings["F1"] = Finding("F1", "missing_texture", "overview", "h" * 40, [1, 1, 5, 5], None, 0.9, 0.9, "", "r")
    action = planner.next_action(inv)
    assert action.tool == "replay"
    nudge = planner.client.beta.messages.calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "F1" in nudge["content"]


def test_refusal_raises_so_the_loop_falls_back(tmp_path):
    planner = make([reply(stop="refusal")], tmp_path)
    with pytest.raises(PlannerError):
        planner.next_action(Investigation("h" * 40, "b" * 40))


def test_transport_errors_become_planner_errors(tmp_path):
    class Broken:
        beta = NS(messages=NS(create=lambda **kw: (_ for _ in ()).throw(ConnectionError("no route"))))

    planner = BedrockPlanner(load_settings(work_dir=tmp_path), client=Broken())
    with pytest.raises(PlannerError, match="no route"):
        planner.next_action(Investigation("h" * 40, "b" * 40))
