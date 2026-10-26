"""Small units: camera math, repeat counts, patch guard, approvals, template rendering."""

import threading
import time

import pytest

from glitchhunter.agent.approvals import ApprovalBroker
from glitchhunter.fixture import render_template
from glitchhunter.games.crypt import HISTORY
from glitchhunter.lab import check_patch, repeats_for, zoom_camera
from glitchhunter.project import DEFAULT_PROJECT


def test_zoom_camera_moves_toward_the_hit_point():
    cam = zoom_camera([3.0, 0.0, -1.0], [0.0, 7.5, 13.0], 3.0)
    assert cam["target"] == [3.0, 0.0, -1.0]
    assert cam["pos"] == pytest.approx([2.0, 2.5, 3.6667], abs=1e-3)


def test_zoom_camera_stays_above_the_floor():
    assert zoom_camera([0, 0, 0], [0, 0.2, 5], 2.0)["pos"][1] == 0.6


def test_repeats_for():
    assert repeats_for(1.0) == 1
    assert repeats_for(0.5) == 5
    assert repeats_for(0.05) == 8  # capped


@pytest.mark.parametrize("path", ["../secrets.txt", "glitchhunter/cli.py", "vendor/three.core.js"])
def test_patch_guard_rejects_files_outside_the_game(path):
    with pytest.raises(ValueError):
        check_patch(f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n", DEFAULT_PROJECT)


def test_patch_guard_allows_game_files():
    check_patch("--- a/game.js\n+++ b/game.js\n@@ -1 +1 @@\n-a\n+b\n", DEFAULT_PROJECT)
    check_patch("--- a/assets/crate.png\n+++ b/assets/crate.png\n", DEFAULT_PROJECT)


def test_approvals_policy_modes():
    yes = ApprovalBroker("auto-approve").request("h", "F1", "file?", ["approve", "reject"])
    assert yes.choice == "approve" and yes.by == "policy:auto-approve"
    no = ApprovalBroker("auto-reject").request("h", "F1", "file?", ["approve", "reject"])
    assert no.choice == "reject"


def test_web_approval_unblocks_the_agent():
    broker = ApprovalBroker("web", timeout_s=10)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("req", broker.request("h", "F2", "file?", ["approve", "reject"])))
    t.start()
    for _ in range(100):
        if broker.pending():
            break
        time.sleep(0.02)
    req_id = broker.pending()[0]["id"]
    broker.decide(req_id, "reject", "not a bug", by="tester")
    t.join(5)
    assert out["req"].choice == "reject" and out["req"].note == "not a bug"
    assert broker.pending() == []


def test_web_approval_times_out_without_a_decision():
    req = ApprovalBroker("web", timeout_s=0.05).request("h", "F3", "file?", ["approve", "reject"])
    assert req.choice is None


TEMPLATE = """// header
//#if __template__
// template-only
//#endif
const a = 1; //#var bug1: const a = 2;
//#if feat
function f() {
  //#if inner
  inner();
  //#endif
  return 1; //#var bug2: return 0;
}
//#endif
"""


def test_render_template_blocks_and_variants():
    base = render_template(TEMPLATE, set(), set())
    assert "template-only" not in base and "function f" not in base and "const a = 1;" in base
    full = render_template(TEMPLATE, {"feat", "inner"}, {"bug1", "bug2"})
    assert "const a = 2;" in full and "inner();" in full and "return 0;" in full
    assert "//#" not in full


def test_history_plants_each_bug_once():
    planted = [v for spec in HISTORY for v in spec.variants] + [f for spec in HISTORY for f in spec.features]
    for name in ("missing_texture", "z_fight", "explosion", "frozen", "hud_overflow", "black_map", "lightning"):
        assert planted.count(name) == 1
