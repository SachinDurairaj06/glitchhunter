"""The rule planner turns visual evidence into the next tool call."""

from glitchhunter.agent.planner import RulePlanner, bisect_repeats
from glitchhunter.agent.state import Finding, Investigation


def finding(**kw) -> Finding:
    base = dict(id="F1", detector="zfight_flicker", scenario="overview", commit="c" * 40, region=[10, 10, 12, 12],
                camera=None, score=0.7, confidence=0.1, note="", run_id="r1")
    base.update(kw)
    return Finding(**base)


def inv_with(*findings) -> Investigation:
    inv = Investigation("h" * 40, "b" * 40)
    inv.surveyed["h" * 40] = {}
    for f in findings:
        inv.findings[f.id] = f
    return inv


def test_survey_first():
    assert RulePlanner().next_action(Investigation("h" * 40, "b" * 40)).tool == "survey"


def test_low_confidence_small_region_zooms_in():
    a = RulePlanner().next_action(inv_with(finding()))
    assert a.tool == "zoom_in" and a.args["factor"] == 3.0
    assert a.evidence["confidence"] == 0.1 and "0.10 < 0.6" in a.reason


def test_screen_space_evidence_is_never_zoomed():
    f = finding(detector="hud_text_overflow", confidence=0.5)
    assert RulePlanner().next_action(inv_with(f)).tool == "replay"


def test_weak_after_zooming_is_dismissed():
    f = finding(confidence=0.2, zooms=2)
    a = RulePlanner().next_action(inv_with(f))
    assert a.tool == "dismiss"


def test_confident_finding_is_replayed():
    assert RulePlanner().next_action(inv_with(finding(confidence=0.9))).tool == "replay"


def test_deterministic_bisects_with_one_capture():
    f = finding(confidence=0.9, status="confirmed", repro={"runs": 3, "hits": 3, "rate": 1.0, "deterministic": True})
    a = RulePlanner().next_action(inv_with(f))
    assert a.tool == "bisect" and a.args["repeats"] == 1


def test_flaky_bisects_with_enough_repeats():
    f = finding(detector="flash", confidence=0.9, status="flaky",
                repro={"runs": 4, "hits": 2, "rate": 0.5, "deterministic": False})
    assert bisect_repeats(f) == 5  # Beta(1,1) estimate 0.5 -> 5 captures keep the miss chance < 5%
    a = RulePlanner().next_action(inv_with(f))
    assert a.tool == "bisect" and a.args["repeats"] == 5


def test_weak_unreproduced_evidence_is_dismissed():
    f = finding(confidence=0.5, status="not_reproduced", repro={"runs": 3, "hits": 0, "rate": 0, "deterministic": False})
    assert RulePlanner().next_action(inv_with(f)).tool == "dismiss"


def test_strong_unreproduced_evidence_gets_more_replays_first():
    f = finding(detector="flash", confidence=0.9, status="not_reproduced",
                repro={"runs": 4, "hits": 0, "rate": 0, "deterministic": False})
    a = RulePlanner().next_action(inv_with(f))
    assert a.tool == "replay" and a.args["n"] == 8
    f.repro["runs"] = 12
    assert RulePlanner().next_action(inv_with(f)).tool == "dismiss"


def test_fix_flow_and_approval_gate():
    culprit = {"sha": "d" * 40, "subject": "snap rune decal flush to the floor", "diff": ""}
    f = finding(confidence=0.9, status="bisected", culprit=culprit)
    p = RulePlanner()
    assert p.next_action(inv_with(f)).tool == "propose_fix"
    f.status = "fix_proposed"
    assert p.next_action(inv_with(f)).tool == "verify_fix"
    f.status, f.verification = "fix_verified", {"passed": True, "reason": "ok", "still_visible": 0}
    a = p.next_action(inv_with(f))
    assert a.tool == "request_human" and a.args["options"] == ["approve", "reject"]
    f.status, f.approval = "approved", {"by": "web"}
    assert p.next_action(inv_with(f)).tool == "file_report"


def test_failed_fix_on_flaky_glitch_rebisects_with_more_repeats():
    f = finding(detector="flash", status="fix_failed", repro={"runs": 3, "hits": 2, "deterministic": False},
                bisect={"repeats": 3}, bisect_attempts=1,
                verification={"passed": False, "reason": "still visible", "still_visible": 1})
    a = RulePlanner().next_action(inv_with(f))
    assert a.tool == "bisect" and a.args["repeats"] == 6


def test_failed_fix_without_visible_glitch_goes_to_a_human():
    f = finding(status="fix_failed", repro={"runs": 3, "hits": 3, "deterministic": True},
                bisect={"repeats": 1}, bisect_attempts=1,
                verification={"passed": False, "reason": "new glitches", "still_visible": 0})
    assert RulePlanner().next_action(inv_with(f)).tool == "request_human"


def test_soak_then_finish_when_everything_is_closed():
    f = finding(status="reported")
    inv = inv_with(f)
    p = RulePlanner()
    assert p.next_action(inv).tool == "soak"
    inv.soaked = True
    assert p.next_action(inv).tool == "finish"
