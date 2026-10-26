"""End to end on the advanced test game: real captures, harder detectors, merge-aware bisect.

Needs Playwright's Chromium. A few minutes.
"""

import pytest

from glitchhunter.config import load_settings
from glitchhunter.fixture import build_fixture

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def iw(tmp_path_factory):
    from glitchhunter import service
    from glitchhunter.lab import Lab

    work = tmp_path_factory.mktemp("iw")
    settings = load_settings(work_dir=work, game="ironworks")
    info = build_fixture(settings.fixture_dir, settings.fixture_manifest, "ironworks")
    lab = Lab(settings)
    service._lab = lab
    yield lab, info
    service._lab = None
    lab.close()


def test_head_shows_the_harder_glitches_and_the_baseline_is_clean(iw):
    lab, info = iw
    head = lab.capture("main", "overview", keep=False)
    found = {d.detector for d, _ in lab.inspect(head)}
    assert {"color_shift", "shadow_acne", "detail_loss", "physics_jitter", "nan_pixels"} <= found
    base = lab.capture("v1.0", "overview", keep=False)
    assert [d.detector for d, _ in lab.inspect(base)] == []


def test_screen_size_and_motion_glitches(iw):
    lab, info = iw
    narrow = lab.capture("main", "overview", keep=False, viewport=(480, 360))
    assert "ui_offscreen" in {d.detector for d, _ in lab.inspect(narrow, ["ui_offscreen"])}
    follow = lab.capture("main", "follow_cam", keep=False)
    assert "stutter" in {d.detector for d, _ in lab.inspect(follow, ["stutter"])}
    fly = lab.capture("main", "flythrough", keep=False)
    assert "pop_in" in {d.detector for d, _ in lab.inspect(fly, ["pop_in"])}


def test_bisect_descends_into_the_merged_branch(iw):
    from glitchhunter import service

    lab, info = iw
    out = service.bisect("frozen_animation", "robot_patrol")
    bug = info.bugs["anim_retarget"]
    assert out["culprit"] == bug["culprit_sha"][:10]
    assert out["via_merge"] == bug["merge_sha"][:10]


def test_bisect_skips_the_commit_that_does_not_build(iw):
    from glitchhunter import service
    from glitchhunter.lab import BuildError

    lab, info = iw
    with pytest.raises(BuildError):
        lab.capture(info.untestable[0], "overview", keep=False)
    out = service.bisect("physics_jitter", "overview")
    assert out["culprit"] == info.bugs["jitter"]["culprit_sha"][:10]
    assert set(out["skipped"]) <= {info.untestable[0][:10]}
