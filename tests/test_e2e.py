"""End to end on a real headless browser: build the game history, capture, detect, bisect.

Needs Playwright's Chromium (python -m playwright install chromium). About a minute.
"""

import pytest

from glitchhunter.config import load_settings
from glitchhunter.fixture import FixtureInfo, build_fixture

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    from glitchhunter.lab import Lab

    work = tmp_path_factory.mktemp("work")
    settings = load_settings(work_dir=work, game="crypt")
    build_fixture(settings.fixture_dir, settings.fixture_manifest, "crypt")
    lab = Lab(settings)
    yield lab
    lab.close()


def test_head_shows_planted_bugs_and_baseline_is_clean(lab):
    head = lab.capture("main", "overview", keep=False)
    found = {d.detector for d, _ in lab.inspect(head)}
    assert {"missing_texture", "geometry_explosion", "hud_text_overflow"} <= found
    base = lab.capture("release-1.0", "overview", keep=False)
    assert [d.detector for d, _ in lab.inspect(base)] == []


def test_visual_bisect_finds_the_texture_commit(lab):
    from glitchhunter import service

    service._lab = lab  # reuse this test's Lab
    run = lab.capture("main", "overview")
    dets = lab.inspect(run, ["missing_texture"])
    region = list(dets[0][0].region)
    out = service.bisect("missing_texture", "overview", region=region)
    info = FixtureInfo.load(lab.settings.fixture_manifest)
    assert out["culprit"] == info.bugs["missing_texture"]["culprit_sha"][:10]
    assert out["subject"] == "reorganize texture paths"
    # the diff must be the culprit's own change, not the newest commit's
    assert "+  crate: 'assets/props/crate_wood.png'," in out["diff"]
    assert "Credits" not in out["diff"]


def test_revert_fix_removes_the_glitch(lab):
    info = FixtureInfo.load(lab.settings.fixture_manifest)
    fix = lab.make_fix(info.bugs["black_map"]["culprit_sha"], "test-black-map")
    run = lab.capture(fix["sha"], "map_view", keep=False)
    assert "black_frame" not in {d.detector for d, _ in lab.inspect(run)}
    lab.delete_branch(fix["branch"])


def test_patch_fix_path(lab):
    """The targeted-patch strategy an LLM planner can use instead of a revert."""
    import difflib

    from glitchhunter.capture import git

    info = FixtureInfo.load(lab.settings.fixture_manifest)
    source = git(lab.repo, "show", "main:game.js") + "\n"
    fixed = source.replace("const MAP_CLIP = [100, 10000];", "const MAP_CLIP = [0.1, 100];")
    assert fixed != source
    patch = "".join(difflib.unified_diff(source.splitlines(True), fixed.splitlines(True),
                                         "a/game.js", "b/game.js"))
    fix = lab.make_fix(info.bugs["black_map"]["culprit_sha"], "test-black-map-patch", "patch", patch)
    assert fix["strategy"] == "patch" and "MAP_CLIP = [0.1, 100]" in fix["diff"]
    run = lab.capture(fix["sha"], "map_view", keep=False)
    assert "black_frame" not in {d.detector for d, _ in lab.inspect(run)}
    lab.delete_branch(fix["branch"])
