"""The per-game test plan (glitchhunter.json), the fixture histories, and bisect over messy histories."""

import json

import pytest

from glitchhunter.agent.tools import _first_bad
from glitchhunter.config import load_settings
from glitchhunter.fixture import build_fixture
from glitchhunter.games import GAME_NAMES, get_game
from glitchhunter.project import DEFAULT_PROJECT, Project, Scenario


def test_camera_path_interpolates_between_keyframes():
    sc = Scenario.from_dict({"name": "fly", "frames": 11, "camera_path": [
        {"frame": 0, "pos": [0, 10, 40], "target": [0, 0, 0], "fov": 50},
        {"frame": 10, "pos": [0, 0, 0], "target": [0, 0, -10], "fov": 70}]})
    assert not sc.zoomable and not sc.camera_static
    mid = sc.camera_at(5)
    assert mid["pos"] == [0, 5, 20] and mid["target"] == [0, 0, -5] and mid["fov"] == 60
    assert sc.camera_at(99)["pos"] == [0, 0, 0]


def test_scenario_defaults():
    sc = Scenario.from_dict({"name": "overview", "frames": 24})
    assert sc.zoomable and sc.camera_static and sc.camera_at(3) is None
    follow = Scenario.from_dict({"name": "follow", "frames": 10, "camera_moves": True})
    assert not follow.zoomable and not follow.camera_static


def test_project_round_trip_and_patch_paths():
    p = Project.from_dict(DEFAULT_PROJECT.to_dict())
    assert list(p.scenarios) == list(DEFAULT_PROJECT.scenarios) and p.baseline == "release-1.0"
    assert p.allows_patch_path("game.js") and p.allows_patch_path("assets/a/b.png")
    assert not p.allows_patch_path("assets/../../etc/passwd") and not p.allows_patch_path("vendor/x.js")
    with pytest.raises(KeyError):
        p.scenario("nope")


@pytest.mark.parametrize("game", GAME_NAMES)
def test_every_bug_is_planted_exactly_once(game):
    spec = get_game(game)
    for name, bug in spec.bugs.items():
        trigger = bug.trigger or name
        hits = [c for c in spec.history if trigger in c.variants or trigger in c.features]
        assert len(hits) == 1, f"{game}:{name} planted {len(hits)} times"
        assert set(bug.scenarios) <= set(spec.scenarios)


def test_first_bad_skips_untestable_commits():
    commits = list("abcdefgh")  # e is the culprit, c does not build
    verdict = {"a": False, "b": False, "c": None, "d": False, "e": True, "f": True, "g": True, "h": True}
    asked = []

    def oracle(sha):
        asked.append(sha)
        return verdict[sha]

    idx, ambiguous = _first_bad(commits, oracle)
    assert commits[idx] == "e" and ambiguous == []
    assert len(asked) <= 4


def test_first_bad_reports_skipped_commits_it_could_not_rule_out():
    commits = list("abcde")  # c does not build and d is bad: the culprit is c or d
    verdict = {"a": False, "b": False, "c": None, "d": True, "e": True}
    idx, ambiguous = _first_bad(commits, verdict.__getitem__)
    assert commits[idx] == "d" and [commits[i] for i in ambiguous] == ["c"]


def test_ironworks_history_has_a_merge_and_a_broken_build(tmp_path):
    """Git-only checks of the advanced fixture (no browser)."""
    from glitchhunter.lab import Lab

    settings = load_settings(work_dir=tmp_path, game="ironworks")
    info = build_fixture(settings.fixture_dir, settings.fixture_manifest, "ironworks")
    lab = Lab(settings)
    try:
        assert lab.project.title == "Ironworks" and lab.project.baseline == "v1.0"
        assert {"overview", "robot_patrol", "crate_drop", "flythrough", "follow_cam"} <= set(lab.project.scenarios)
        anim = info.bugs["anim_retarget"]
        merge = anim["merge_sha"]
        assert len(lab.parents(merge)) == 2
        side = lab.side_commits(merge)
        assert anim["culprit_sha"] in side and len(side) == 3
        assert merge in lab.commits_between(info.baseline, info.head)
        assert anim["culprit_sha"] not in lab.commits_between(info.baseline, info.head)  # first-parent only
        assert len(info.untestable) == 1 and info.untestable[0] in lab.commits_between(info.baseline, info.head)
        manifest = json.loads((settings.fixture_dir / "glitchhunter.json").read_text())
        assert manifest["viewports"] == [[480, 360], [840, 360]]
        shown = lab.show_commit(merge)
        assert shown["merge"] and "Robot_Hips" in shown["diff"]
        # reverting the side-branch commit on main is clean
        fix = lab.make_fix(anim["culprit_sha"], "test-anim")
        assert "Robot_Hips" in fix["diff"]
        lab.delete_branch(fix["branch"])
    finally:
        lab.close()


def test_parallel_probes_find_the_same_culprit_in_fewer_rounds():
    commits = [f"c{i}" for i in range(16)]
    culprit = 11
    rounds = []

    def probe_many(shas):
        rounds.append(shas)
        return [None if sha == "c6" else int(sha[1:]) >= culprit for sha in shas]

    def oracle(sha):
        return probe_many([sha])[0]

    idx, ambiguous = _first_bad(commits, oracle, probe_many, width=2)
    assert commits[idx] == "c11" and ambiguous == []
    single_rounds = []
    _first_bad(commits, lambda sha: single_rounds.append(sha) or int(sha[1:]) >= culprit)
    assert len(rounds) < len(single_rounds)
