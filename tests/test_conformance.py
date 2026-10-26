"""Engine onboarding: `init` scaffolds any engine, `bridge-check` grades a bridge (with the fake engine)."""

import json
import sys
from pathlib import Path

import pytest

from glitchhunter.config import REPO_ROOT
from glitchhunter.conformance import check_build, format_checks
from glitchhunter.project import Project
from glitchhunter.scaffold import ENGINES, init_project

ENGINE = [sys.executable, str(REPO_ROOT / "bridges" / "python" / "fake_engine.py"), "--gh-port", "{port}",
          "--width", "{width}", "--height", "{height}"]


def test_a_conforming_engine_passes_every_check():
    checks = check_build(ENGINE, (320, 180), REPO_ROOT, frames=6, ready_timeout=30)
    by_name = {c.name: c for c in checks}
    assert not [c for c in checks if c.status == "fail"], format_checks(checks)
    assert by_name["determinism"].status == "pass" and by_name["pick"].status == "pass"
    assert "ready for GlitchHunter" in format_checks(checks)


def test_unseeded_randomness_fails_the_determinism_check():
    checks = check_build(ENGINE + ["--unseeded"], (320, 180), REPO_ROOT, frames=6, ready_timeout=30)
    det = next(c for c in checks if c.name == "determinism")
    assert det.status == "fail" and "diverge from frame 0" in det.detail


def test_a_build_that_never_connects_is_reported():
    checks = check_build([sys.executable, "-c", "print('no bridge here')"], (320, 180), REPO_ROOT, ready_timeout=5)
    assert len(checks) == 1 and checks[0].name == "connect" and checks[0].status == "fail"


@pytest.mark.parametrize("engine", ENGINES)
def test_init_scaffolds_a_loadable_test_plan_and_the_bridge(tmp_path, engine):
    lines = init_project(tmp_path / "game", engine, baseline="release-1")
    manifest = json.loads((tmp_path / "game" / "glitchhunter.json").read_text())
    project = Project.from_dict(manifest)
    assert project.baseline == "release-1" and set(project.scenarios) == {"overview", "move"}
    assert project.driver["type"] == ("web" if engine == "web" else "bridge")
    copied = [Path(line.split(None, 1)[1]) for line in lines if line.startswith("copied")]
    assert copied and all(p.exists() for p in copied)
    again = init_project(tmp_path / "game", engine)
    assert all(not line.startswith(("wrote", "copied")) for line in again)  # nothing overwritten


def test_deploy_parameters_carry_what_the_stack_needs():
    from glitchhunter.awsdeploy import stack_parameters

    params = {p["ParameterKey"]: p["ParameterValue"] for p in stack_parameters(
        "https://github.com/x/glitchhunter.git", "203.0.113.7/32", "t" * 24, model="anthropic.claude-opus-5-5")}
    template = (REPO_ROOT / "deploy" / "cloudformation.yaml").read_text()
    assert all(f"  {key}:" in template for key in params)  # every parameter exists in the template
    assert params["AllowedCidr"] == "203.0.113.7/32" and params["Planner"] == "bedrock"
