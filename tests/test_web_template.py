"""The drop-in web test API (bridges/web/glitchhunter-test-api.js) on a minimal three.js game.

A game that only imports the module and passes its canvas, camera, scene and a tick() gets
capture, ray-cast picking and visual bisect, with no GlitchHunter code of its own.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from glitchhunter.config import GAMES_DIR, REPO_ROOT, fs_path, load_settings

INDEX = """<!doctype html>
<html><head><meta charset="utf-8"><style>body{margin:0;background:#000}</style>
<script type="importmap">{"imports": {"three": "./vendor/three.module.js"}}</script></head>
<body><canvas id="game"></canvas><script type="module" src="./game.js"></script></body></html>
"""

GAME = """import * as THREE from 'three';
import { installTestApi, testParams } from './glitchhunter-test-api.js';

const params = testParams();
const W = params.width || 320;
const H = params.height || 180;
const canvas = document.getElementById('game');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x6c7a89);
const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 100);
camera.position.set(0, 2, 6);
camera.lookAt(0, 0.5, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 2.0));
const floor = new THREE.Mesh(new THREE.BoxGeometry(12, 0.2, 12), new THREE.MeshLambertMaterial({ color: 0x556070 }));
floor.name = 'Floor';
floor.position.y = -0.1;
scene.add(floor);
const crateMaterial = new THREE.MeshLambertMaterial({ color: 0x9a6a3a });
const crate = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.4, 1.4), crateMaterial);
crate.name = 'Crate';
crate.position.set(0, 0.7, 0);
scene.add(crate);
const keys = new Set();

function tick() {
  if (keys.has('KeyD')) crate.position.x += 0.1;
  crate.rotation.y += 0.03;
  renderer.render(scene, camera);
}

const api = installTestApi({ THREE, canvas, camera, scene, tick,
  setKey: (code, down) => (down ? keys.add(code) : keys.delete(code)) });
renderer.render(scene, camera);
api.markReady();
if (!params.test) {
  const loop = () => { tick(); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}
"""

MISSING = """const checker = new Uint8Array(8 * 8 * 4).map((_, i) => {
  const p = i >> 2;
  const on = ((p & 7) + (p >> 3)) % 2 === 0;
  return (i & 3) === 3 ? 255 : (i & 3) === 1 ? 0 : (on ? 255 : 0);
});
const placeholder = new THREE.DataTexture(checker, 8, 8);
placeholder.magFilter = THREE.NearestFilter;
placeholder.needsUpdate = true;
const crateMaterial = new THREE.MeshBasicMaterial({ map: placeholder });"""


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-01T08:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-01T08:00:00+00:00"}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def web_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("web-game")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t.invalid")
    (repo / "vendor").mkdir()
    for name in ("three.module.js", "three.core.js", "LICENSE-three.txt"):
        shutil.copyfile(fs_path(GAMES_DIR / "vendor" / "three" / name), repo / "vendor" / name)
    template = REPO_ROOT / "bridges" / "web" / "glitchhunter-test-api.js"
    shutil.copyfile(fs_path(template), repo / "glitchhunter-test-api.js")
    (repo / "index.html").write_text(INDEX, encoding="utf-8")
    manifest = {"name": "cubes", "baseline": "v1.0", "viewport": [320, 180], "entry": "index.html",
                "driver": {"type": "web"}, "patch_paths": ["game.js"],
                "scenarios": [{"name": "spin", "frames": 8, "description": "the crate spins"},
                              {"name": "push", "frames": 8, "events": [[0, "down", "KeyD"]], "motion_expected": True,
                               "description": "hold D: the crate slides"}]}
    (repo / "glitchhunter.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    steps = [("first playable", GAME, "v1.0"),
             ("darker floor", GAME.replace("0x556070", "0x526070"), None),
             ("new crate material", GAME.replace("0x556070", "0x526070").replace(
                 "const crateMaterial = new THREE.MeshLambertMaterial({ color: 0x9a6a3a });", MISSING), None),
             ("tidy", GAME.replace("0x556070", "0x526070").replace(
                 "const crateMaterial = new THREE.MeshLambertMaterial({ color: 0x9a6a3a });", MISSING) + "// tidy\n",
              None)]
    shas = []
    for subject, code, tag in steps:
        (repo / "game.js").write_text(code, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", subject)
        shas.append(_git(repo, "rev-parse", "HEAD"))
        if tag:
            _git(repo, "tag", tag)
    return repo, shas


@pytest.fixture(scope="module")
def lab(web_repo, tmp_path_factory):
    from glitchhunter import service
    from glitchhunter.lab import Lab

    repo, _ = web_repo
    lab = Lab(load_settings(work_dir=tmp_path_factory.mktemp("work"), repo=repo))
    service._lab = lab
    yield lab
    service._lab = None
    lab.close()


@pytest.mark.slow
def test_template_game_captures_keys_and_picks(lab):
    run = lab.capture("v1.0", "push", keep=False)
    assert len(run.frames) == 8 and run.frames[0].shape == (180, 320, 3)
    assert (run.frames[0] != run.frames[-1]).any()
    hit = lab.pick(run, 160, 120)
    assert hit["hit"] and hit["object"] in ("Crate", "Floor")


@pytest.mark.slow
def test_template_game_is_bisected_to_the_material_change(lab, web_repo):
    from glitchhunter import service

    _, shas = web_repo
    run = lab.capture("main", "spin", keep=False)
    dets = lab.inspect(run, ["missing_texture"])
    assert dets
    x, y, w, h = dets[0][0].region
    assert lab.pick(run, x + w / 2, y + h / 2)["object"] == "Crate"
    out = service.bisect("missing_texture", "spin")
    assert out["culprit"] == shas[2][:10]
