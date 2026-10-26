// Crypt Runner: a tiny three.js dungeon scene.
//#if __template__
//
// This file is a template. glitchhunter/fixture.py renders it into a git
// history: `//#if <feature>` ... `//#endif` blocks appear once a feature is
// committed, and a line ending in `//#var <name>: <code>` switches to the
// alternate code from the commit that introduces that change (most of them
// plant a bug). Rendered commits contain plain JS without these markers.
//#endif
//#if credits
// Credits: textures are generated procedurally by the build script.
//#endif
import * as THREE from './vendor/three.module.js';

const W = 640;
const H = 360;
const DT = 1 / 30;
const params = new URLSearchParams(location.search);
const TEST_MODE = params.get('test') === '1';

const canvas = document.getElementById('game');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  preserveDrawingBuffer: true,
  powerPreference: 'low-power',
});
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.autoClear = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
scene.fog = new THREE.Fog(0x0b0d12, 18, 42);

const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 200);
const DEFAULT_CAM = { pos: [0, 7.5, 13], target: [0, 0.6, 0], fov: 55 };
const camTarget = new THREE.Vector3();
let activeCamera = camera;

const errors = [];
const keys = new Set();
const pending = [];
const loader = new THREE.TextureLoader();
const MISSING = makeMissingTexture();

const TEX = {
  floor: 'assets/floor_tiles.png',
  wall: 'assets/wall_bricks.png',
  crate: 'assets/crate_wood.png', //#var missing_texture: crate: 'assets/props/crate_wood.png',
  rune: 'assets/rune.png',
};

function makeMissingTexture() {
  // Engine-style placeholder: a magenta/black checker shown when an asset fails to load.
  const size = 8;
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const on = (x + y) % 2 === 0;
      data[i] = on ? 255 : 0;
      data[i + 1] = 0;
      data[i + 2] = on ? 255 : 0;
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function loadTex(url, material, repeat) {
  const p = new Promise((resolve) => {
    loader.load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        if (repeat) {
          tex.wrapS = THREE.RepeatWrapping;
          tex.wrapT = THREE.RepeatWrapping;
          tex.repeat.set(repeat[0], repeat[1]);
        }
        resolve(tex);
      },
      undefined,
      () => {
        errors.push({ type: 'asset_missing', url });
        resolve(MISSING);
      },
    );
  }).then((tex) => {
    material.map = tex;
    material.needsUpdate = true;
  });
  pending.push(p);
}

// --- room -------------------------------------------------------------------
const floorMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95 });
loadTex(TEX.floor, floorMat, [5, 5]);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), floorMat);
floor.rotation.x = -Math.PI / 2;
floor.name = 'floor';
scene.add(floor);

const wallMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
loadTex(TEX.wall, wallMat, [6, 2]);
const backWall = new THREE.Mesh(new THREE.PlaneGeometry(20, 6), wallMat);
backWall.position.set(0, 3, -10);
backWall.name = 'back_wall';
scene.add(backWall);
const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(20, 6), wallMat);
leftWall.position.set(-10, 3, 0);
leftWall.rotation.y = Math.PI / 2;
leftWall.name = 'left_wall';
scene.add(leftWall);
const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(20, 6), wallMat);
rightWall.position.set(10, 3, 0);
rightWall.rotation.y = -Math.PI / 2;
rightWall.name = 'right_wall';
scene.add(rightWall);

scene.add(new THREE.HemisphereLight(0x9fb4d8, 0x2a2018, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(5, 10, 7);
scene.add(sun);
const TORCH_COLOR = 0xffa040; //#var torch_tweak: const TORCH_COLOR = 0xff9a3c;
const torch = new THREE.PointLight(TORCH_COLOR, 18, 16, 2);
torch.position.set(-4, 3.2, -6);
scene.add(torch);

//#if crate
// --- crate ------------------------------------------------------------------
const crateMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8 });
loadTex(TEX.crate, crateMat);
const crate = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.4, 1.4), crateMat);
crate.position.set(-3.2, 0.7, -1.5);
crate.rotation.y = 0.35;
crate.name = 'crate';
scene.add(crate);
//#endif

//#if pillar_decal
// --- pillar and rune decal --------------------------------------------------
const pillar = new THREE.Mesh(
  new THREE.CylinderGeometry(0.6, 0.7, 4.5, 24),
  new THREE.MeshStandardMaterial({ color: 0x8a8f99, roughness: 0.9 }),
);
pillar.position.set(3.2, 2.25, -3.2);
pillar.name = 'pillar';
scene.add(pillar);

const runeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, emissive: 0x0a3a3a });
loadTex(TEX.rune, runeMat);
const rune = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.4), runeMat);
rune.rotation.x = -Math.PI / 2;
rune.position.set(3.2, 0.02, -1.4); //#var z_fight: rune.position.set(3.2, 0.0, -1.4); // sit flush on the floor
rune.name = 'rune_decal';
scene.add(rune);
//#endif

//#if rock
// --- rock prop ----------------------------------------------------------------
function hashNoise(x, y, z) {
  const h = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
  return (h - Math.floor(h)) * 2 - 1;
}

function displaceRock(geometry, amp) {
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = hashNoise(
      Math.round(v.x * 1000) / 1000,
      Math.round(v.y * 1000) / 1000,
      Math.round(v.z * 1000) / 1000,
    );
    v.multiplyScalar(1 + amp * n);
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  geometry.computeVertexNormals();
}

const ROCK_NOISE_AMP = 0.18; //#var explosion: const ROCK_NOISE_AMP = 1.8; // bolder silhouette
const rockGeo = new THREE.IcosahedronGeometry(0.9, 3);
displaceRock(rockGeo, ROCK_NOISE_AMP);
const rock = new THREE.Mesh(
  rockGeo,
  new THREE.MeshStandardMaterial({ color: 0x6b5a4a, roughness: 1.0, flatShading: true }),
);
rock.position.set(-5.5, 0.75, 4.0);
rock.name = 'rock';
scene.add(rock);
//#endif

//#if player_gem
// --- player and gem -----------------------------------------------------------
const player = new THREE.Mesh(
  new THREE.CapsuleGeometry(0.35, 0.8, 6, 12),
  new THREE.MeshStandardMaterial({ color: 0x3d7bd9, roughness: 0.5 }),
);
player.position.set(0, 0.75, 5.5);
player.name = 'player';
scene.add(player);

const gem = new THREE.Mesh(
  new THREE.OctahedronGeometry(0.4),
  new THREE.MeshStandardMaterial({ color: 0x46e08a, emissive: 0x0f5a2c, roughness: 0.3 }),
);
gem.position.set(0, 0.9, 1.5);
gem.name = 'gem';
scene.add(gem);
let score = 0;

function updatePlayer() {
  const step = 3.0 * DT;
  if (keys.has('KeyW')) player.position.z -= step;
  if (keys.has('KeyS')) player.position.z += step;
  if (keys.has('KeyA')) player.position.x -= step;
  if (keys.has('KeyD')) player.position.x += step;
  player.position.x = THREE.MathUtils.clamp(player.position.x, -9, 9);
  player.position.z = THREE.MathUtils.clamp(player.position.z, -9, 9);
  gem.rotation.y += 0.08;
  if (gem.visible && player.position.distanceTo(gem.position) < 0.9) {
    gem.visible = false;
    score += 10; //#var frozen: score += gem.userData.reward.points;
    //#if hud
    hud.draw();
    //#endif
  }
}
//#endif

//#if hud
// --- HUD ----------------------------------------------------------------------
const hudCanvas = document.createElement('canvas');
hudCanvas.width = 256;
hudCanvas.height = 64;
const hudCtx = hudCanvas.getContext('2d');
const hudTex = new THREE.CanvasTexture(hudCanvas);
hudTex.colorSpace = THREE.SRGBColorSpace;
hudTex.minFilter = THREE.LinearFilter;
hudTex.generateMipmaps = false;
const hudScene = new THREE.Scene();
const hudCam = new THREE.OrthographicCamera(0, W, H, 0, -1, 1);
const hudQuad = new THREE.Mesh(
  new THREE.PlaneGeometry(256, 64),
  new THREE.MeshBasicMaterial({ map: hudTex, transparent: true, depthTest: false }),
);
hudQuad.position.set(16 + 128, H - 16 - 32, 0);
hudScene.add(hudQuad);
const LABELS = { hp: 'HP', score: 'PTS' }; //#var hud_overflow: const LABELS = { hp: 'Hit Points', score: 'Score Total' };

const hud = {
  hp: 100,
  draw() {
    const c = hudCtx;
    c.clearRect(0, 0, 256, 64);
    c.fillStyle = 'rgba(27, 42, 74, 0.92)';
    c.fillRect(2, 2, 168, 60);
    c.strokeStyle = '#6fa8ff';
    c.lineWidth = 3;
    c.strokeRect(2, 2, 168, 60);
    c.fillStyle = '#ffffff';
    c.font = 'bold 20px Arial, Helvetica, sans-serif';
    c.textBaseline = 'middle';
    const pts = typeof score === 'number' ? score : 0;
    c.fillText(`${LABELS.hp} ${this.hp}   ${LABELS.score} ${pts}`, 12, 33);
    hudTex.needsUpdate = true;
  },
};
hud.draw();
//#endif

//#if jitter
// --- TAA-style sub-pixel camera jitter ----------------------------------------
const HALTON = [
  [0.5, 0.333], [0.25, 0.667], [0.75, 0.111], [0.125, 0.444],
  [0.625, 0.778], [0.375, 0.222], [0.875, 0.556], [0.0625, 0.889],
];

function applyJitter(cam, frameIndex) {
  const [hx, hy] = HALTON[frameIndex % HALTON.length];
  cam.setViewOffset(W, H, hx - 0.5, hy - 0.5, W, H);
}
//#endif

//#if mapview
// --- map view -----------------------------------------------------------------
const MAP_CLIP = [0.1, 100]; //#var black_map: const MAP_CLIP = [100, 10000]; // draw distance in cm
const mapCam = new THREE.OrthographicCamera(-12, 12, 6.75, -6.75, MAP_CLIP[0], MAP_CLIP[1]);
mapCam.position.set(0, 30, 0);
mapCam.up.set(0, 0, -1);
mapCam.lookAt(0, 0, 0);

function toggleMap() {
  activeCamera = activeCamera === mapCam ? camera : mapCam;
}
//#endif

//#if lightning
// --- ambient lightning ----------------------------------------------------------
const flashLight = new THREE.AmbientLight(0xdfe8ff, 0);
scene.add(flashLight);
let flash = 0;

function updateLightning() {
  if (flash > 0) flash -= 1;
  else if (Math.random() < 0.02) flash = 2;
  flashLight.intensity = flash > 0 ? 3.5 : 0;
}
//#endif

// --- loop -----------------------------------------------------------------------
let frame = 0;
let alive = true;

function setCamera(spec) {
  const s = { ...DEFAULT_CAM, ...(spec || {}) };
  camera.position.set(s.pos[0], s.pos[1], s.pos[2]);
  camTarget.set(s.target[0], s.target[1], s.target[2]);
  camera.fov = s.fov;
  camera.updateProjectionMatrix();
  camera.lookAt(camTarget);
}

function cameraSpec() {
  return {
    pos: camera.position.toArray(),
    target: camTarget.toArray(),
    fov: camera.fov,
    map: activeCamera !== camera,
  };
}

function renderFrame() {
  //#if jitter
  applyJitter(activeCamera, frame);
  //#endif
  renderer.clear();
  renderer.render(scene, activeCamera);
  //#if hud
  renderer.clearDepth();
  renderer.render(hudScene, hudCam);
  //#endif
}

function tick() {
  if (!alive) return; // an uncaught error stops the loop, like a dead requestAnimationFrame chain
  try {
    frame += 1;
    torch.intensity = 18 * (0.92 + 0.08 * Math.sin(frame * 0.37) * Math.sin(frame * 0.11 + 1.3));
    //#if player_gem
    updatePlayer();
    //#endif
    //#if lightning
    updateLightning();
    //#endif
    renderFrame();
  } catch (e) {
    alive = false;
    errors.push({ type: 'exception', message: String((e && e.message) || e), frame });
  }
}

function setKey(code, down) {
  if (down) {
    //#if mapview
    if (code === 'KeyM' && !keys.has('KeyM')) toggleMap();
    //#endif
    keys.add(code);
  } else {
    keys.delete(code);
  }
}

function pick(x, y) {
  const ndc = new THREE.Vector2((x / W) * 2 - 1, -(y / H) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, activeCamera);
  const hits = ray.intersectObjects(scene.children, true).filter((h) => h.object.visible);
  if (!hits.length) return { hit: false };
  const h = hits[0];
  // Surfaces at (almost) the same depth as the first hit: a strong hint for z-fighting.
  const coplanar = [...new Set(hits
    .filter((x) => Math.abs(x.distance - h.distance) < 0.005)
    .map((x) => x.object.name || 'unnamed'))];
  return { hit: true, point: h.point.toArray(), object: h.object.name || 'unnamed', distance: h.distance, coplanar };
}

setCamera(DEFAULT_CAM);

window.__gh = {
  ready: false,
  setCamera(spec) {
    setCamera(spec);
    return cameraSpec();
  },
  key: setKey,
  advance(n = 1) {
    for (let i = 0; i < n; i++) tick();
    return frame;
  },
  grab() {
    return canvas.toDataURL('image/png');
  },
  state() {
    return { frame, alive, errors: errors.slice(), camera: cameraSpec() };
  },
  pick,
};

Promise.all(pending).then(() => {
  renderFrame();
  window.__gh.ready = true;
  if (!TEST_MODE) {
    window.addEventListener('keydown', (e) => setKey(e.code, true));
    window.addEventListener('keyup', (e) => setKey(e.code, false));
    const loop = () => {
      tick();
      if (alive) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
});
