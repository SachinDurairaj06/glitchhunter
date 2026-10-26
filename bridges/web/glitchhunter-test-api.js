// GlitchHunter test API for web games: three.js out of the box, any other engine (Babylon.js,
// PlayCanvas, a 2D canvas) through hooks.
//
//   import * as THREE from 'three';
//   import { installTestApi, testParams } from './glitchhunter-test-api.js';
//
//   const params = testParams();            // ?test=1&seed=<n>&w=<width>&h=<height>
//   const api = installTestApi({
//     THREE, canvas: renderer.domElement, camera, scene,
//     tick: () => { update(1 / 30); renderer.render(scene, camera); },   // one fixed step, then draw
//     setKey: (code, down) => input.set(code, down),                     // KeyboardEvent.code names
//   });
//   api.markReady();                        // once assets are loaded and the first frame is drawn
//   if (!params.test) startYourOwnLoop();   // in test mode GlitchHunter advances the game itself
//
// Hooks you may add: cameraSpec() and setCamera(spec) for other engines' cameras, pick(x, y)
// (return {hit, object, point, distance, coplanar}), state() for extra fields. Seed your random
// generators from params.seed and read time from the frame counter, never from the clock: replay
// and visual bisect need the same frames on every run.

export function testParams() {
  const q = new URLSearchParams(window.location.search);
  return {
    test: q.get('test') === '1',
    seed: Number(q.get('seed') || 1),
    width: Number(q.get('w') || 0) || null,
    height: Number(q.get('h') || 0) || null,
  };
}

function threeCameraSpec(camera, target) {
  return {
    pos: camera.position.toArray(),
    target: target ? target.toArray() : null,
    fov: camera.fov,
  };
}

export function installTestApi(hooks) {
  const { canvas, tick } = hooks;
  let frame = 0;
  const errors = [];
  window.addEventListener('error', (e) => errors.push({ message: String(e.message), frame }));
  window.addEventListener('unhandledrejection', (e) => errors.push({ message: String(e.reason), frame }));
  const target = hooks.THREE ? new hooks.THREE.Vector3() : null;

  const cameraSpec = hooks.cameraSpec || (() => threeCameraSpec(hooks.camera, target));
  const setCamera = hooks.setCamera || ((spec) => {
    const cam = hooks.camera;
    if (spec.pos) cam.position.fromArray(spec.pos);
    if (spec.target) {
      target.fromArray(spec.target);
      cam.lookAt(target);
    }
    if (spec.fov) {
      cam.fov = spec.fov;
      cam.updateProjectionMatrix();
    }
  });
  const pick = hooks.pick || ((x, y) => {
    const { THREE, camera, scene } = hooks;
    const ndc = new THREE.Vector2((x / canvas.width) * 2 - 1, -(y / canvas.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects(scene.children, true).filter((h) => h.object.visible && h.object.isMesh);
    if (!hits.length) return { hit: false };
    const first = hits[0];
    const coplanar = [...new Set(hits.filter((h) => Math.abs(h.distance - first.distance) < 0.005)
      .map((h) => h.object.name || 'unnamed'))];
    return { hit: true, object: first.object.name || 'unnamed', point: first.point.toArray(),
      distance: first.distance, coplanar };
  });

  const api = {
    ready: false,
    setCamera(spec) {
      setCamera(spec);
      return cameraSpec();
    },
    key(code, down) {
      hooks.setKey(code, down);
    },
    advance(n = 1) {
      for (let i = 0; i < n; i++) {
        tick();
        frame += 1;
      }
      return frame;
    },
    grab() {
      return canvas.toDataURL('image/png');
    },
    state() {
      return { frame, alive: true, errors: errors.slice(), camera: cameraSpec(),
        viewport: [canvas.width, canvas.height], ...(hooks.state ? hooks.state() : {}) };
    },
    pick,
  };
  window.__gh = api;
  return {
    markReady() {
      api.ready = true;
    },
    frame: () => frame,
  };
}
