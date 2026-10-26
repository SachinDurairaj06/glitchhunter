# Using GlitchHunter with any engine

GlitchHunter needs three things from a game: a way to build any commit, a way to play a scripted
scenario one fixed time step at a time, and the rendered frames. It gets them in one of three ways.

| Engine | How GlitchHunter drives it | What works | Status |
|---|---|---|---|
| Web (three.js, Babylon.js, PlayCanvas, any WebGL/WebGPU canvas) | Headless Chromium + the `window.__gh` test API (drop-in module: `bridges/web/glitchhunter-test-api.js`) | Everything: survey, zoom, sweep, replay, visual bisect, fix + verify | Tested (Crypt Runner, Ironworks, and a minimal game built only on the drop-in module) |
| Godot 4 (desktop) | Native build + the bridge autoload `bridges/godot/glitchhunter_bridge.gd` | Everything | Tested (Hangar 9, Godot 4.7.2, Vulkan Forward+) |
| Unity (standalone player) | Native build + `bridges/unity/GlitchHunterBridge.cs`, built per commit by `bridges/unity/Editor/GlitchHunterBuild.cs` | Everything | Reference implementation, not yet run in a Unity project |
| Unreal, custom engines | Native build + a bridge you write against the protocol below (about 200 lines) | Everything | Protocol documented; Python reference engine in `bridges/python/fake_engine.py`, graded by `bridge-check` |
| Any engine, no code changes | Video triage of a recording (`python -m glitchhunter scan-video`) | Detection, zoom (native resolution), timestamps, report; no bisect or fix | Works on any MP4/AVI |

The agent, the detectors, bisect and the dashboard are the same in every case. Only the
`driver` section of the game's `glitchhunter.json` changes.

## Onboarding a project in three commands

```bash
python -m glitchhunter init --engine godot --dir path/to/game
```

`init` (engines: `web`, `godot`, `unity`, `unreal`, `custom`) writes a starter `glitchhunter.json`
with the right driver and two scenarios, copies the engine's bridge into the project and prints
what is left to do by hand (register the Godot autoload, add the Unity component). It never
overwrites a file unless given `--force`.

```bash
python -m glitchhunter bridge-check --repo path/to/game
```

`bridge-check` launches the build with the project's launch command (or any command after `--`,
with `{port}`, `{width}` and `{height}` placeholders) and grades the bridge: hello, reset, grab
size, frame stepping, set_camera, key input, pick, state, quit, and above all determinism: the
same seed played twice must give identical frames, because replay, golden comparison and visual
bisect rest on it. When two runs differ it says from which frame and by how much, which points
at the usual causes (an unseeded random generator, wall-clock time in a shader, temporal
antialiasing noise). Output for Hangar 9:

```
  PASS  connect      connected back on the port given as {port}
  PASS  hello        engine godot 4.7.2-stable (official), api 1, viewport [640, 360]
  PASS  reset        fresh scene in 0.19s
  PASS  grab         PNG 640x360
  PASS  advance      advance 3 moved the frame counter 0 -> 3
  PASS  determinism  12 frames identical across two runs with seed 1 (51 ms per frame advanced and grabbed)
  INFO  seed         seed 2 gives the same frames as seed 1 (fine if the game has no randomness)
  PASS  state        frame, alive and errors reported
  PASS  set_camera   moving the camera changes the picture
  PASS  key          Space down/up accepted; no visible reaction in 6 frames (fine if the key is unbound)
  PASS  pick         centre of the screen: Ground at 20.0 m
  PASS  quit         process exited

12 checks, 0 failed, 0 warnings: ready for GlitchHunter
```

```bash
python -m glitchhunter hunt --repo path/to/game
```

## The same glitch in every engine

The detectors look at pixels, so they do not care which engine drew them. The bugs planted in the
three test games are the ones Unity, Unreal and Godot teams file every week:

| Glitch (detector) | Unity | Unreal | Godot (planted in Hangar 9) | three.js (planted in Ironworks / Crypt Runner) |
|---|---|---|---|---|
| Missing texture or shader (`missing_texture`) | magenta error shader | default checker / WorldGridMaterial | missing resource | magenta placeholder texture |
| Inverted normal map (`shading_inverted`) | texture not imported as Normal Map, or DirectX-convention art | green channel flip setting | DirectX-convention normal map (`normal_flip`) | |
| Missing shadow (`shadow_missing`) | Mesh Renderer Cast Shadows off | Cast Shadow unchecked | `cast_shadow = OFF` on props (`no_shadow`) | |
| Culled while on screen (`pop_in`) | SkinnedMeshRenderer bounds too small | wrong bounds scale | `custom_aabb` from the chassis only (`cull_bounds`) | |
| LOD pop (`pop_in`) | LOD Group transition | HLOD / LOD screen size | visibility range with a wrong far mesh (`lod_pop`) | `THREE.LOD` distances (`lod_pop`) |
| Shadow acne (`shadow_acne`) | light bias 0 | shadow bias | spot light bias 0 (`acne`) | `shadow.bias = 0` (`acne`) |
| Physics tunnelling (`object_missing`) | Collision Detection = Discrete | CCD off | `continuous_cd = false` (`tunneling`) | too few substeps (`tunneling`) |
| T-pose / sliding character (`frozen_animation`) | Animator avatar mismatch, renamed bones | retarget failure | animation tracks point at renamed nodes (`anim_path`) | clip tracks bound to renamed bones (`anim_retarget`) |
| Wrong color space (`color_shift`) | Gamma vs Linear project setting | sRGB flag on a texture | grading saturation (`color_shift`) | sRGB encoded twice (`color_shift`) |
| Low-res texture (`detail_loss`) | streaming mip budget | texture streaming pool | 32 px mip resident (`lowres`) | missing `map` guard (`lowres`) |
| NaN pixels (`nan_pixels`) | `pow` of a negative number in a shader | same | `n * sqrt(n)` in a spatial shader (`nan`) | same, in a post pass (`nan`) |
| UI off screen (`ui_offscreen`) | Canvas anchors, Canvas Scaler | UMG anchors | panel pinned to 640 px (`hud_anchor`) | `OBJECTIVE_X` pinned to 640 (`ui_anchor`) |

## glitchhunter.json

Every game repository has one file at its root. It is read at the branch head, so the test plan
is the same for every commit the agent tests.

```json
{
  "name": "hangar",
  "title": "Hangar 9 (Godot 4)",
  "baseline": "v1.0",
  "viewport": [640, 360],
  "viewports": [[480, 360]],
  "patch_paths": ["main.gd", "main.tscn", "project.godot"],
  "driver": {
    "type": "bridge",
    "launch": ["{godot}", "--path", "{dir}", "--fixed-fps", "30", "--resolution", "{width}x{height}",
               "--", "--gh-port", "{port}"],
    "error_patterns": ["Parse Error", "Failed to load script"]
  },
  "scenarios": [
    {"name": "overview", "description": "Static wide camera", "frames": 24},
    {"name": "supply_drop", "description": "Press E: a pallet drops", "frames": 40,
     "events": [[1, "down", "KeyE"], [3, "up", "KeyE"]],
     "camera": {"pos": [2.5, 6, 4.5], "target": [7.5, 3, -5], "fov": 50}},
    {"name": "flyby", "description": "Camera flies toward the tower", "frames": 60, "motion_expected": true,
     "camera_path": [{"frame": 0, "pos": [0, 12, 52], "target": [4, 9, -34]},
                     {"frame": 59, "pos": [0, 9, 2], "target": [4, 9, -34]}]}
  ]
}
```

| Key | Meaning |
|---|---|
| `baseline` | Tag or commit of the approved build: golden frames come from it, bisect starts from it |
| `viewport`, `viewports` | Default screen size, and extra sizes for the UI sweep (4:3, 21:9, ...) |
| `patch_paths` | Files a proposed fix may touch (the patch guard refuses anything else) |
| `driver.type` | `web` (default) or `bridge` |
| `driver.build` | Optional command run once per commit before launching (e.g. a Unity or Godot export) |
| `driver.launch` | Command that starts the build. Placeholders: `{dir}` exported commit, `{port}`, `{width}`, `{height}`, `{seed}`, and tool names such as `{godot}` or `{unity}` (found via `GH_TOOL_<NAME>`, a local tools folder, or PATH) |
| `driver.error_patterns` | Log lines that mean the game's scripts failed to load (the commit is then skipped as untestable) |
| `scenarios[].events` | `[frame, "down"|"up", KeyboardEvent.code]` key presses |
| `scenarios[].camera` / `camera_path` | Fixed camera, or keyframes interpolated every frame (scenarios with a path cannot be zoomed) |
| `scenarios[].camera_moves` | The game moves the camera itself (a follow rig): no zoom, motion detectors adapt |

## The bridge protocol (native builds)

GlitchHunter listens on a local port and starts the game with `--gh-port <port>`; the game
connects back. Then it is request/reply, one JSON object per line. Key codes use the browser's
`KeyboardEvent.code` names (`KeyW`, `Digit1`, `Space`, `ArrowUp`), so one test plan serves web
and native builds. Screen coordinates have their origin at the top left.

| Request | Reply | The game must |
|---|---|---|
| `{"cmd": "hello"}` | `{"ok": true, "api": 1, "engine": "godot 4.7.2", "viewport": [w, h]}` | Answer right after connecting |
| `{"cmd": "reset", "seed": 1}` | `{"ok": true}` once the scene is fresh | Reload the level, seed its random generators, stay paused |
| `{"cmd": "set_camera", "pos": [x,y,z], "target": [x,y,z], "fov": 55}` | `{"ok": true, "camera": {...}}` | Move the active camera |
| `{"cmd": "key", "code": "KeyE", "down": true}` | `{"ok": true}` | Feed the key to gameplay input (works while paused) |
| `{"cmd": "advance", "frames": 1}` | `{"ok": true, "frame": n}` after the frames ran | Run exactly that many frames with a fixed delta, then pause |
| `{"cmd": "grab"}` | `{"ok": true, "png": "<base64>", "size": [w, h]}` | Return the last rendered frame |
| `{"cmd": "state"}` | `{"ok": true, "frame": n, "alive": true, "errors": [...], "camera": {...}}` | Report script errors it saw |
| `{"cmd": "pick", "x": 320, "y": 180}` | `{"ok": true, "hit": true, "object": "LandingPad", "point": [x,y,z], "distance": d, "coplanar": [...]}` | Ray-cast from the camera through the pixel |
| `{"cmd": "quit"}` | `{"ok": true}` | Exit |

Determinism is the part that matters: a fixed frame time (Godot `--fixed-fps 30`, Unity
`Time.captureDeltaTime`), a paused simulation between commands, seeded randomness (particles
included: Godot's `use_fixed_seed`), and shaders that read time from a uniform the game sets
rather than wall-clock time. `bridge-check` measures it.

## Web games

Import the drop-in module and hand it your canvas, camera, scene and a `tick()` that advances one
fixed step and draws:

```js
import * as THREE from 'three';
import { installTestApi, testParams } from './glitchhunter-test-api.js';

const params = testParams();            // ?test=1&seed=<n>&w=<width>&h=<height>
const api = installTestApi({
  THREE, canvas: renderer.domElement, camera, scene,
  tick: () => { update(1 / 30); renderer.render(scene, camera); },
  setKey: (code, down) => input.set(code, down),   // KeyboardEvent.code names
});
api.markReady();                        // once assets are loaded and the first frame is drawn
if (!params.test) startYourOwnLoop();   // in test mode GlitchHunter advances the game
```

Camera control and ray-cast picking default to three.js; for Babylon.js, PlayCanvas or a 2D
canvas pass `cameraSpec`, `setCamera` and `pick` hooks. `tests/test_web_template.py` builds a
minimal game on this module alone and runs capture, picking and a visual bisect on it.

## Godot 4

1. Copy `bridges/godot/glitchhunter_bridge.gd` into the project and register it as an autoload
   named `GlitchHunter` (or run `init --engine godot`).
2. Poll input in `_process` (`Input.is_key_pressed`, actions) rather than relying only on input
   callbacks: the driver presses keys while the tree is paused.
3. Give objects you care about colliders (pick uses physics ray casts) and optionally
   `set_meta("gh_name", "LandingPad")` for readable names.
4. Use the `driver` section above. Point `GH_TOOL_GODOT` at the Godot binary if it is not on PATH.

Godot needs a GPU or a software Vulkan/OpenGL driver to render; `--headless` has no renderer.

## Unity

1. `init --engine unity` copies `GlitchHunterBridge.cs` and the editor build script
   `Editor/GlitchHunterBuild.cs` into `Assets/GlitchHunter/`. Add the bridge component to a
   GameObject in the first scene.
2. Read keys through `GlitchHunterBridge.GetKey` / `GetKeyDown` (or add an Input System device
   writer), and keep randomness on `UnityEngine.Random` (the bridge seeds it on reset).
3. The driver's `build` step runs `Unity -batchmode -quit -projectPath {dir} -executeMethod
   GlitchHunterBuild.Build` for each commit it tests (set `GH_TOOL_UNITY`; it uses your Unity
   license). A build that fails exits with code 1, and bisect skips that commit. The player starts
   with the screen size and port:

   ```json
   "launch": ["{dir}/Build/Game.exe", "-screen-width", "{width}", "-screen-height", "{height}",
              "-screen-fullscreen", "0", "--gh-port", "{port}"]
   ```

4. `bridge-check --repo .` grades the integration before the first hunt.

The bridge has not been run inside Unity here (no license on the development machine), but it is
type-checked against stand-ins carrying Unity 2021.3's API signatures, with the C# version Unity
compiles with: `dotnet build bridges/unity/check` (needs a .NET SDK; clone to a short path on
Windows). Reviewing it that way found two bugs that only show at run time in a real project: a
scene reload started a second bridge that connected again, and a scripted key press could be
consumed by a paused frame (so a jump would happen with a zero time step on some runs and not
others).

## Unreal Engine

Implement the nine commands in a game instance subsystem: an `FSocket` client in `Initialize`,
`UGameplayStatics::SetGamePaused` for stepping with a fixed `FApp::SetFixedDeltaTime(1/30.f)` and
`FApp::SetUseFixedTimeStep(true)`, `FScreenshotRequest` or a scene capture for `grab`,
`APlayerController::DeprojectScreenPositionToWorld` plus a line trace for `pick`.
`init --engine unreal` writes the driver (BuildCookRun per commit) and copies
`bridges/python/fake_engine.py`, a complete, runnable implementation of the protocol to port;
`bridge-check` tells you when yours behaves the same. Until then, record playtests and use video
triage.

## Video triage (no engine integration)

```bash
python -m glitchhunter scan-video playtest.mp4 --golden approved-build.mp4
```

Shots are found from histogram and pixel cuts; every detector that does not need the game runs
over sliding windows with thresholds for lossy video; small or uncertain findings are re-read at
the recording's native resolution and every frame; findings seen in several windows gain
confidence; a person approves before the report is written. With `--golden` (the same scripted
run recorded on an approved build, frame-aligned) the golden comparisons run too. There is no
bisect: a recording has no history.
