"""Hangar 9: a Godot 4 project, driven as a native build over the GlitchHunter bridge.

It shows the same agent working on a real engine: Godot's renderer (Forward+, ACES, glow,
directional shadows), its physics (rigid bodies with continuous collision detection), its
AnimationPlayer, visibility-range LOD, a spatial shader and Control-node anchors. The game is
built in code (no imported assets), so a commit runs straight from `git archive`.

Eleven bugs are planted after v1.0, the kind a Unity or Unreal team meets as often as a Godot
one: a normal map in the other engine's convention, a prop that stopped casting shadows, culling
bounds too small for the mesh, a broken animation binding, physics tunnelling. One arrives
through a merged branch, and one commit does not build (a GDScript parse error), which bisect
has to skip.
"""

from __future__ import annotations

from ..config import GAMES_DIR
from .base import BugSpec, CommitSpec, GameSpec

BRANCH = "feature/robot-rig"

HISTORY: tuple[CommitSpec, ...] = (
    CommitSpec("init: landing field, sky, sun and camera", scenarios=("overview",)),
    CommitSpec("add landing pad and supply drop", features=("pad",), scenarios=("supply_drop",)),
    CommitSpec("add maintenance robot with walk animation", features=("robot",), scenarios=("robot_walk",)),
    CommitSpec("add radio tower with visibility-range LOD", features=("tower",), scenarios=("flyby",)),
    CommitSpec("add jet engine with heat haze shader", features=("haze",)),
    CommitSpec("add HUD panels", features=("hud",)),
    CommitSpec("add color grading", features=("grading",)),
    CommitSpec("stream the ground texture", features=("streaming",)),
    CommitSpec("add fuel truck and generator", features=("props",)),
    CommitSpec("hangar wall cladding with normal maps", features=("panels",), scenarios=("hangar_wall",)),
    CommitSpec("jet exhaust sparks (CPU particles)", features=("exhaust",)),
    CommitSpec("polish lighting for v1.0", variants=("polish",), tag="v1.0"),
    # --- after the approved build ---------------------------------------------------------------
    CommitSpec("tweak fog density", variants=("fog_tweak",)),
    CommitSpec("pad floodlight: zero shadow bias for crisper contact shadows", variants=("acne",)),
    CommitSpec("robot: adopt the new rig naming", variants=("anim_path",), branch=BRANCH),
    CommitSpec("physics: disable CCD on supply pallets", variants=("tunneling",)),
    CommitSpec("art pack v2: cladding normal maps", variants=("normal_flip",)),
    CommitSpec("add credits header", features=("credits",), branch=BRANCH),
    CommitSpec("Merge branch 'feature/robot-rig'", merge=BRANCH),
    CommitSpec("grading: muted look for the dusk mission", variants=("color_shift",)),
    CommitSpec("WIP: inventory", features=("inventory",), variants=("wip_inventory",)),
    CommitSpec("fix build: close the inventory dictionary", drop_variants=("wip_inventory",)),
    CommitSpec("perf: props skip the shadow pass", variants=("no_shadow",)),
    CommitSpec("ground: stream the 32 px mip first", variants=("lowres",)),
    CommitSpec("LOD: cheaper far tower mesh", variants=("lod_pop",)),
    CommitSpec("truck: culling bounds from the chassis mesh", variants=("cull_bounds",)),
    CommitSpec("heat haze: n*sqrt(n) instead of pow", variants=("nan",)),
    CommitSpec("HUD: pin the objective panel to the mock-up layout", variants=("hud_anchor",)),
)

BUGS: dict[str, BugSpec] = {
    "acne": BugSpec("acne", "Shadow acne under the pad floodlight (shadow bias set to zero)", "shadow_acne",
                    ("overview", "supply_drop")),
    "anim_path": BugSpec("anim_path", "Robot slides without walking: animation tracks point at renamed nodes",
                         "frozen_animation", ("robot_walk",)),
    "tunneling": BugSpec("tunneling", "Supply pallet falls through the landing pad (CCD disabled)",
                         "object_missing", ("supply_drop",)),
    "color_shift": BugSpec("color_shift", "Whole picture desaturated by the color grade", "color_shift",
                           ("overview", "robot_walk", "supply_drop")),
    "lowres": BugSpec("lowres", "Ground texture stuck at a 32 px mip (blurry)", "detail_loss", ("overview",)),
    "lod_pop": BugSpec("lod_pop", "Radio tower pops when it switches level of detail", "pop_in", ("flyby",)),
    "nan": BugSpec("nan", "Heat haze shader writes NaN: black blocks over the jet engine", "nan_pixels",
                   ("overview",)),
    "hud_anchor": BugSpec("hud_anchor", "Objective panel cut off on 4:3 screens (pinned to 640 px)",
                          "ui_offscreen", ("overview",), viewport=(480, 360)),
    "normal_flip": BugSpec("normal_flip", "Wall cladding lit upside down: normal maps in the DirectX convention",
                           "shading_inverted", ("hangar_wall",)),
    "no_shadow": BugSpec("no_shadow", "Generator and fuel truck cast no shadow (props skip the shadow pass)",
                         "shadow_missing", ("overview",)),
    "cull_bounds": BugSpec("cull_bounds", "Fuel truck vanishes while still on screen (culling bounds from the "
                           "chassis only)", "pop_in", ("flyby",)),
}

SCENARIOS: dict[str, dict] = {
    s["name"]: s
    for s in (
        {"name": "overview", "frames": 24,
         "description": "Static wide camera over the landing field for 24 frames."},
        {"name": "robot_walk", "frames": 40, "motion_expected": True,
         "description": "Static close camera on the maintenance robot's patrol; 40 frames.",
         "camera": {"pos": [-1.0, 2.4, 8.0], "target": [-2.6, 1.1, 1.5], "fov": 50}},
        {"name": "supply_drop", "frames": 40, "events": [[1, "down", "KeyE"], [3, "up", "KeyE"]],
         "description": "Press E: a supply pallet is fired onto the landing pad; static camera, 40 frames.",
         "camera": {"pos": [2.5, 6.0, 4.5], "target": [7.5, 3.0, -5.0], "fov": 50}},
        {"name": "hangar_wall", "frames": 16,
         "description": "Static camera on the ribbed hangar wall cladding (normal-mapped); 16 frames.",
         "camera": {"pos": [3.0, 3.4, -8.0], "target": [-0.5, 3.2, -15.7], "fov": 50}},
        {"name": "flyby", "frames": 60, "motion_expected": True,
         "description": "Camera flies across the field toward the radio tower; 60 frames.",
         "camera_path": [{"frame": 0, "pos": [0, 12, 52], "target": [4, 9, -34], "fov": 55},
                         {"frame": 59, "pos": [0, 9, 2], "target": [4, 9, -34], "fov": 55}]},
    )
}

SPEC = GameSpec(
    name="hangar",
    title="Hangar 9",
    template_dir=GAMES_DIR / "hangar",
    history=HISTORY,
    bugs=BUGS,
    textures={},
    assets_by_feature={},
    manifest={
        "name": "hangar", "title": "Hangar 9 (Godot 4)", "baseline": "v1.0", "branch": "main",
        "entry": "project.godot", "viewport": [640, 360], "viewports": [[480, 360]], "sweep_scenarios": ["overview"],
        "patch_paths": ["main.gd", "main.tscn", "project.godot"], "fps": 30,
        "driver": {
            "type": "bridge",
            "launch": ["{godot}", "--path", "{dir}", "--fixed-fps", "30", "--resolution", "{width}x{height}",
                       "--position", "40,40", "--", "--gh-port", "{port}"],
            "ready_timeout": 90,
            "error_patterns": ["Parse Error", "Failed to load script", "SCRIPT ERROR"],
        },
    },
    scenarios=SCENARIOS,
    vendor=(("bridges/godot/glitchhunter_bridge.gd", "glitchhunter_bridge.gd"),),
    author=("Hangar 9 Dev", "dev@hangar9.invalid"),
    clean_subjects=("tweak fog density",),
    notes={"untestable_subjects": ("WIP: inventory",), "branch": BRANCH},
    template_file="main.tpl.gd",
    output_file="main.gd",
    static_files=("project.godot", "main.tscn"),
    gitignore=".godot/\n*.import\n",
    engine="Godot 4 (native, bridge)",
)
