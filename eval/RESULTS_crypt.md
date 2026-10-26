# GlitchHunter evaluation: Crypt Runner

Run 2026-10-26 02:01 on AMD64 with OpenCV 5.0.0 (2 repeat(s) per mode). Ground truth: 7 bugs planted in a 17-commit history (6 deterministic, 1 intermittent).

| mode | recall | precision | culprit found | fix verified | false alarms / run | captures | time (s) |
|---|---|---|---|---|---|---|---|
| full | 100% | 100% | 100% | 100% | 0.0 | 130 | 350 |
| no_zoom | 86% | 100% | 86% | 64% | 0.0 | 133 | 387 |
| fixed_pipeline | 79% | 84% | 0% | 0% | 1.0 | 3 | 8 |

## Per bug (fraction of runs)

| bug | detector | full detected | no_zoom detected | fixed_pipeline detected | full: culprit | full: fix verified |
|---|---|---|---|---|---|---|
| Crate renders with the magenta missing-texture placeholder (deterministic) | `missing_texture` | 100% | 100% | 100% | 100% | 100% |
| Rune decal z-fights with the floor and flickers (deterministic) | `zfight_flicker` | 100% | 0% | 0% | 100% | 100% |
| Random full-screen white flashes (photosensitivity risk) (intermittent) | `flash` | 100% | 100% | 50% | 100% | 100% |
| Rock mesh explodes into spikes (deterministic) | `geometry_explosion` | 100% | 100% | 100% | 100% | 100% |
| Game freezes when the player picks up the gem (deterministic) | `frozen_frame` | 100% | 100% | 100% | 100% | 100% |
| HUD text overflows its panel (deterministic) | `hud_text_overflow` | 100% | 100% | 100% | 100% | 100% |
| Map view renders a black screen (deterministic) | `black_frame` | 100% | 100% | 100% | 100% | 100% |

## False alarms on clean builds

0 of 12 captures of clean builds (04febd6e13, 46bab01cf4; every scenario and screen size) produced a detection (0.0%).

