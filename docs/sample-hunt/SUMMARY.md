# GlitchHunter hunt `h1023-030846-08f5`

- Planner: rule
- Head: `c7a252ba80`  baseline: `ad32e651f1`
- Steps: 46  captures: 140  time: 298s
- Ended because: planner finished: reported ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']; waiting for a human on []

| id | glitch | severity | status | culprit | fix | report |
|---|---|---|---|---|---|---|
| F6 | Game freezes during 'walk_to_gem' | critical | reported | `ae5601f66e` use per-gem reward values on pickup | `glitchhunter/h1023-030846-08f5-f6-frozen_frame` | [F6.md](reports/F6.md) |
| F7 | Black screen in 'map_view' | critical | rejected | `b6d6bd6739` extend map view draw distance |  |  |
| F4 | Exploded / stretched geometry on rock | major | reported | `1f366b94a3` tune rock noise amplitude | `glitchhunter/h1023-030846-08f5-f4-geometry_explosion` | [F4.md](reports/F4.md) |
| F1 | Missing texture placeholder on crate | major | reported | `c08ccda574` reorganize texture paths | `glitchhunter/h1023-030846-08f5-f1-missing_texture` | [F1.md](reports/F1.md) |
| F3 | Full-screen flashes (photosensitivity risk) | major | reported | `c64d8467d7` add ambient lightning flashes | `glitchhunter/h1023-030846-08f5-f3-flash` | [F3.md](reports/F3.md) |
| F5 | HUD text overflows its panel | minor | reported | `b1d02d0348` localize HUD labels | `glitchhunter/h1023-030846-08f5-f5-hud_text_overflow` | [F5.md](reports/F5.md) |
| F2 | Z-fighting flicker on floor+rune_decal | minor | reported | `ba98e5d336` snap rune decal flush to the floor | `glitchhunter/h1023-030846-08f5-f2-zfight_flicker` | [F2.md](reports/F2.md) |

Full decision trace: [trace.jsonl](trace.jsonl)
