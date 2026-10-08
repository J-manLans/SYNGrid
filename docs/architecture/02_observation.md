# 02 — Observation

```text
gymnasium/observation_space/
├── perceptions/
│   ├── composite/
│   │   ├── composite_fully_pomdp.py
│   │   ├── composite_grid_markovian.py
│   │   └── composite_markovian.py
│   ├── spatial/
│   │   └── grid_pixel.py
│   ├── vector/
│   │   ├── vector_fog_of_war.py
│   │   ├── vector_markovian.py
│   │   └── vector_markovian_easy.py
│   └── base_perception.py
└── observation_handler.py
```

This package turns a `GridWorld` into the array the agent sees. Each
`SYNGridEnv` owns one `ObservationHandler`, which owns one perception and the
episode's `steps_left` counter. A perception declares the Gymnasium observation
space once, then on every reset and step reads the droid position and the orb
list from the world and writes them into its buffer.

Which perception is used comes from the scenario YAML's `perception` key. It
travels as `ObservationRules.perception` on the `Scenario` and is looked up in
the `PERCEPTIONS` table in `observation_handler.py`. The scenario also supplies
the numbers that size the observation (slot count, sort limit, tier bound), so
perceptions never read config themselves.

Only `vector_fog_of_war` and `vector_markovian` are selectable; the other five
are in the table but not in the `Perception` enum. Both selectable ones produce
the same layout — droid position, then one `[active, y, x, identity]` slot per
orb, nearest first. Fog of war additionally hides every orb outside the 3×3
window around the droid.
