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

The handler looks the perception up by name in its `PERCEPTIONS` table, which
lists seven. It is still built from the old `ObsConfig`, and each perception
from a `PerceptionConf` that supplies the grid size, the step budget and the
score and tier bounds. Neither class is in `config/models/` any more, so the
package does not import. Their replacement is `ObservationRules` in
`scenario/blocks/observation.py`, which the tier-chain builder fills in and
nothing reads yet.

The `Perception` enum in the config models names two of the seven,
`vector_fog_of_war` and `vector_markovian`. Both produce the same layout: droid
position, then one `[active, y, x, identity]` slot per orb, nearest first. Fog
of war additionally hides every orb outside the 3×3 window around the droid.
