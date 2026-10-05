# 09 — Support packages

```text
assets/
├── fonts/
│   └── Minecraft.ttf
├── sounds/
│   └── training_complete.wav
├── sprites/
│   ├── droid.png
│   ├── hud.png
│   ├── negative_orb.png
│   └── positive_orb.png
└── tiles/
    └── floor.png

plot/
├── plot_eval.py
├── plot_training.py
└── plot_utils.py

rendering/
└── pygame_renderer.py

utils/
├── date_utils.py
└── paths_util.py
```

`rendering/` contains `PygameRenderer`, which draws the grid, orbs, droid and
HUD from the data `SYNGridEnv.render` passes it, either to a window (`human`)
or to a pixel array (`rgb_array`). It also reads the arrow keys for
`HumanRunner`. Its images, font and the training-complete sound come from
`assets/`.

`utils/` resolves paths — package-relative for config and assets, and relative
to the current working directory for `output/` — and produces the timestamp
used in run ids.

`plot/` is a standalone set of scripts that read the CSV and TensorBoard output
of finished runs and draw figures. Nothing in the run path imports it.
