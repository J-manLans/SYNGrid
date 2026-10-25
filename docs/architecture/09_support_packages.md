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
or to a pixel array (`rgb_array`). It also draws an episode summary and a pause
screen over the last frame, sets the window's title, and closes the window when
the env is closed. It reads no input. Its images and font come from `assets/`,
and the training-complete sound there is played by the agent runners. It takes
the grid size from the world and its image paths from its own table.

`utils/` resolves paths — package-relative for config and assets, and relative
to the current working directory for `output/` — and produces the timestamp
used in run ids.

`plot/` is a standalone set of scripts that read the CSV and TensorBoard output
of finished runs and draw figures. Nothing in the run path imports it.
