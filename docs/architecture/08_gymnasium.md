# 08 — Gymnasium

```text
gymnasium/
├── observation_space/
│   ├── perceptions/
│   │   ├── composite/
│   │   │   ├── composite_fully_pomdp.py
│   │   │   ├── composite_grid_markovian.py
│   │   │   └── composite_markovian.py
│   │   ├── spatial/
│   │   │   └── grid_pixel.py
│   │   ├── vector/
│   │   │   ├── vector_fog_of_war.py
│   │   │   ├── vector_markovian.py
│   │   │   └── vector_markovian_easy.py
│   │   └── base_perception.py
│   └── observation_handler.py
├── utils/
│   ├── episode_logging/
│   │   ├── csv_episode_logger.py
│   │   └── keys.py
│   ├── env_factory.py
│   └── episode_termination.py
├── action_space.py
└── environment.py
```

This package is the adapter between the simulation and Gymnasium.
`SYNGridEnv` in `environment.py` implements `reset`, `step` and `render`. It
owns the action space (the four `DroidAction` moves), an `ObservationHandler`,
and a `PygameRenderer` when a render mode is set. It is still the environment
from before the rebuild: it takes `world_conf` and `obs_conf`, builds its
`GridWorld` from five config blocks, and does not import, because those config
classes are gone.

A step is three calls in order: the world performs the action and returns a
reward, `check_episode_end` in `utils/episode_termination.py` decides whether
the episode ends and what the final reward is, and the observation handler
encodes the result. On a terminal step the environment also puts the engine's
chain counts into `info`.

`utils/env_factory.py` registers the environment id and creates instances
through `gym.make`. It already passes a `Scenario`, which `SYNGridEnv` does not
accept yet. `utils/episode_logging/` holds the wrapper that writes one CSV row
per finished episode. `observation_space/` is covered in `02_observation.md`.
