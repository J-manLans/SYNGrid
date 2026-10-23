# 00 — File structure

## Repository

```text
SYNGrid/
├── docs/
├── output/
├── reproduction_package/
├── scripts/
├── src/syn_grid/
├── tests/
├── AGENTS.md
├── pyproject.toml
├── pytest.ini
└── README.md
```

## `src/syn_grid/`

```text
syn_grid/
├── assets/
│   ├── fonts/
│   ├── sounds/
│   ├── sprites/
│   └── tiles/
│
├── config/
│   ├── models/
│   │   ├── scenario/
│   │   │   ├── common_models.py
│   │   │   └── orb_models.py
│   │   ├── global_models.py
│   │   └── runner_models.py
│   ├── yaml/
│   │   ├── global_config.yaml
│   │   ├── goal_tier_chain_spatial.yaml
│   │   └── runner_config.yaml
│   └── config_manager.py
│
├── core/
│   ├── droid/
│   │   ├── digestion/
│   │   │   ├── digesters/
│   │   │   │   ├── tier_chain/
│   │   │   │   │   ├── max_tier_digester.py
│   │   │   │   │   ├── step_wise_digester.py
│   │   │   │   │   ├── threshold_digester.py
│   │   │   │   │   └── tier_digester.py
│   │   │   │   └── neg_digester.py
│   │   │   ├── digester.py
│   │   │   └── engine.py
│   │   ├── droid_action.py
│   │   └── synergy_droid.py
│   ├── orbs/
│   │   ├── direct/
│   │   │   └── negative_orb.py
│   │   ├── synergy/
│   │   │   ├── chain/
│   │   │   │   └── tier_orb.py
│   │   │   └── effect/
│   │   ├── base_orb.py
│   │   ├── orb_factory.py
│   │   └── orb_meta.py
│   ├── utils/
│   │   └── timer.py
│   └── grid_world.py
│
├── gymnasium/
│   ├── observation_space/
│   │   ├── perceptions/
│   │   │   ├── composite/
│   │   │   │   ├── composite_fully_pomdp.py
│   │   │   │   ├── composite_grid_markovian.py
│   │   │   │   └── composite_markovian.py
│   │   │   ├── spatial/
│   │   │   │   └── grid_pixel.py
│   │   │   ├── vector/
│   │   │   │   ├── vector_fog_of_war.py
│   │   │   │   ├── vector_markovian.py
│   │   │   │   └── vector_markovian_easy.py
│   │   │   └── base_perception.py
│   │   └── observation_handler.py
│   ├── utils/
│   │   ├── episode_logging/
│   │   │   ├── csv_episode_logger.py
│   │   │   └── keys.py
│   │   ├── env_factory.py
│   │   └── episode_termination.py
│   └── environment.py
│
├── legacy/
│
├── plot/
│   ├── plot_eval.py
│   ├── plot_training.py
│   └── plot_utils.py
│
├── rendering/
│   └── pygame_renderer.py
│
├── runners/
│   ├── agent_runners/
│   │   ├── sb3/
│   │   │   ├── artifact_manager.py
│   │   │   ├── base_sb3_runner.py
│   │   │   ├── execution_strategy.py
│   │   │   ├── frame_stack_ppo.py
│   │   │   ├── lstm_ppo.py
│   │   │   ├── policy_resolver.py
│   │   │   └── stateless_ppo.py
│   │   ├── utils/
│   │   │   └── extractors.py
│   │   ├── base_agent_runner.py
│   │   ├── runner_bundle.py
│   │   └── runner_registry.py
│   └── human_runner/
│       └── human_runner.py
│
├── scenario/
│   ├── blocks/
│   │   ├── populations/
│   │   │   ├── neg_population.py
│   │   │   ├── population.py
│   │   │   └── tier_chain_population.py
│   │   ├── recipes/
│   │   │   ├── tier_digester_recipe.py
│   │   │   └── world_recipe.py
│   │   ├── hud.py
│   │   ├── metrics.py
│   │   ├── observation.py
│   │   ├── orb_bundle.py
│   │   ├── readers.py
│   │   └── termination.py
│   ├── families/
│   │   ├── continuous/
│   │   └── goal/
│   │       ├── tier_chain/
│   │       │   ├── builders.py
│   │       │   ├── config.py
│   │       │   └── termination.py
│   │       └── config.py
│   ├── registry.py
│   └── scenario.py
│
├── utils/
│   ├── date_utils.py
│   └── paths_util.py
│
├── __main__.py
├── app.py
└── check_env.py
```

`app.py` is the entry point: it loads the three YAML files through `config/`,
asks `scenario/` to build one `Scenario`, hands it to a runner from `runners/`,
and the runner creates the Gymnasium environments in `gymnasium/`. `config/`,
`scenario/` and `runners/` are on that structure. `core/`, `gymnasium/` and
`rendering/` are still the code from before the rebuild and name config classes
that no longer exist, so the app does not start.

`scenario/` is the only package that turns config into simulation objects. It
also holds the config models that are specific to a scenario type or family, in
a `config.py` beside what they configure; `config/models/` keeps only what
every scenario shares.
`legacy/` is a frozen copy of the stable code, run instead of the working app
when `USE_LEGACY` in `__main__.py` is set.
`core/` is the simulation itself, `gymnasium/` wraps one world as an
environment and encodes observations, `runners/` trains, evaluates or lets a
human play, and `rendering/` draws. `plot/`, `utils/` and `assets/` are
support.
