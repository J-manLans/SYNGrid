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
│   │   ├── common_models.py
│   │   ├── global_models.py
│   │   ├── runner_models.py
│   │   ├── scenario_registry.py
│   │   └── tier_chain_models.py
│   ├── scenarios/
│   ├── yaml/
│   │   ├── global_config.yaml
│   │   ├── goal_tier_chain_spatial.yaml
│   │   └── runner_config.yaml
│   └── config_manager.py
│
├── core/
│   ├── digestion/
│   │   ├── digestion.py
│   │   ├── engine.py
│   │   ├── negative_digester.py
│   │   └── tier_digester.py
│   ├── droid/
│   │   └── synergy_droid.py
│   ├── orbs/
│   │   ├── direct/
│   │   │   └── negative_orb.py
│   │   ├── effects/
│   │   ├── synergy/
│   │   │   └── tier_orb.py
│   │   ├── base_orb.py
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
│   │   └── env_factory.py
│   ├── action_space.py
│   └── environment.py
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
│   │   ├── agent_registry.py
│   │   ├── base_agent_runner.py
│   │   └── runner_bundle.py
│   └── human_runner/
│       └── human_runner.py
│
├── scenario/
│   ├── rules/
│   │   ├── observation.py
│   │   ├── population.py
│   │   ├── spawning.py
│   │   └── termination.py
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
and the runner creates the Gymnasium environments in `gymnasium/`. Each
environment asks the scenario for its own world, which is made of the pieces in
`core/`.

`scenario/` is the only package that turns config into simulation objects.
`core/` is the simulation itself, `gymnasium/` wraps one world as an
environment and encodes observations, `runners/` trains, evaluates or lets a
human play, and `rendering/` draws. `plot/`, `utils/` and `assets/` are
support.
