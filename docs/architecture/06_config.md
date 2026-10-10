# 06 — Configuration

```text
config/
├── models/
│   ├── common_scenario_models.py
│   ├── global_models.py
│   └── runner_models.py
├── yaml/
│   ├── global_config.yaml
│   ├── goal_tier_chain_spatial.yaml
│   └── runner_config.yaml
└── config_manager.py

scenario/
├── goal/
│   ├── tier_chain/
│   │   └── config.py
│   └── config.py
└── registry.py
```

```text
ScenarioConf
├── world_conf : WorldConf
│     ├── max_steps
│     ├── grid_conf    : GridConf
│     ├── droid_conf   : DroidConf
│     └── neg_orb_conf : NegOrbConf       (optional)
└── obs_conf : ObsConf
      └── observation_handler_conf

GoalScenarioConf (ScenarioConf)
└── world_conf : GoalWorldConf (WorldConf)
      └── goal_conf : GoalConf

TierScenarioConf (GoalScenarioConf)
└── world_conf : TierWorldConf (GoalWorldConf)
      └── tier_orb_conf : TierOrbConf
```

```text
TierScenarioConf ──── TierDelayScenarioConf
└── world_conf ──────── TierDelayWorldConf
      └── tier_orb_conf ── TierDelayOrbConf      (delay)

TierScenarioConf ──── TierDenseScenarioConf
└── world_conf ──────── TierDenseWorldConf
      └── tier_orb_conf ── TierDenseOrbConf      (the reward ladder)
```

Configuration is three YAML files in `config/yaml/`, each validated by a
pydantic model through `ConfigManager`. `global_config.yaml` names the scenario
and sets `human_control` and `snapshot`; `runner_config.yaml` holds the
algorithm, seed and train/eval settings; the third file is named after the
scenario and holds its world and observation tunables. The scenario name picks
both that file and, through its entry in `SCENARIOS` in `scenario/registry.py`,
the model it is validated against. The same entry names the scenario's builder.

The models are split by how widely they are shared. `config/models/` holds what
is not specific to a scenario: the global and runner models, and
`common_scenario_models.py`, the vocabulary every scenario is written in: grid, droid,
orb kinds, the observation handler, and the base `ScenarioConf`. What only
some scenarios have lives in the scenario package, in a `config.py` beside what
it configures.

A scenario config has two halves. `world_conf` is everything the world acts
on: how long an episode lasts, the grid, the droid, the goal and the orbs.
`obs_conf` is what the Gymnasium side needs to turn the world into an
observation.

Each layer adds its own block to the world. `scenario/goal/config.py` adds
`goal_conf`, what reaching the objective pays and what missing the deadline
costs. `scenario/goal/tier_chain/config.py` adds `tier_orb_conf`, the chain
length and the chain-break penalty. Each orb kind
has its own `*_orb_conf` block in the world, holding what the kind's orbs and
its digester need, and the base world carries the optional `neg_orb_conf`. A
variant subclasses its block, then narrows the field on its own world model and
on its own scenario model: the delay variant adds `delay`, and the dense variant
adds the reward ladder. Dense and delay have models but no YAML yet.

Every model is frozen, forbids unknown keys and is strict. Single-field bounds
are `Field` constraints, and `model_validator`s cover the cross-field rules.
Any block that extends `PenaltyCheckedConf` rejects a positive value in a field
whose name ends in `_penalty`; `TierWorldConf` checks that the chain fits the
grid; the runner models check that render and video settings agree.

Scenario config is consumed almost entirely in `scenario/registry.py`, which
turns it into rule objects and world pieces. Below that, only `SynergyDroid`
and `NegativeOrb` still receive a config block directly.
