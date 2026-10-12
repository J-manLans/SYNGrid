# 06 — Configuration

```text
config/
├── models/
│   ├── scenario/
│   │   ├── common_models.py
│   │   └── orb_models.py
│   ├── global_models.py
│   └── runner_models.py
├── yaml/
│   ├── global_config.yaml
│   ├── goal_tier_chain_spatial.yaml
│   └── runner_config.yaml
└── config_manager.py

scenario/
├── families/
│   └── goal/
│       ├── tier_chain/
│       │   └── config.py
│       └── config.py
└── registry.py
```

```text
ScenarioConf
├── world_conf : WorldConf
│     ├── max_steps
│     ├── grid_conf    : GridConf
│     └── droid_conf   : DroidConf
└── obs_conf : ObsConf
      └── observation_handler_conf

GoalScenarioConf (ScenarioConf)
└── world_conf : GoalWorldConf (WorldConf)
      └── goal_conf : GoalConf

TierScenarioConf (GoalScenarioConf)
└── world_conf : TierWorldConf (GoalWorldConf)
      ├── tier_orb_conf : TierOrbConf
      └── neg_orb_conf  : NegOrbConf     (optional)
```

```text
TierScenarioConf ──── TierDelayScenarioConf
└── world_conf ──────── TierDelayWorldConf
      └── tier_orb_conf ── TierDelayOrbConf      (delay)
```

Configuration is three YAML files in `config/yaml/`, each validated by a
pydantic model through `ConfigManager`. `global_config.yaml` names the scenario
and sets `human_control` and `snapshot`; `runner_config.yaml` holds the
algorithm, seed and train/eval settings; the third file is named after the
scenario and holds its world and observation tunables. The scenario name picks
both that file and, through its entry in `SCENARIOS` in `scenario/registry.py`,
the model it is validated against. The same entry names the scenario's builder.

The models are split by how widely they are shared. `config/models/` holds what
is not specific to a scenario: the global and runner models, and a `scenario/`
folder with what every scenario is built from. That is
`common_models.py`, the vocabulary every scenario is written in (grid, droid,
the observation handler and the base `WorldConf` and `ScenarioConf`), and
`orb_models.py`, the base class of each orb kind. What only some scenarios have
lives in the scenario package, in a `config.py` beside what it configures.

A scenario config has two halves. `world_conf` is everything the world acts
on: how long an episode lasts, the grid, the droid, the goal and the orbs.
`obs_conf` is what the Gymnasium side needs to turn the world into an
observation.

Each layer adds its own block to the world. `scenario/families/goal/config.py` adds
`goal_conf`, what missing the deadline costs.
`scenario/families/goal/tier_chain/config.py` adds `tier_orb_conf`, which is the chain
length, what each tier is worth and the chain-break penalty, and the optional
`neg_orb_conf`. Each orb
kind has its own `*_orb_conf` block in the world, holding what the kind's orbs
and its digester need. The base world declares none of them; a scenario's world
model declares the kinds it holds. A variant subclasses its block, then narrows
the field on its own world model and
on its own scenario model. The delay variant is the only one, and it adds
`delay`. Spatial, scaling sparse and scaling dense share `TierScenarioConf`.
Only spatial has a YAML; the other three registered scenarios have models and
no file yet.

A tier orb is worth `base_reward * tier ** growth_factor`, where a growth
factor of 1 is linear. `timeout_penalty` and `chain_break_penalty` are required
in every scenario, and no value has a default.

Every model is frozen, forbids unknown keys and is strict. Single-field bounds
are `Field` constraints, and `model_validator`s cover the cross-field rules.
Any block that extends `PenaltyCheckedConf` rejects a positive value in a field
whose name ends in `_penalty`; `TierWorldConf` checks that the chain fits the
grid; the runner models check that render and video settings agree.

Scenario config is consumed in the family's `builders.py`, which turns it into
rule objects and world pieces. Below that, `NegativeOrb` takes a `NegOrbConf`
directly. The rest of `core/`, `gymnasium/` and `rendering/` still name config
classes from before the rebuild, which `config/models/` no longer has.
