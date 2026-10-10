# 06 — Configuration

```text
config/
├── models/
│   ├── common_models.py
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
│     ├── grid_conf  : GridConf
│     └── droid_conf : DroidConf
├── obs_conf : ObsConf
│     └── observation_handler_conf
└── neg_orb_conf : NegOrbConf       (optional)

GoalScenarioConf (ScenarioConf)
└── goal_conf : GoalConf

TierScenarioConf (GoalScenarioConf)
└── tier_orb_conf : TierOrbConf
```

```text
TierScenarioConf ──── TierDelayScenarioConf
└── tier_orb_conf ──── TierDelayOrbConf          (delay)

TierScenarioConf ──── TierDenseScenarioConf
├── tier_orb_conf ──── TierDenseOrbConf          (the reward ladder)
└── obs_conf ────────── TierDenseObsConf
      └── perception_conf : PerceptionConf        (max_score)
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
`common_models.py`, the vocabulary every scenario is written in: grid, droid,
orb kinds, the two observation blocks, and the base `ScenarioConf`. What only
some scenarios have lives in the scenario package, in a `config.py` beside what
it configures.

Each layer adds its own block at the top of the scenario config, beside
`world_conf` and `obs_conf`, so the world and the droid are the same models in
every scenario. There is no single orb block: each orb kind has its own
`*_orb_conf` block at that level, holding what the kind's orbs and its digester
need. The base model carries the optional `neg_orb_conf`. `scenario/goal/config.py` adds `goal_conf`, what reaching the
objective pays and what missing the deadline costs.
`scenario/goal/tier_chain/config.py` adds `tier_orb_conf`, the chain length and
the chain-break penalty. A variant subclasses that block and narrows
`tier_orb_conf` on its own scenario model: the delay variant adds `delay`, and the dense variant
adds the reward ladder and a `perception_conf` block. Dense and delay have
models but no YAML yet.

Every model is frozen, forbids unknown keys and is strict. Single-field bounds
are `Field` constraints, and `model_validator`s cover the cross-field rules.
Any block that extends `PenaltyCheckedConf` rejects a positive value in a field
whose name ends in `_penalty`; `TierScenarioConf` checks that the chain fits the
grid; the runner models check that render and video settings agree.

Scenario config is consumed almost entirely in `scenario/registry.py`, which
turns it into rule objects and world pieces. Below that, only `SynergyDroid`
and `NegativeOrb` still receive a config block directly.
