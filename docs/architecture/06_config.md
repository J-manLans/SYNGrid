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
ScenarioConf ─────────────── TierScenarioConf ──── TierDelayScenarioConf
├── world_conf : WorldConf ─── TierWorldConf ─────── TierDelayWorldConf
│     ├── grid_conf  : GridConf
│     ├── droid_conf : DroidConf ── GoalDroidConf ── TierDroidConf
│     └── orb_conf   : OrbPoolConf ─ TierOrbPoolConf ─ TierDelayOrbPoolConf
└── obs_conf : ObsConf
      └── observation_handler_conf
```

```text
TierScenarioConf ──── TierDenseScenarioConf
├── world_conf ──────── TierDenseWorldConf
│     └── orb_conf ────── TierDenseOrbPoolConf
│           └── tier ────── TierDenseOrbConf      (the reward ladder)
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
it configures. `scenario/goal/config.py` is what every goal scenario shares,
today `GoalDroidConf`. `scenario/goal/tier_chain/config.py` is the tier-chain
family, which extends those blocks by
subclassing and then narrowing a field's type in the containing model. The
delay variant adds `delay` to the tier orb; the dense variant adds the reward
ladder and a `perception_conf` block, and has a model but no YAML or builder
yet. Every
model is frozen, forbids unknown keys and is strict. Single-field bounds are
`Field` constraints, and `model_validator`s cover the cross-field rules
(penalties must not be positive, the chain must fit the grid, render and video
settings must agree).

Scenario config is consumed almost entirely in `scenario/registry.py`, which
turns it into rule objects and world pieces. Below that, only `SynergyDroid`
and `NegativeOrb` still receive a config block directly.
