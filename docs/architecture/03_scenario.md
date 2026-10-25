# 03 — Scenario

```text
scenario/
├── blocks/
│   ├── populations/
│   │   ├── neg_population.py
│   │   ├── population.py
│   │   └── tier_chain_population.py
│   ├── recipes/
│   │   ├── tier_digester_recipe.py
│   │   └── world_recipe.py
│   ├── hud.py
│   ├── metrics.py
│   ├── observation.py
│   ├── orb_bundle.py
│   ├── readers.py
│   └── termination.py
├── families/
│   ├── continuous/
│   └── goal/
│       ├── single_tier_chain/
│       │   ├── builders.py
│       │   ├── config.py
│       │   └── termination.py
│       └── config.py
├── registry.py
└── scenario.py
```

This package holds three different things. `scenario.py` and `blocks/` are the
**representation**: the `Scenario` dataclass and the pieces one scenario is made
of. `registry.py` is the **selection**: the `SCENARIOS` table, whose entry gives
a scenario name its config class and its builder. `families/` is the
**construction**: under a group such as `goal/`, a family folder such as
`single_tier_chain/` holds its config models in `config.py`, its own implementation
of the termination interface in `termination.py`, and, in `builders.py`, the
builders that turn a config into a `Scenario`.

`blocks/` holds everything a family does not own. Its files are the interfaces
and the rule classes. Its two folders are the shared implementations:
`populations/` has the population protocol and the classes that make a kind's
orbs, and `recipes/` has the frozen objects that make something with episode
state when called, a tier digester or a world.

A `Scenario` is a frozen description: `scenario_name` and `scenario_tag`, the
`ObservationRules`, the termination rules, the `hud` elements, the `metrics`,
and `build_world`, a `WorldRecipe`. It holds no world and no episode state, so one
`Scenario` is shared by every environment in the process. The blocks are
stateless too. A HUD element and a metric are a static description plus a
reader that is handed the world. The readers every scenario can use are in
`blocks/readers.py`, and a family's own readers sit beside its builder. An `OrbBundle` is one orb kind: the population
that makes its orbs and a way to make its digester, so the world builder joins
the orbs and collects the digesters without asking which kinds they are.

`build_scenario` looks the scenario name up in `SCENARIOS`. The entry checks
that the config it is given is its config class, then runs the builder, once at
startup. Each environment then calls `scenario.build_world()` for a world of
its own. Four single tier chain names are registered. `build_single_tier_chain_spatial`
returns a `Scenario`, and the other three builders raise `NotImplementedError`.
Its bundles make the orbs and the digesters, and two scenarios built from the
same config compare equal. Its termination and its world readers are written,
against an `energy` and a `get_digester` method the droid does not have yet.
Calling `build_world` raises until `core/` takes pieces instead of config.
