# 03 — Scenario

```text
scenario/
├── blocks/
│   ├── hud.py
│   ├── metrics.py
│   ├── observation.py
│   ├── orb_bundle.py
│   ├── population.py
│   ├── spawning.py
│   └── termination.py
├── goal/
│   ├── tier_chain/
│   │   ├── builders.py
│   │   ├── config.py
│   │   ├── population.py
│   │   └── termination.py
│   └── config.py
├── registry.py
└── scenario.py
```

This package holds three different things. `scenario.py` and `blocks/` are the
**representation**: the `Scenario` dataclass and the pieces one scenario is made
of. `registry.py` is the **selection**: the `SCENARIOS` table, whose entry gives
a scenario name its config class and its builder. The folders below it, such as
`goal/tier_chain/`, are the **construction**: a family's config models in
`config.py`, its own implementations of the block interfaces in
`population.py` and `termination.py`, and, in `builders.py`, the builders that
turn a config into a `Scenario` and the function that builds a world.

A `Scenario` is a frozen recipe: `scenario_name` and `scenario_tag`, the
`ObservationRules`, the termination rules, the `hud` elements, the `metrics`,
and a `build_world` callable. It holds no world and no episode state, so one
`Scenario` is shared by every environment in the process. The blocks are
stateless too. A HUD element and a metric are a static description plus a
reader that is handed the world. An `OrbBundle` is one orb kind: the population
that makes its orbs and a way to make its digester, so the world builder joins
the orbs and collects the digesters without asking which kinds they are.

`build_scenario` looks the scenario name up in `SCENARIOS`. The entry checks
that the config it is given is its config class, then runs the builder, once at
startup. Each environment then calls `scenario.build_world()` for a world of
its own. Four tier-chain names are registered. `build_tier_chain_spatial`
returns a `Scenario`, and the other three builders raise `NotImplementedError`.
The pieces the spatial scenario is made of are stubs: the population makes no
orbs, the termination never ends an episode, the spawning hooks have no bodies,
and `build_world` raises until `core/` takes pieces instead of config.
