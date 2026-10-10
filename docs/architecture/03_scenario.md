# 03 — Scenario

```text
scenario/
├── rules/
│   ├── observation.py
│   ├── population.py
│   ├── spawning.py
│   └── termination.py
├── registry.py
└── scenario.py
```

This package holds two different things. `scenario.py` and `rules/` are the
**representation**: the `Scenario` dataclass and the rule objects that describe
one scenario. `registry.py` is the **construction**: the builders that read a
scenario config and assemble those objects, and the function that builds a
world.

A `Scenario` is a frozen recipe: `scenario_name` and `scenario_tag`, the
`ObservationRules`, the termination rules, and a `build_world` callable. It
holds no world and no episode state, so one `Scenario` is shared by every
environment in the process. The rule objects in `rules/` are stateless too —
they decide what an observation is sized against, how the orb pool is
populated, when orbs spawn, and when an episode ends, and they are handed the
world as an argument whenever they need to look at it.

`build_scenario` looks the scenario name up in `SCENARIOS`, whose entry holds
the scenario's config class and its builder. The entry checks that the config
it is given is that class, then runs the builder, once at startup. Each
environment then calls `scenario.build_world()`, which creates that
environment's own digesters, `DigestionEngine`,
`SynergyDroid` and `GridWorld`. Only `goal_tier_chain_spatial` has a working
builder; the other three registered names are placeholders.
