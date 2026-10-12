# 05 — Orbs

```text
config/models/scenario/
└── orb_models.py

core/droid/digestion/
├── digesters/
│   ├── tier_chain/
│   │   ├── max_tier_digester.py
│   │   ├── step_wise_digester.py
│   │   ├── threshold_digester.py
│   │   └── tier_digester.py
│   └── neg_digester.py
├── digestion.py
└── engine.py

core/orbs/
├── direct/
│   └── negative_orb.py
├── synergy/
│   ├── chain/
│   │   └── tier_orb.py
│   └── effect/
├── base_orb.py
├── orb_factory.py
└── orb_meta.py

scenario/blocks/
├── populations/
│   ├── neg_population.py
│   ├── population.py
│   └── tier_chain_population.py
├── orb_bundle.py
└── spawning.py

scenario/families/goal/tier_chain/
└── config.py
```

```text
BaseOrb (ABC)
├── NegativeOrb
└── TierOrb
```

```text
NegOrbConf, TierOrbConf                  orb_models.py, one base class per kind
    ↓
TierWorldConf                            declares tier_orb_conf and neg_orb_conf
    ↓
OrbBundle(population, make_digester)     one per kind, made by the builder
    ↓
build_world                              joins the orbs, collects the digesters
```

Orb code is spread over four packages. `core/orbs/` defines what an orb is:
`BaseOrb` with position, reward, timer and active flag, and `OrbMeta` carrying
the kind, the tier of a tier orb and the identity the agent sees. The kind is
what digestion routes on. Config for a kind is a
base class in `orb_models.py`, extended by a family that needs more and declared
as a field on the world model of each scenario that holds the kind.

In the scenario layer one orb kind is an `OrbBundle`: an `OrbPopulation` that
makes the orbs and a way to make the kind's digester. The chain above is written
down to the bundle. In `scenario/blocks/populations/`, `TierChainPopulation` makes one
tier orb per tier and `NegPopulation` makes one negative orb. The builders'
negative bundle pairs `NegPopulation` with `NegDigester`, and their tier bundle
pairs `TierChainPopulation` with a `TierDigesterRecipe` from
`scenario/blocks/recipes/`, which holds the scoring mode's class and makes a new
digester when called. `build_world` is a `WorldRecipe`, and calling it raises
before it joins anything. The tier
digesters themselves are written in
`core/droid/digestion/digesters/tier_chain/`, a base class and one subclass
per scoring mode.

`core/` does not use any of that yet. `GridWorld` creates its orbs through
`OrbFactory` from config classes that no longer exist, and the droid creates
its `DigestionEngine` from config values where the engine now takes
digesters. Both orbs take plain values: `NegativeOrb` its reward and cool-down,
and `TierOrb` its tier, base reward, growth factor and cool-down, from which it
works its reward out.
