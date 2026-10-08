# 05 — Orbs

```text
config/models/
├── common_models.py
└── tier_chain_models.py

core/digestion/
├── digestion.py
├── engine.py
├── negative_digester.py
└── tier_digester.py

core/orbs/
├── direct/
│   └── negative_orb.py
├── effects/
├── synergy/
│   └── tier_orb.py
├── base_orb.py
└── orb_meta.py

scenario/rules/
├── population.py
└── spawning.py
```

```text
BaseOrb (ABC)
├── TierOrb
└── NegativeOrb
```

```text
config (TierOrbConf.max_tier)
    ↓
TierChainPopulation.create()        one TierOrb per tier
    ↓
GridWorld.ALL_ORBS                  spawned by SpawningRules.on_reset
    ↓
SynergyDroid.consume_orb(orb)       droid lands on the orb's cell
    ↓
DigestionEngine.digest(orb)         routed by kind to TierOrbDigester
    ↓
reward + ChainProgressed / ChainBroken / ChainCompleted
```

Orb code is spread over four packages. `core/orbs/` defines what an orb is:
`BaseOrb` with position, reward, timer and active flag, and `OrbMeta` carrying
category, type, tier and the identity the agent sees. An orb's *kind* is its
`(category, type)` pair, and that is what digestion routes on.

There is no `OrbFactory`. Orbs are created by an `OrbPopulation` from
`scenario/rules/population.py`; the only one in use is `TierChainPopulation`,
which builds one `TierOrb` per tier. The builder gives it `max_tier` from
`TierOrbConf` and states the reward ladder and cool-down itself. `GridWorld`
calls it once at construction. When orbs appear, cool down or return is decided by
`SpawningRules`, which calls the world's spawn and reactivate methods.

An orb does not score itself. When the droid consumes one, the
`DigestionEngine` hands it to the digester for its kind, which returns the
reward and the chain events that termination and logging read. Negative orbs
have a class and a digester but no population creates them at present.
