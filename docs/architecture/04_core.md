# 04 — Core

```text
core/
├── digestion/
│   ├── digestion.py
│   ├── engine.py
│   ├── negative_digester.py
│   └── tier_digester.py
├── droid/
│   └── synergy_droid.py
├── orbs/
│   ├── direct/
│   │   └── negative_orb.py
│   ├── effects/
│   ├── synergy/
│   │   └── tier_orb.py
│   ├── base_orb.py
│   └── orb_meta.py
├── utils/
│   └── timer.py
└── grid_world.py
```

```text
GridWorld
 ├── droid : SynergyDroid
 │     └── digestion_engine : DigestionEngine
 │            └── digesters (TierDigester, NegativeDigester)
 ├── ALL_ORBS : list[BaseOrb]
 └── _spawning : SpawningRules   (from the scenario, shared)
```

`core/` is the simulation, with no knowledge of Gymnasium spaces or training.
`GridWorld` is the top object: it holds the droid and the orbs and runs one
step — move the droid, tick orb timers, consume the orb under the droid, then
let the scenario's spawning rules act. It receives its droid, population and
spawning rules already built and contains no scenario-specific branches.

`SynergyDroid` owns position and score. It moves, applies step and boundary
penalties, and passes a consumed orb to its `DigestionEngine`. The engine turns
"an orb was consumed" into a reward by routing the orb to the digester that
owns its kind, letting the other digesters notice it, and counting the events
they report. `TierDigester` carries the chain state and the three scoring
modes; `NegativeDigester` simply pays the orb's own reward.

Orbs (`orbs/`) are passive objects with a position, a reward, an `OrbMeta`
describing what they are, and a `Timer` used for both lifespan and cooldown.
See `05_orbs.md`.
