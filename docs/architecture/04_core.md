# 04 — Core

```text
core/
├── droid/
│   ├── digestion/
│   │   ├── digesters/
│   │   │   ├── neg_digester.py
│   │   │   └── tier_digester.py
│   │   ├── digestion.py
│   │   └── engine.py
│   └── synergy_droid.py
├── orbs/
│   ├── direct/
│   │   └── negative_orb.py
│   ├── synergy/
│   │   ├── chain/
│   │   │   └── tier_orb.py
│   │   └── effect/
│   ├── base_orb.py
│   ├── orb_factory.py
│   └── orb_meta.py
├── utils/
│   └── timer.py
└── grid_world.py
```

```text
GridWorld
 ├── _conf : GridWorldConf
 ├── droid : SynergyDroid
 │     └── digestion_engine : DigestionEngine
 └── ALL_ORBS : list[BaseOrb]    (from OrbFactory)
```

`core/` is the simulation, with no knowledge of Gymnasium spaces or training.
It is still the code from before the rebuild. `GridWorld` builds its own droid
and, through `OrbFactory`, its own orbs from five config blocks, and what
differs between scenarios is flags on its config: `single_chain_mode`,
`delay_mode` and `de_spawn_tiers`. Those config classes are no longer in
`config/models/`, so `grid_world.py`, `synergy_droid.py` and `orb_factory.py` do
not import.

One step moves the droid, ticks the orb timers, despawns an orb whose lifespan
ran out, and lets the droid consume the orb it stands on. Depending on the
flags, the world then puts the whole field on cooldown and brings it back, or
spawns one more orb.

`SynergyDroid` owns its position and one score, clipped at 0. It moves, applies
the step and boundary penalties, and passes a consumed orb to its
`DigestionEngine`. The engine is one class holding the chain state, the three
scoring modes and the episode's chain counts. It picks the mode from flags on
the tier orb, which `TierOrb` no longer has. `digestion.py` holds the contract
the engine is to be rebuilt on: the routing key, the result of digesting an orb
and the `OrbDigester` protocol. The two files in `digesters/` are empty. Orbs
are covered in `05_orbs.md`.
