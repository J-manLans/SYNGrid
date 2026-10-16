# 04 — Core

```text
core/
├── droid/
│   ├── digestion/
│   │   ├── digesters/
│   │   │   ├── tier_chain/
│   │   │   │   ├── max_tier_digester.py
│   │   │   │   ├── step_wise_digester.py
│   │   │   │   ├── threshold_digester.py
│   │   │   │   └── tier_digester.py
│   │   │   └── neg_digester.py
│   │   ├── digester.py
│   │   └── engine.py
│   ├── droid_action.py
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
`DigestionEngine`. It still creates the engine from three config values, which
the engine no longer takes.

The digestion package is rebuilt, and nothing uses it yet. `digester.py` holds
the contract: the `OrbDigester` protocol of `KIND`, `reset` and a `digest` that
returns a reward. `engine.py` is a router built from digesters: it hands a
consumed orb to the digester of its kind, `OrbKind` in `orb_meta.py`, returns
a digester by its class to a reader, and raises when a digester is missing or
a kind has two. `digesters/tier_chain/` holds `TierDigester`, a base class
that tracks the chain, counts its progress, breaks and completions and sums
what its orbs are worth, and three subclasses that each say what progress, a
break and a completion pay, and what a running chain is owed when an episode
ends: `MaxTierDigester`, `StepWiseDigester` and `ThresholdDigester`.
`neg_digester.py` holds `NegDigester`, which pays a negative orb's own reward
and counts how many were eaten. Orbs are covered in `05_orbs.md`.
