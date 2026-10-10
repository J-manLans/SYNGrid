# Design Decisions

Choices made on purpose during the rebuild, with the options that were turned down and the reason. The other files in
this folder record things that happened (a break, a mismatch, a deferral, a touch). This one records what was chosen,
so that later nobody has to guess why something is shaped the way it is or put back what was left out deliberately.

Each entry says when it would be worth reopening.

---

## Entry template

### Title (date)

**Decision:**
**Where:**
**Alternatives considered:**
**Why this one:**
**Status:** settled / revisit if X

---

## Digestion lives in the droid, with a folder of digesters (2026-10-09)

**Decision:** Digestion is the droid's own organ. The package holds three
things:

- `digestion.py`, the interface: `Event`, `DigestionResult`, `OrbKind`,
  `kind_of()` and the `OrbDigester` protocol.
- `engine.py`, a router with a tally. It holds no scoring rules.
- `digesters/`, every digester there is, one per orb kind.

The droid owns and creates the engine and get's the `make_digesters` handed to it (I suppose) and calls the engine when
it consumes an orb. It no longer builds the engine from config. `build_world` builds it from the digesters the
scenario's orb bundles supply, and hands it to the droid (Here I need to decide if build world should do the whole pass, or lets the droid create the engine, or if the scenario creates it and hands it to the droid, I wrote that the droid should do the creation, and thats because its one of its organs, so it makes logical sense to me, but if that complicate things, that can be changed).

**Where:** `src/syn_grid/core/droid/digestion/`

**Alternatives considered:**

- `core/digestion/`, the path the reference skeleton imports from. Rejected because digesting is something the droid
does.
- Digesters living in the scenario package, next to the scenario termination and spawning etc. Rejected because
digesters belong with the digester.

**Why this one:** The engine stays a plain mechanism that any scenario family can use. The scenario decides which
digesters to build and how each is configured, through the bundle's `make_digester`, but it doesn't own their code.

**Status:** settled. `NegativeDigester` follows the same rule and goes in `digesters/`; it hasn't been read or ported
yet, so check how much it knows about tiers when it is.

---

## The tier digester is a template with one subclass per scoring mode (2026-10-09)

**Decision:** The tier digester is a base class that owns the chain tracking: if the orb is the next tier the chain
progresses, or completes at max tier; otherwise it breaks. Each scoring mode is a subclass (or using the template method?) that fills in what is paid at those three moments:

| | On progress | On break | On completion |
|---|---|---|---|
| Step-wise | that tier's reward, right away | small wrong-order penalty | its reward plus the earlier tiers' total |
| Threshold | nothing, held back | the held reward | everything accumulated |
| Max-tier | nothing | chain-break penalty | completion reward |

There is no `ScoringMode` enum. The scenario's builder picks the subclass. Spatial needs only max-tier, so only the
base and that subclass get written first (or perhaps I keep the enum, but for the fixed scenarios I pass the certain mode in the builder method in scenario, keep it out of the config, but for the sandbox continuous scenario for example, it can be a tunable parameter).

**Where:** `core/droid/digestion/digesters/`

**Alternatives considered:**

- One digester with a `ScoringMode` enum and a branching method per mode (the reference). Rejected because the builder
already decides the mode, so the enum states the same decision a second time, and it is the kind of conditional that
does not belong in the component holding it (but read the parenthesis above for reservations about this, and even if
the `ScoringMode` enum persists, its only used as a router as to what template or subclass to create).
- The orb carrying its own scoring (the old engine). Rejected because the engine then has to inspect orbs to know how
to score them and they should be the same across the board, how the droid digest them should be up to its own digestion
system.

**Why this one:** Adding a mode is adding a class, not editing one that works. A scenario only loads the mode it uses,
and each mode can be tested on its own.

**Watch:** Threshold breaks differently from the other two. Consecutive tier-1 pickups don't count as breaks, and the
chain restarts only if a reward is held. So the break step has to be a hook a subclass can override, or the base class
can't share it. Folding that quirk into the base would change behaviour and move training away from the baseline. I
think this is a residue from the thesis era stress. It has to do with it working for both a goal scenario and a
continuous scenario, for a chain break in a goal scenario means game over, in a continuous one it just means you get no
reward for this, or in the threshold mode — you get the reward up to this orb only. So look over this for all modes
(here the new structure of the two top scenario types — goal and continuous — can come in handy as a distinguisher):

**Status:** settled for spatial. Revisit if the shared base turns out not to fall cleanly out of the second mode once
it is written.

---

## A new scoring mode is a new scenario (2026-10-09)

**Decision:** Scoring mode is generally not a config setting when it comes to locked down scenarios (continuous sandbox
mode the exception). A different score-mode means a new registered scenario with its own builder, reusing the shared
tier-chain helper, and its own config model if the inputs differ. Scaling sparse and scaling dense already work this
way.

**Where:** `scenario/(goal|continuous)/*/builders.py`, `scenario/registry.py`, `config/models/*_models.py`

**Alternatives considered:** The mode in the YAML, with a lookup from its name to a digester class. Rejected for now
(allowed for continuous sandbox scenario).

**Why this one:**

- The scenario name is the run's identity: run IDs, result folders and any comparison hang on it. A mode set in config
would let two runs of the same name pay differently.
- The config models enforce the inputs. Max-tier needs a completion reward and tier orbs worth nothing; threshold and
step-wise need a reward per tier, which only the dense models carry. A switchable mode would need runtime checks for
what the types now guarantee.

**Status:** settled. Revisit if a single experiment needs to sweep scoring modes. The lookup would then sit at the
edge, next to `SCENARIO_MODELS` or in the builders, and wouldn't touch the digesters.
