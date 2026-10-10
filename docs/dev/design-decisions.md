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

**Where:** `scenario/(goal|continuous)/*/builders.py`, `scenario/(goal|continuous)/*/config.py`, `scenario/registry.py`

**Alternatives considered:** The mode in the YAML, with a lookup from its name to a digester class. Rejected for now
(allowed for continuous sandbox scenario).

**Why this one:**

- The scenario name is the run's identity: run IDs, result folders and any comparison hang on it. A mode set in config
would let two runs of the same name pay differently.
- The config models enforce the inputs. Max-tier needs a completion reward and tier orbs worth nothing; threshold and
step-wise need a reward per tier, which only the dense models carry. A switchable mode would need runtime checks for
what the types now guarantee.

**Status:** settled. Revisit if a single experiment needs to sweep scoring modes. The lookup would then sit at the
edge, next to `SCENARIOS` or in the builders, and wouldn't touch the digesters.

---

## Two orb categories, and an orb meta that stores only type and tier (2026-10-10)

**Decision:** There are two orb categories and there will not be a third.

- **Direct:** changes the score and nothing else, on the step it is eaten. Penalty and reward orbs. The only variety is
  how the number is decided: fixed, a random draw, or a value that depends on the orb's own age.
- **Synergy:** everything else. Eating it leaves state behind that changes a later outcome. A tier orb leaves a chain,
  a timed orb leaves a countdown, an item orb sits in the droid until the next orb, a world orb changes the field.

The test for a new orb: does its digester need anything but the orb itself? If it needs the droid, the world or a
memory of earlier orbs, it is synergy. So an orb bomb, growing the droid, a teleport, an instant game over and a
percentage of the current score are all synergy, even though they happen at once.

`OrbMeta` says what an orb is, not how it acts. It stores two things:

- `type`, one member of `DirectType` or `SynergyType`. Every orb that behaves differently gets its own member
  (`NEGATIVE`, `POSITIVE`, ... and `TIER`, `FLIP_REWARD`, `OBSCURE`, `NULLIFY`, `HAZARD_BORDER`, ...).
- `tier`, an int for `SynergyType.TIER` and `None` for everything else.

The category is not stored. It follows from which enum the type comes from.

Inside synergy there are two groups. They are folders, not types, enums or categories:

- **Chain:** the payoff depends on which orbs of the same kind came before, and in what order. The tier orb is the
  only one so far; a streak or a collect-the-set orb would be others.
- **Effect:** eating it changes something other than itself: other orbs' rewards, the observation, the next orb, the
  world.

In digestion the routing key is the type alone: `OrbKind = DirectType | SynergyType` and `kind_of(orb)` returns
`orb.META.TYPE`. The engine and the digesters treat it as an opaque key and never take it apart.

**Where:** `core/orbs/orb_meta.py`, and what reads it: `core/grid_world.py`, `rendering/pygame_renderer.py`,
`gymnasium/observation_space/perceptions/`. The orbs are packaged as `core/orbs/direct/`, `core/orbs/synergy/chain/`
(holding `tier_orb.py`) and `core/orbs/synergy/effect/`. The routing key lives in
`core/droid/digestion/digestion.py`.

**Alternatives considered:**

- A third `EffectType` enum beside `SynergyType`. Rejected because it reads as if effect orbs were not synergy orbs,
and nothing in the code needs to ask "is this an effect?".
- One `SynergyType.EFFECT` member with a sub-field for which effect. Rejected because each effect needs its own
digester, so the routing key has to tell them apart; this would force a three-part key or one effect digester that
branches inside.
- Storing how an effect acts (timed, item, world) in the meta. Rejected because no reader of the meta needs it. Those
are capabilities of the effect's digester: being ticked, acting before the owning digester runs, or sending something
out for the world to act on.
- Direct meaning "happens right away". Rejected because an orb bomb or a teleport happens right away and still changes
more than the score, which blurs the category.
- Leaving `tier_orb.py` loose at the top of `synergy/` until a second chain orb exists. Rejected because the `chain/`
folder tells a later reader how synergy is divided, even with one file in it.
- Keeping `CATEGORY` as a stored field. Rejected because the enum already says it, and the validator in `OrbMeta`
exists only to check that the two stored fields agree.

**Why this one:** Direct stays a small closed set and every interesting mechanic is synergy, which lines up with the
digester split: a direct digester is stateless, a synergy digester holds episode state. Effect orbs no longer need a
tier. A new orb is a new enum member and a digester, with no new field on the meta.

**Status:** settled as a design, not built. Digestion is written first and already uses the type alone as its key, so
it has nothing to update later. The meta itself is redone in its own pass, with the observation or just before it,
and as the only change in that step so the observation can be compared with legacy. Things that pass has to deal
with:

- `grid_world.py` uses `TIER == 0` to mean "not a tier orb" when deciding which orbs despawn. With `tier` as `None`
and more synergy types this has to become a check on the type.
- The renderer's `orb_meta.TIER is not None` is always true today, because a missing tier is stored as 0.
- The radix `IDENTITY` is only a sort key; the world replaces it with dense ids when it is built. It could become a
plain sort on type and tier. Negative has to keep sorting before the tiers or the existing observations change.
- The world writes the dense id back onto the meta, so the meta can't be frozen until that id lives somewhere else.

Open: whether synergy orbs other than tier orbs despawn and respawn like negative orbs or stay on the field. Revisit
the "two categories" rule only if an orb turns up that fits neither test.

---

## Scenario configs live with the scenario, as blocks in the world (2026-10-10)

**Decision:** A config model lives in the folder of the thing it configures, at the level where it is shared.

- `config/models/` keeps what is not specific to a scenario: the global and runner models, and
  `common_scenario_models.py` with the grid, the droid, the negative orb, the observation handler and the base
  `WorldConf` and `ScenarioConf`.
- `scenario/goal/config.py` holds what every goal scenario has. `scenario/continuous/config.py` will do the same for
  continuous.
- `scenario/goal/tier_chain/config.py` holds the tier-chain family.

A scenario config has two halves. `world_conf` is everything the world acts on. `obs_conf` is what the Gymnasium side
needs to turn the world into an observation, which the world never reads. Each layer adds its own block to the world:

```
ScenarioConf                    world_conf, obs_conf
  WorldConf                       max_steps, grid_conf, droid_conf, neg_orb_conf (optional)
    GoalWorldConf                 + goal_conf       (timeout_penalty, completion_reward)
      TierWorldConf               + tier_orb_conf   (max_tier, chain_break_penalty)
        TierDelayWorldConf          tier_orb_conf   + delay
        TierDenseWorldConf          tier_orb_conf   + the reward ladder
```

Each world model has a scenario model that narrows `world_conf` to it (`GoalScenarioConf`, `TierScenarioConf`,
`TierDelayScenarioConf`, `TierDenseScenarioConf`). A variant is three classes: its block, its world and its scenario
model.

There is no single orb block to look at. Each orb kind has its own `*_orb_conf` block in the world, holding what that
kind's orbs and its digester need. This mirrors `OrbBundle` in the scenario layer, where a kind's orbs and its
digester travel together, so a kind is one brick in the config as well.

`scenario/registry.py` holds one table, `SCENARIOS`, whose entry gives a name its config class and its builder.

**Where:** `config/models/common_scenario_models.py`, `scenario/goal/config.py`, `scenario/goal/tier_chain/config.py`,
`scenario/registry.py`, `config/yaml/goal_tier_chain_spatial.yaml`

**Alternatives considered:**

- A separate config tree that mirrors the scenario package. Rejected because two parallel trees can drift apart and
one can't.
- Goal and tier values on the droid (`GoalDroidConf`, `TierDroidConf`), with the orbs in one `orb_conf` pool under
`world_conf`. This is what was there. Rejected because the digester and termination read those values, not the droid,
and because one new leaf field took four classes (orb, orb pool, world, scenario).
- The goal and orb blocks at the top of the scenario config, beside `world_conf`. Tried first, since a variant is then
two classes instead of three. Rejected because the world acts on the goal and on the orbs, so they are part of it; the
top level is kept for the real divide, between the world and how it is observed.
- A `mode: goal` field. Rejected because the scenario name already says which type it is.

**Why this one:** The config tree and the scenario tree are the same tree, so a scenario's builder and its config are
found in one folder. `world_conf` means what its name says. The orb pool level is gone, so a variant costs one class
less than before, and each orb kind can be varied without touching the others.

**Status:** settled. Still open:

- `obs_conf` is two levels of nesting around one field, `perception`.
- The builders still take the base `ScenarioConf` and `cast` it.

---

## Energy and score are two numbers (2026-10-10)

**Decision:** The droid has an energy and a score. Both move with every reward and penalty.

- **Energy** is what the droid has left. It starts at `max_energy`, stays between 0 and `max_energy`, and the episode
  ends when it reaches 0, in every scenario. The droid starts fully charged and cannot be overcharged, so one config
  value is both the start and the cap.
- **Score** is how the episode went. It always starts at 0, has no bounds and no config.

They move together until the cap bites: a reward at full charge raises the score but not the energy. The score cannot
fall much below minus `max_energy`, because the episode ends first; the last penalty can overshoot by its own size.

The observation carries energy, in the slot where the old code carried its score. Its bounds are exactly 0 and
`max_energy`, so there is no separate bound to configure and nothing to clip.

**Where:** `DroidConf.max_energy` in `config/models/common_scenario_models.py`, replacing `starting_score`. The droid,
the base termination rule and the observation still have to be written to match.

**Alternatives considered:**

- One number, as before: it started at `starting_score`, was clipped at 0 and ended the episode there, so it was a
score in name and a life in behaviour. Rejected because the two meanings want different bounds and a different start.
- Only penalties move the energy, rewards only move the score. Rejected: energy should respond to both.
- Energy with no upper cap, and a `max_score` in the config to bound and clip its observation. This is what was there.
Rejected because a droid that cannot be overcharged gives an exact bound, which removed `PerceptionConf`,
`TierDenseObsConf` and `max_score`.
- "Life" as the name. Rejected in favour of "energy", which suits a droid.

**Why this one:** Energy is the old number under an honest name, so the four goal scenarios behave as before: in each
of them a positive reward only arrives on the step that ends the episode, so the cap is never reached. The score is
new and changes no behaviour.

**Status:** settled. Two things to carry forward:

- Continuous changes on purpose when it is rebuilt: the old number could grow without limit, so a droid could bank
rewards against later penalties. With a cap it can't.
- In the observation pass, check whether anything scales the observation by its upper bound. The values in that slot
are unchanged for the goal scenarios, but the bound moves from the old `max_score` to `max_energy`.

---

## The step clock lives in the world (2026-10-10)

**Decision:** `max_steps` is a field of `world_conf`, and the world is what counts the steps of an episode. Anything
that needs the steps left (termination, the observation, the HUD, the metrics) reads it from the world it is handed.

**Where:** `WorldConf.max_steps` in `config/models/common_scenario_models.py`. The count itself still has to move
into `GridWorld`; it is on `ObservationHandler` today.

**Alternatives considered:**

- In `obs_conf.observation_handler_conf`, where it was. Rejected because the observation is only one of its readers,
and a reader that is handed just the world could not get at it.
- At the top of the scenario config, beside `world_conf`. Rejected because the steps taken are state that changes
during an episode, and that state belongs in the world.
- In `grid_conf`. Rejected because that block is the geometry; the episode length is about time and is set
independently of the grid's shape.
- In `goal_conf`. Rejected because a continuous episode also ends on the clock.

**Why this one:** It follows the rule that a `Scenario` holds nothing that changes during an episode and that rules
reach state through the world. It is still per scenario, since the whole file is.

**Status:** settled for the config. Revisit if the clock turns out to need pausing or resetting by something other
than the world, such as a delay.

---

## Goal and chain values are additive (2026-10-10)

**Decision:** The three values that end or interrupt a goal tier chain are each added to what is already being paid,
by the same rule in every scenario. Nothing asks which scoring mode is in use.

| Moment | Pays |
|---|---|
| Timeout | the reward the chain was holding + `timeout_penalty` |
| Chain break | the reward the chain was holding + `chain_break_penalty` |
| Completion | what digestion paid for the orbs + `completion_reward` |

Goal termination pays `timeout_penalty` and `completion_reward`, the two ways a goal episode can end. The tier
digester pays for the orbs and for breaking the chain; on completion it reports the chain as complete and pays only
what the orbs were worth.

`timeout_penalty` and `completion_reward` are in `goal_conf`, because any scenario with an objective and a deadline
has them, whatever its orbs are. `chain_break_penalty` is in `tier_orb_conf`. All three are required, except in dense,
where they default to 0 and can be left out of the YAML.

**Where:** `GoalConf` in `scenario/goal/config.py`; `TierOrbConf`, `TierDenseOrbConf` and `TierDenseGoalConf` in
`scenario/goal/tier_chain/config.py`. The termination and digester code still has to be written to this rule, and the
tier digester has to expose the held reward (0 under max-tier scoring) for termination to read.

**Alternatives considered:**

- A rule per scoring mode, as the old termination had: under max-tier a timeout paid the penalty, under threshold it
paid the held reward. Rejected because it puts a scoring-mode conditional in termination.
- Moving the three values into a max-tier scoring block, since dense never used them. Rejected because a timeout
penalty is a goal-level idea: a goal scenario with other orbs would use it too.
- Optional fields, `float | None = None`. Rejected because `None` either means the same as 0 or hides a second rule
behind a value.
- A default of 0 on the shared fields. Rejected because a `timeout_penalty` forgotten in a max-tier YAML would then
load and train with none. The defaults are on dense's own classes only.
- Requiring the values in dense and writing zeros in its YAML. Rejected because the YAML should list what is meant to
be tuned.

**Why this one:** Max-tier scenarios behave as before, because nothing is ever held and the orbs are worth nothing.
Dense with the defaults reproduces the thesis runs (`reproduction_package/tier_scaling_scenario/dense/`), where a
timeout and a break paid the held reward and a completion paid the accumulated ladder. The thesis dense configs set
`chain_break_penalty: -0.1`, but threshold scoring never read it.

**Status:** settled. Revisit if a scenario needs a timeout or a completion to replace the step's reward instead of
adding to it.

---

## A weight belongs to a sampled field, and a tier chain holds one negative orb (2026-10-10)

**Decision:** A field is filled in one of two ways, and that is a separate question from goal or continuous.

- **Fixed:** every orb is stated and present. The tier chain is this kind.
- **Sampled:** orbs are drawn at random from a pool, and each kind's `weight` sets the mix.

`weight` is a field of an orb kind's block only in a family whose field is sampled. `NegOrbConf` is `reward` and
`cool_down`, which every negative orb has. A family with a sampled field uses a weighted version of the block that
adds `weight`, and other kinds in that family carry a weight on their own block the same way.

A tier chain with a negative block holds exactly one negative orb. It spawns, despawns at the end of its lifespan and
returns elsewhere after its cool-down. The count is not config.

**Where:** `NegOrbConf` in `config/models/common_scenario_models.py`. The weighted version is not written yet; it goes
in the same file when the first sampled family is built.

**Alternatives considered:**

- `weight` on every negative orb block, as before (`OrbKindConf`). Rejected because a weight means nothing in a fixed
field, so a tier-chain config had to supply a value nothing read.
- `weight` as something only continuous scenarios have. Rejected because a goal scenario can have a sampled field
too: tier orbs spawning in and out beside a separate objective, a target score to reach before the deadline, or a
fixed goal orb surrounded by a random stream of hazards.
- One pool block listing every kind's weight. Rejected because a kind's block should stay one self-contained brick,
and a separate list could name a kind that has no block.
- A configurable number of negative orbs in a tier chain. Rejected: one is enough.

**Why this one:** Each block holds only what its family reads. A weight is only meaningful relative to the other
weights in the same pool, so it appears exactly where there is a pool.

**Status:** settled for the tier chain. Revisit when the first sampled family is built, and for delay, where the
negative orb is meant to freeze and return with the field.
