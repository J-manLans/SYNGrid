# Design Decisions

Choices made on purpose during the rebuild, with the options that were turned down and the reason.
`open-questions.md` lists what is not decided yet, and `regressions.md` what has broken. This one records what was
chosen, so that later nobody has to guess why something is shaped the way it is or put back what was left out
deliberately.

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

**Decision:** Digestion is the droid's own organ. The package holds three things:

- `digestion.py`, the contract: the `OrbDigester` protocol, which is `KIND`, `reset` and `digest`. `KIND` is the
  routing key, an `OrbKind` from `core/orbs/orb_meta.py`. `digest` returns the reward as a float.
- `engine.py`, a router. It holds no scoring rules: it hands an eaten orb to the digester registered for the orb's
  kind, and it lets a reader fetch a digester by its class.

  The engine raises on every sign that the digesters were put together wrong: a `ValueError` when two digesters own
  the same kind, a `KeyError` when an eaten orb's kind has no digester, and a `KeyError` when `get` is asked for a
  class no registered digester is an instance of. `get` matches with `isinstance`, so asking for `TierDigester`
  finds whichever scoring mode is registered.
- `digesters/`, every digester there is, one per orb kind. A kind with several files gets a folder, as the tier
  digester has in `tier_chain/`.

The droid creates the engine. The scenario supplies a way to make each digester, one per `OrbBundle`
(`make_digester`). `build_world` calls each of them, so every world gets digesters of its own, and hands them to the
droid, which builds its engine from them. The scenario holds a way to make a digester and never a digester, because a
digester holds episode state and one `Scenario` is shared by every environment in the process.

**Where:** `src/syn_grid/core/droid/digestion/`

**Alternatives considered:**

- `core/digestion/`, the path the reference skeleton imports from. Rejected because digesting is something the droid
does.
- Digesters living in the scenario package, next to the scenario termination and spawning etc. Rejected because
digesters belong with the digester.
- `build_world` or the scenario creating the engine. Rejected because the engine is a part of the droid, and nothing
outside the droid needs to construct it. The droid's constructor takes digesters, so it never imports a concrete one
and never learns which scenario it is in.
- One method on the scenario that returns every digester. Rejected because adding an optional orb kind would then
mean editing that method as well as adding the kind's bundle.
- A `kind_of(orb)` function for the lookup. Rejected because the key is the kind alone, so it would wrap
`orb.META.KIND` and nothing else.
- `get` returning `None` when no digester of the asked class is registered. Rejected because every reader would have
to check for it and choose a number to report, and a metric that reads 0 because its digester is missing looks the
same as one that reads 0 because nothing was eaten. A reader and its digester belong together: a reader for an
optional orb kind is added where that kind's bundle is added.
- An unrouted orb paying 0, and a second digester for a kind replacing the first. Rejected for the same reason: both
can only come from a builder wired wrong, and both would go unnoticed.

**Why this one:** The engine stays a plain mechanism that any scenario family can use. The scenario decides which
digesters to build and how each is configured, through the bundle's `make_digester`, but it doesn't own their code.

**Status:** settled; who creates the engine was settled on 2026-10-11, and what the engine does on a miss on
2026-10-12. `digestion.py` and `engine.py` are written. The droid does not take digesters yet: it still builds the
engine from three config values, so `synergy_droid.py`, and the places that read the old engine's `chained_tiers`,
`stats` and `_pending_reward`, wait for phases 3 and 5. `NegDigester` is in `digesters/neg_digester.py`: it pays the
orb's own reward and counts `negatives_eaten`, so a metric can report it.

---

## Digesters do not talk to each other (2026-10-11)

**Decision:** A digester scores its own kind and knows nothing about the others. `digest` returns a float. There is
no `Event`, no `DigestionResult`, no tally in the engine, and no hook by which a digester is told that another kind's
orb was eaten.

A tier chain is broken only by a tier orb eaten out of order. A negative orb eaten in the middle of a chain costs its
own reward and leaves the chain alone.

A reader that needs what a digester knows asks the engine for the digester by its class and reads its attributes, as
in `engine.get(TierDigester).chains_broken`. The readers that do this are in the scenario family's own folder: its
termination, its metric readers, its HUD and observation readers.

A reader is handed the world and does not hold the engine. It asks the droid: `world.droid.get_digester(TierDigester)`,
a method that forwards to the engine's `get`. The engine stays the droid's own, which fits the droid creating it.
Settled on 2026-10-12, with the alternative of a public `digestion_engine` attribute turned down because every reader
would walk three levels into the droid. `TierChainTermination` is written against the method; the droid does not
have it yet.

**Where:** `core/droid/digestion/digestion.py`, `engine.py`, and the attributes of
`digesters/tier_chain/tier_digester.py`.

**Alternatives considered:**

- Events returned from `digest` and counted by the engine (the reference). Rejected because nothing reacted to an
event. Every reader asked for a count since the last reset, and a count since reset is state a digester already
keeps. Reading a digester's state has to exist anyway for values that are not counts, such as where the chain is, so
events were a second way out for the same information. Legacy kept the counts as plain ints.
- A `notice` hook on every digester, or in a second protocol that only some implement (the reference, and the
roadmap as first written). Rejected together with the rule that needed it. It makes every pair of digesters a
possible conversation, the result can depend on which digester the engine tells first, and a step's reward can no
longer be explained from one place.
- Any orb that is not a tier orb breaks a running chain (legacy and the reference). Rejected because a direct orb is
defined as changing the score and nothing else, and a negative orb that breaks a chain does something else. The orb
would also carry two costs, the second one depending on the chain and arriving later. With the chain untouched,
adding a negative orb to a scenario changes one thing, so its effect on a learning curve can be read.
- The builder handing the tier digester a set of kinds that break it. Not needed now. It is the shape to use if a
scenario wants an existing kind to break chains.

**Why this one:** Each digester can be read and tried alone, and each orb kind has one meaning: a tier orb is about
order, and a negative orb costs what its reward says.

**Status:** settled. It differs from legacy on purpose. Revisit when an orb has to affect another kind:

- An orb that changes what another kind's orb pays needs the engine to pass each reward through it. Neither events
nor `notice` did that.
- Events come back if something has to react to what happened and not only to how much it paid.
- An orb whose purpose is to break chains is a synergy kind of its own, added to a scenario on purpose.

---

## The tier digester is a template with one subclass per scoring mode (2026-10-09)

**Decision:** `TierDigester` is a base class that owns the chain. For each eaten tier orb it decides which of three
things happened, asks the subclass what that pays, counts it and moves the chain:

- **Progress:** the orb is the next tier, and not the last one. The chain moves up. A tier 1 with no chain running
  starts a chain.
- **Completion:** the orb is the next tier, and the last one. The chain is over.
- **Break:** the orb is any other tier. A tier 1 starts a new chain at 1, and any other tier leaves no chain.

A scoring mode is a subclass that writes three methods: `_score_progress`, `_score_break` and `_score_completion`.
Each is handed the orb and returns a reward, and each is called before the chain moves, so the mode sees the chain as
it was when the orb was eaten. The scenario's builder picks the subclass.

A mode writes a fourth method, `owed_at_episode_end`, which is public and takes nothing. It returns what the running
chain has gathered and digestion has not paid, and changes nothing. A termination reads it; the entry "Digestion pays
for orbs, and termination pays for the clock" says how it is used. Added on 2026-10-12.

The base holds what every mode shares and what readers outside take from it: `chained_tiers`, the counters
`chains_progressed`, `chains_broken` and `chains_completed`, and `chain_reward`, the summed reward of the orbs in the
running chain. It takes `max_tier` at construction, and each mode takes what it pays with.

Whether a `ScoringMode` enum exists as a router for the continuous sandbox is still open. If it does, its only job is
to pick which subclass to create.

**Where:** `core/droid/digestion/digesters/tier_chain/`: `tier_digester.py` for the base, and one file per mode.

**Alternatives considered:**

- One digester with a `ScoringMode` enum and a branching method per mode (the reference). Rejected because the builder
already decides the mode, so the enum states the same decision a second time, and it is the kind of conditional that
does not belong in the component holding it.
- The orb carrying its own scoring (the old engine). Rejected because the engine then has to inspect orbs to know how
to score them. Orbs are the same in every mode, and how the droid digests them is up to its own digestion.
- Reading `max_tier` off the orb, as legacy did. Not available: `TierOrb` no longer stores it.
- Properties for the values readers take. Not used: they are plain attributes without an underscore, which is the
convention in the rest of the code for something read from outside. A property blocks an accidental write to the
public name and nothing more. The owner has left open whether to return to properties.
- The last orb's reward held in the base beside the sum (legacy's `_pending_reward`). Not needed: no mode reads it.

**Why this one:** Adding a mode is adding a class, not editing one that works. A scenario only loads the mode it uses,
and each mode can be tried on its own.

**Status:** built on 2026-10-11 with all three modes; the next entry says what they pay. The base needed no way for a
subclass to replace the break as a whole: threshold's one exception fits inside its `_score_break`. The spatial
builder names `MaxTierDigester` since 2026-10-12, through a `TierDigesterRecipe`; see the next entry. No world has
been built with one, because `build_world` still raises.

---

## A tier bundle makes its digester from a frozen recipe (2026-10-12)

**Decision:** A tier orb's `OrbBundle` holds a `TierDigesterRecipe` as its `make_digester`. The recipe is a frozen
dataclass of three values: `scoring_mode`, which is the tier digester class to create, `max_tier` and
`chain_break_penalty`. Calling it returns a new digester.

A builder states its scoring mode by naming the class. The tier-chain family does it through `_tier_bundle`, which
takes the tier orb block and the class and returns the bundle, so each builder's choice is one argument.

The negative bundle needs no recipe. `NegDigester` takes no values, so the class itself is the `make_digester`. A
recipe is needed exactly when a digester's constructor takes a value, and it holds what that one kind's digester
takes. There is no base class for recipes and no generic one: what every `make_digester` has in common is the type
on the bundle, a callable with no arguments that returns an `OrbDigester`.

`scenario/blocks/recipes/` is for every recipe, not only a digester's. A recipe is a frozen object a scenario holds that
makes something with episode state when a world is built.

A scenario's `build_world` is the second one, `WorldRecipe`, added on 2026-10-12. It holds the world config, the
bundles and the field's size, and calling it builds a world. It replaced a `functools.partial` of
`_build_tier_chain_world`, which is gone. There is one `WorldRecipe` for every family and not one per family.
Building a world reads the grid, the droid's settings and the step limit, which are all on the base `WorldConf`, and
takes everything else from the bundles and the field's size. What a family adds to its world config,
`tier_orb_conf`, `neg_orb_conf` and `goal_conf` in a tier chain, is used up before that, by the bundles and by
termination. The old function was typed with `TierWorldConf` only because it sat in the tier-chain builders.

With it, two scenarios built from the same config compare equal, and a scenario's full world config can be read from
outside the builder, as `scenario.build_world.world_conf`.

**Where:** `scenario/blocks/recipes/tier_digester_recipe.py` and `world_recipe.py`; `_tier_bundle` and
`_tier_chain_scenario` in `scenario/families/goal/tier_chain/builders.py`.

**Alternatives considered:**

- A `functools.partial` of the digester class. Rejected because two partials never compare equal, so two scenarios
built from the same config would differ in their bundles. `build_world` had that problem until it became a recipe
too; see below.
- A module-level function per scoring mode, as `_make_tier_digester` was. Rejected because a function with no
arguments has nowhere to get `max_tier` and the penalty from.
- The recipe in the tier-chain builders. Rejected because it holds nothing of the family: every scoring mode takes
the same two values, and a sandbox that reads its mode from config would hand the same recipe the class it chose.
- The recipe in `core/`, beside the digesters. Rejected for the reason the populations are not there: only the
scenario layer needs a way to make a digester later. Core is handed digesters.
- The folder named `scenario/digestion/`, as it was for a few hours on 2026-10-12. Renamed to `scenario/recipes/`
because `core/droid/digestion/` is where digestion happens and two packages should not end in the same word, because
`populations/` beside it is named after what its files are, and because a recipe need not make a digester. Both
folders moved under `scenario/blocks/` later the same day.

**Why this one:** It compares by value, it is the one place a scoring mode is chosen, and the choice is a class and
not a name that has to be looked up. If a `ScoringMode` name ever appears in a config, its only job is to pick the
class that goes in this field.

**Status:** built on 2026-10-12 and used by the spatial builder. Checked by building the spatial scenario from its
YAML: the bundles of two builds compare equal, each call returns a new digester, and the tier orbs fed in order to
the digesters the bundles make pay what the three tier orbs are worth together.

`WorldRecipe` is written without its body: calling it raises `NotImplementedError`, because the droid and the world
do not take pieces until phase 3. Checked on 2026-10-12 with the spatial YAML: two builds of the same config compare
equal and are hashable, and a different grid, a different `chain_break_penalty` and an added negative block each
make a scenario that does not equal the first. That it needs nothing of a family is read from the config models and
from the steps the old function listed, not from a body that runs. Delay is the one to watch: its field freezes
after a consume, and where that mechanic lives is not decided.

Tied to it: a digester's `KIND` is set in `__init__` as `self.KIND: Final[OrbKind]`, not in the class body. A `Final`
with a value in a class body is a class variable to a type checker, the protocol's `KIND: Final[OrbKind]` is an
instance variable, and Pyright does not match the two. It showed when a digester class was first handed to a bundle.
Inheriting from the protocol was tried and turned down: Pyright refuses to assign a `Final` the parent declares, and
a subclass that forgets a method inherits the protocol's empty one.

---

## The scoring modes pay the same orbs at different times (2026-10-11)

**Decision:** A tier orb is worth the same in every scoring mode. A mode decides only when that worth is paid. A
break always costs `chain_break_penalty`.

| | Progress | Break | Completion | Owed at episode end |
|---|---|---|---|---|
| Step-wise | the orb's reward | the penalty, plus the orb's reward if it restarts a chain | the orb's reward | 0 |
| Threshold | 0 | the penalty, plus what the chain had gathered | what the chain had gathered, plus the orb's reward | what the chain has gathered |
| Max-tier | 0 | the penalty | what the chain had gathered, plus the orb's reward | 0 |

A chain completed without a break pays the same total in all three. Step-wise and threshold pay the same total for
any sequence of orbs, once threshold has paid out what it still holds.

The break is one rule in every mode and every scenario: a tier orb that is not the next one is a break. That covers a
wrong tier while a chain is running, a wrong tier with no chain running, and a tier 1 eaten while a chain is running,
a second tier 1 at chain 1 included.

A tier 1 that breaks a longer chain is the first orb of a new one. It counts toward the new chain's `chain_reward`,
and in step-wise it is paid its reward on top of the penalty, so a chain completed after a restart pays its full
worth in every mode.

Step-wise and threshold share one exception. A tier 1 eaten while the chain is at 1 pays the penalty alone. Otherwise
every tier 1 would be paid, and tier 1 orbs could be farmed wherever orbs respawn.

**Where:** `max_tier_digester.py`, `step_wise_digester.py` and `threshold_digester.py` in
`core/droid/digestion/digesters/tier_chain/`.

**Alternatives considered:**

- Max-tier pays a fixed number handed to the digester, with tier orbs worth nothing (the reference; legacy overwrote
the prize with a fixed 10). Rejected because sparse against dense at the same chain length is the comparison the
scaling scenarios exist for, and with the same worth in both, the only difference left is what a broken chain gives.
It also needs no extra config field, and it works unchanged in a scenario where completing a chain does not end the
episode. The cost is accepted: the prize is not written in the YAML, and it grows with the chain length unless the
base reward is set per scenario.
- Max-tier pays only the last orb's own reward (legacy, before termination overwrote it). Rejected because every
lower tier would have a reward that is never paid.
- A bonus at completion in step-wise: the sum of the earlier tiers paid a second time (legacy). Rejected because a
completed chain would be worth nearly double what it is in max-tier, so a comparison of the two modes would mix the
timing with the amount. It was added by feel and never verified.
- A separate `tier_consumption_penalty` for step-wise (legacy). Rejected because it was a second number for the same
idea. In every saved config it was tied to the step penalty.
- A tier 1 never costs a penalty (legacy step-wise paid a tier 1 its reward in any chain state). Rejected because it
can be farmed where orbs respawn, and because every other tier is right in exactly one chain state and wrong in the
rest.
- A threshold break pays only the last tier's reward (legacy). Rejected as a special case that needs its own held
value beside the sum.
- A threshold break with no penalty. Rejected because cashing out a chain with a tier 1 would then cost nothing, and
a break would cost the penalty in two modes and not in the third.
- A restarted chain's sum starts at 0, so that an orb eaten out of order never pays its own reward. It would close
threshold's farming loophole in the base with no exception. Rejected because the break is already paid for with the
penalty, and a completed chain should always pay the same.
- Step-wise paying the bare penalty for every break, the restarting tier 1 included. Rejected because a step-wise
chain would then total one tier-1 reward less than a threshold one after each restart.

**Why this one:** The modes form one family, so comparing two of them isolates when the reward arrives. The agent
gets one rule for order: eat the tier one above the chain.

**Status:** settled and built. What follows from it and is not done:

- The population gives the tier orbs their rewards since 2026-10-12, from `base_reward` and `growth_factor`.
- The rule assumes the agent sees where the chain is. Without that, a first and a second tier 1 look alike and pay
differently. Keep the chain position in the observation.
- In a goal scenario the rule assumes an eaten tier orb does not return in that episode. A tier 1 that breaks a chain
starts a new one and is paid for it, at once in step-wise and at the ending in threshold. Where orbs respawn that is
right, because the new chain can be completed. Where a break ends the episode it would pay for a chain that can never
grow. A tier chain has one orb per tier, so with no return there is no second tier 1 to break with and the case
cannot arise. Legacy behaved this way in single-chain mode, with and without delay. In the rebuild the orb is meant
to enforce it: a tier-chain tier orb has no cool-down, see "An orb carries its own cool-down and lifespan". Revisit
the restart rule if a goal scenario ever lets tier orbs respawn. Noted on 2026-10-12.

Checked on 2026-10-11 in a throwaway script: over every sequence of up to 7 orbs on a 4-tier chain, step-wise and
threshold paid the same total, counting what threshold still held at the end. Checked again on 2026-10-12 with
`owed_at_episode_end` as what threshold still held: no mismatch in the 21,844 sequences.

Not carried over and not discussed to a decision: legacy threshold's `reward_multiplier`, which scaled and rounded
each orb's reward. If it returns, it is a setting of the orb's reward and not of the mode.

Revisit if the prize's growth with the chain length makes long chains train worse. The first things to try are the
discount and how long the agent explores, then the base reward per scenario, then reward normalisation.

---

## A new scoring mode is a new scenario (2026-10-09)

**Decision:** Scoring mode is generally not a config setting when it comes to locked down scenarios (continuous sandbox
mode the exception). A different score-mode means a new registered scenario with its own builder, reusing the shared
tier-chain helper, and its own config model if the inputs differ. Scaling sparse and scaling dense already work this
way.

**Where:** `scenario/families/(goal|continuous)/*/builders.py`, `scenario/families/(goal|continuous)/*/config.py`, `scenario/registry.py`

**Alternatives considered:** The mode in the YAML, with a lookup from its name to a digester class. Rejected for now
(allowed for continuous sandbox scenario).

**Why this one:**

- The scenario name is the run's identity: run IDs, result folders and any comparison hang on it. A mode set in config
would let two runs of the same name pay differently.
- When this was written, a second reason was that the config models enforce each mode's inputs. That no longer holds
since 2026-10-11: all three modes take the same inputs, a reward per tier and `chain_break_penalty`. The first reason
carries the decision alone.

**Status:** settled. Revisit if a single experiment needs to sweep scoring modes. The lookup would then sit at the
edge, next to `SCENARIOS` or in the builders, and wouldn't touch the digesters.

---

## Two orb categories, and an orb meta that stores only kind and tier (2026-10-10)

**Decision:** There are two orb categories and there will not be a third.

- **Direct:** changes the score and nothing else, on the step it is eaten. Penalty and reward orbs. The only variety is
  how the number is decided: fixed, a random draw, or a value that depends on the orb's own age.
- **Synergy:** everything else. Eating it leaves state behind that changes a later outcome. A tier orb leaves a chain,
  a timed orb leaves a countdown, an item orb sits in the droid until the next orb, a world orb changes the field.

The test for a new orb: does its digester need anything but the orb itself? If it needs the droid, the world or a
memory of earlier orbs, it is synergy. So an orb bomb, growing the droid, a teleport, an instant game over and a
percentage of the current score are all synergy, even though they happen at once.

`OrbMeta` says what an orb is, not how it acts. It stores two things:

- `kind`, one member of `DirectKind` or `SynergyKind`. Every orb that behaves differently gets its own member
  (`NEGATIVE`, `POSITIVE`, ... and `TIER`, `FLIP_REWARD`, `OBSCURE`, `NULLIFY`, `HAZARD_BORDER`, ...).
- `tier`, an int for `SynergyKind.TIER` and `None` for everything else.

The category is not stored. It follows from which enum the kind comes from.

Inside synergy there are two groups. They are folders, not kinds, enums or categories:

- **Chain:** the payoff depends on which orbs of the same kind came before, and in what order. The tier orb is the
  only one so far; a streak or a collect-the-set orb would be others.
- **Effect:** eating it changes something other than itself: other orbs' rewards, the observation, the next orb, the
  world.

In digestion the routing key is the kind alone: `OrbKind = DirectKind | SynergyKind`, read from `orb.META.KIND`.
The engine and the digesters treat it as an opaque key and never take it apart.

**Where:** `core/orbs/orb_meta.py`, and what reads it: `core/grid_world.py`, `rendering/pygame_renderer.py`,
`gymnasium/observation_space/perceptions/`. The orbs are packaged as `core/orbs/direct/`, `core/orbs/synergy/chain/`
(holding `tier_orb.py`) and `core/orbs/synergy/effect/`. The routing key, `OrbKind`, is defined beside the
enums in `orb_meta.py` since 2026-10-12, so that the meta and the perceptions can name it without importing from
digestion.

**Alternatives considered:**

- A third `EffectKind` enum beside `SynergyKind`. Rejected because it reads as if effect orbs were not synergy orbs,
and nothing in the code needs to ask "is this an effect?".
- One `SynergyKind.EFFECT` member with a sub-field for which effect. Rejected because each effect needs its own
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
existed only to check that the two stored fields agreed.
- A separate meta class for tier orbs, or the tier on `TierOrb` itself, so that other orbs have no tier field at
all. Rejected because the renderer and the perceptions are handed metas of every kind and would have to check the
kind before touching the field anyway; `None` gives them the same check without a second class.

**Why this one:** Direct stays a small closed set and every interesting mechanic is synergy, which lines up with the
digester split: a direct digester is stateless, a synergy digester holds episode state. Effect orbs no longer need a
tier. A new orb is a new enum member and a digester, with no new field on the meta.

**Status:** settled, and built on 2026-10-11 except for the identity. Every identity is the same number as before, so
no observation changed.

- `OrbMeta` takes and stores `kind` and `tier`. An orb no longer states a category.
- `tier` is stored as `None` for every kind but `SynergyKind.TIER`, and the validator rejects a tier on any other
kind. `grid_world.py` decides which orbs despawn by checking the kind, and the composite grid perception writes 0
for a missing tier.
- The routing key is `OrbKind = DirectKind | SynergyKind`, in `core/orbs/orb_meta.py`.
- `SynergyKind.TIER` is a plain member with the value 1. It used to be forced to the highest value in the enum. The
identity does not need that: the tier has the highest slot in the radix formula, so a tier orb always sorts after
every orb without a tier, whatever its kind's value is. That holds as long as only tier orbs carry a tier.

The identity is left for its own pass, with the observation or just before it, and as the only change in that step
so the observation can be compared with legacy:

- The radix `IDENTITY` still has a category digit, which keeps a direct kind and a synergy kind with the same value
apart. The digit is worked out inside `compute_radix_identity` from the kind's enum, and `OrbCategory` exists only
for that.
- The radix number is only a sort key; the world replaces it with dense ids when it is built. It could become a
plain sort on kind and tier. Negative has to keep sorting before the tiers or the existing observations change.
- The world writes the dense id back onto the meta, so the meta can't be frozen until that id lives somewhere else.

- The word is kind everywhere since 2026-10-12: `DirectKind`, `SynergyKind`, `OrbMeta.KIND` and `OrbKind`. The meta
and its enums said type until then, while digestion and the config said kind, for one thing. Kind was kept because
`type` is a Python builtin and is also used for scenario types and Python types.

Open: whether synergy orbs other than tier orbs despawn and respawn like negative orbs or stay on the field. Revisit
the "two categories" rule only if an orb turns up that fits neither test.

---

## Scenario configs live with the scenario, as blocks in the world (2026-10-10)

**Decision:** A config model lives in the folder of the thing it configures, at the level where it is shared.

- `config/models/` keeps what is not specific to a scenario: the global and runner models, and a `scenario/` folder
  with what every scenario is built from. That is `common_models.py` with the grid, the droid, the
  observation handler and the base `WorldConf` and `ScenarioConf`, and `orb_models.py` with the base class of each
  orb kind.
- `scenario/families/goal/config.py` holds what every goal scenario has. `scenario/families/continuous/config.py` will do the same for
  continuous.
- `scenario/families/goal/tier_chain/config.py` holds the tier-chain family.

A scenario config has two halves. `world_conf` is everything the world acts on. `obs_conf` is what the Gymnasium side
needs to turn the world into an observation, which the world never reads. Each layer adds its own block to the world:

```
ScenarioConf                    world_conf, obs_conf
  WorldConf                       max_steps, grid_conf, droid_conf
    GoalWorldConf                 + goal_conf       (timeout_penalty)
      TierWorldConf               + tier_orb_conf   (max_tier, the reward ladder, chain_break_penalty),
                                    neg_orb_conf (optional)
        TierDelayWorldConf          tier_orb_conf   + delay
```

Each world model has a scenario model that narrows `world_conf` to it (`GoalScenarioConf`, `TierScenarioConf`,
`TierDelayScenarioConf`). A variant is three classes: its block, its world and its scenario model. Scaling dense has
no variant of its own since 2026-10-11 and uses `TierScenarioConf`; see "The reward ladder is one formula on the
shared tier orb block".

There is no single orb block to look at. Each orb kind has its own `*_orb_conf` block in the world, holding what that
kind's orbs and its digester need. This mirrors `OrbBundle` in the scenario layer, where a kind's orbs and its
digester travel together, so a kind is one brick in the config as well. The base `WorldConf` declares no orb block;
where a kind's class lives and which world model declares its field is in "An orb kind's config class is shared, and
each scenario declares the field".

`scenario/registry.py` holds one table, `SCENARIOS`, whose entry gives a name its config class and its builder.

**Where:** `config/models/scenario/common_models.py`, `config/models/scenario/orb_models.py`, `scenario/families/goal/config.py`,
`scenario/families/goal/tier_chain/config.py`,
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

**Where:** `DroidConf.max_energy` in `config/models/scenario/common_models.py`, replacing `starting_score`. The droid,
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

**Where:** `WorldConf.max_steps` in `config/models/scenario/common_models.py`. The count itself still has to move
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

## Digestion pays for orbs, and termination pays for the clock (2026-10-10, changed 2026-10-11 and 2026-10-12)

**Decision:** A reward caused by eating an orb is paid by digestion. Termination decides when an episode ends, pays
what the clock causes, and settles what a running chain is owed.

| Moment | Paid by | Pays |
|---|---|---|
| Chain break | the tier digester | `chain_break_penalty`, plus what the chain had gathered in threshold |
| Completion | the tier digester | what the scoring mode says the chain is worth |
| Any ending | termination | what the tier digester says the running chain is owed |
| Timeout | termination | `timeout_penalty`, on top of the row above |

Termination only adds to the step's reward and never replaces it. The step penalty, the border penalty and an eaten
orb's reward are paid before termination is asked, and they stay.

What a chain is owed is stated by the tier digester's `owed_at_episode_end`: what it has gathered in threshold, and 0
in max-tier and step-wise. Termination adds it on every step where it ends the episode, whatever the cause: energy
reaching 0, the clock, and in a goal scenario a break or a completion. It does not ask which mode is running. A tier
1 that breaks a chain starts a new one, so a threshold episode that ends on that break is paid the new chain's tier 1
by termination. With that, threshold and step-wise total the same over a whole episode.

The digester is the same in a goal scenario and a continuous one. What differs is the termination: which moments end
the episode, and whether the clock costs `timeout_penalty`.

It also differs in how the clock ends an episode. A goal scenario never truncates. Its deadline is part of the task,
so running out of steps is a failed attempt and sets `terminated`. A scenario with no objective, survive as long as
you can, has nothing to fail, and there the clock sets `truncated`. That is a continuous scenario by definition.
`EpisodeOutcome` is one type for every family, with `truncated` defaulting to false, so a goal strategy never names
it and a continuous one sets it. Settled on 2026-10-12.

There is no `completion_reward`. Termination reads `chains_completed` to end the episode and adds nothing for it.

`timeout_penalty` is in `goal_conf`, because any scenario with an objective and a deadline has it, whatever its orbs
are. `chain_break_penalty` is in `tier_orb_conf`.

**Where:** `GoalConf` in `scenario/families/goal/config.py`; `TierOrbConf` in `config/models/scenario/orb_models.py`; the
tier digesters in `core/droid/digestion/digesters/tier_chain/`; `TierChainTermination` in
`scenario/families/goal/tier_chain/termination.py`.

**Alternatives considered:**

- Termination adds a `completion_reward` to the step that completes the chain. This was the decision on 2026-10-10.
Rejected on 2026-10-11 because completing a chain is caused by eating an orb, because what it pays follows the scoring
mode (a static prize in legacy max-tier, the accumulated sum in threshold), and because a scenario where completion
does not end the episode has no termination to pay it.
- A scenario-level completion reward for a goal that is one orb being eaten. Rejected: that orb's own reward is the
prize, and a second number for the same event would have to be looked up in a second place.
- A rule per scoring mode inside one termination, as the old termination had. Rejected because it puts a
scoring-mode conditional in termination.
- A termination of its own for a threshold scenario, which pays `chain_reward`. Rejected because the digester's mode
and the termination would be chosen in two places that can disagree, and the sandbox, where the mode comes from
config, would need the conditional after all.
- `owed_at_episode_end` with a default of 0 on the base. Rejected because a new mode that holds a sum would then pay
nothing at a timeout without anyone being told. It is abstract, so every mode has to say.
- Nothing settled when energy reaches 0 (legacy). Rejected because a threshold chain would then be paid at every
ending but one. The cost is accepted: an agent that runs out of energy while it holds a chain is paid on the step it
fails.
- The timeout replacing the step's reward (legacy). Rejected because the last step's own penalties and any orb eaten
on it were lost.
- Threshold paying no `timeout_penalty` (legacy). Rejected because the clock running out is the same event in every
mode.
- `EpisodeOutcome` without a `truncated` field, the environment returning false for it. Considered while the docs
said `truncated` is always false. Rejected because a continuous scenario needs it.
- An outcome type per family group, without `truncated` for goal scenarios. Rejected because the environment hands
Gymnasium both flags on every step, and would have to ask which type it was given.
- Moving `timeout_penalty` into a max-tier scoring block, since dense never used it. Rejected because a timeout
penalty is a goal-level idea: a goal scenario with other orbs would use it too.
- Optional fields, `float | None = None`. Rejected because `None` either means the same as 0 or hides a second rule
behind a value.
- A default of 0 on the shared fields. Rejected because a `timeout_penalty` forgotten in a YAML would then load and
train with none.
- Defaults of 0 on classes of dense's own, so its YAML can leave the values out. This was in place until
2026-10-11. Removed because the defaults existed to reproduce the thesis runs, and with them gone dense needs no
config classes at all. Every scenario writes `timeout_penalty` and `chain_break_penalty` in its YAML.

**Why this one:** The digester does the same thing in a goal scenario and a continuous one, and the only difference
between the two is whether termination ends the episode. What reaching the goal pays is found in one place.

**Status:** settled, and applied to the config on 2026-10-11: `completion_reward` is gone from `GoalConf`, the
tier-chain termination, the builder and the spatial YAML. `TierChainTermination.evaluate` is written since
2026-10-12.

When two endings fall on one step, the clock is charged only if it alone ended the episode. A chain completed or
broken on the last step, or energy running out on it, pays no `timeout_penalty`. What the chain is owed is added
either way.

`evaluate` reads three things: `world.droid.energy`, the tier digester through `world.droid.get_digester(TierDigester)`,
and the steps left, which it is still handed as an argument. Neither `energy` nor `get_digester` exists on the droid
until phase 3, so the file imports and runs against a stand-in world, and a type checker reports both as unknown.
The argument goes when the world counts the steps.

Checked on 2026-10-12 with a stand-in world around real digesters, in all three scoring modes: each of the four
endings, a step that ends nothing, and the three cases where the clock runs out on a step that something else ended.
No test covers it.

- Settled on 2026-10-12: what a running chain is paid when an episode ends. On 2026-10-10 every goal scenario paid
"what the chain was holding" by one rule, which worked because max-tier never held anything. The base digester keeps
`chain_reward` in every mode since 2026-10-11, so a max-tier scenario must not pay it while a threshold scenario
does. Each mode now says what its chain is owed, and termination pays it. `owed_at_episode_end` is written in the
base and the three modes, and `TierChainTermination.evaluate` adds it at every ending.

A scenario-level completion reward earns a place only when the goal is something no single digester can see, such as
surviving a number of steps or clearing the field. Revisit when such a scenario exists.

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

Every population says how many orbs it makes, through `count` on the `OrbPopulation` protocol. The member knows no
scenario; what is done with it depends on how the field is filled. Added on 2026-10-12.

| | Fixed field | Sampled field |
|---|---|---|
| Orbs in the pool | the sum of the counts | the sum of the counts |
| Orbs on the field at once | the same number | `max_active_orbs` from the config, fewer than the pool |
| Observation slots and sort limit | follow the field | follow the field |

In a fixed field the family's helper sums the counts of its bundles, as `_tier_chain_scenario` does for `orb_count`,
and nothing adds one by hand when a kind joins. In a sampled field a kind's share of the pool depends on every other
kind's weight, so one population cannot work out its own count. The family's builder is expected to apportion the
pool and hand each population its number.

**Where:** `NegOrbConf` in `config/models/scenario/orb_models.py`. The weighted version is not written yet; it goes
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
- The helper keeping its own tally: `max_tier`, plus one inside the negative block. This was in place for a few hours
on 2026-10-12. Rejected because a forgotten `+= 1` leaves the observation one slot short without an error, and the
sort limit then drops the farthest orb.
- Counting the orbs a population creates, with `len(population.create())`. Rejected because it makes orbs only to
count them.

**Why this one:** Each block holds only what its family reads. A weight is only meaningful relative to the other
weights in the same pool, so it appears exactly where there is a pool.

**Status:** settled for the tier chain. Revisit when the first sampled family is built, and for delay, where the
negative orb is meant to freeze and return with the field. Eating the negative orb does not affect a running chain;
see "Digesters do not talk to each other".

---

## An orb carries its own cool-down and lifespan (2026-10-12)

**Decision:** Whether an orb returns after it leaves the field, and whether it expires while it is on it, are values
of the orb, set by the population that makes it.

- **Cool-down:** a number of steps, or `None` for an orb that does not return in the episode.
- **Lifespan:** a number of steps, or `None` for an orb that does not expire.

The world treats every orb alike. It ticks a lifespan where there is one, and it respawns from the inactive orbs that
say they are ready to return. An orb with no cool-down is never ready, so it stays inactive for the rest of the
episode. It is not thrown away: the next episode's reset places every orb again whatever its cool-down, and the
observation still has a slot for it.

A population takes these as values and knows no scenario. The tier-chain family's builders hand
`TierChainPopulation` a cool-down of `None`, stated once in `_tier_bundle`. The negative population is handed a
number. A family whose tier orbs come and go hands the same `TierChainPopulation` numbers.

The populations are in `scenario/blocks/populations/`, shared by every family, and a family's folder holds none. A
population is the counterpart of `make_digester`: an `OrbBundle` holds both, and `build_world` runs each once per
world. What they make, an orb and a digester, is in core.

**Where:** `core/orbs/base_orb.py`; `scenario/blocks/populations/tier_chain_population.py` and `neg_population.py`;
`_tier_bundle` in `scenario/families/goal/tier_chain/builders.py`.

**Alternatives considered:**

- A cool-down of 0 on tier orbs, with spawning deciding per kind which orbs return. Rejected because a tier chain
holds tier orbs that never return beside a negative orb that does, so the world or the spawning rule would have to
ask each orb its kind. That check existed twice already: the world ticked a lifespan only for kinds other than
`TIER`, and `SpawningRules`, which has since been removed, had `tier_orb_expires`.
- Dropping an eaten orb that has no cool-down from the world. Rejected because the reset and the observation both
need it.
- Both populations in the tier-chain folder. Rejected because one negative orb in a fixed field is not a tier-chain
idea.
- The tier population in the tier-chain folder with "no cool-down" written into the class, and only the negative one
shared. This was built first on 2026-10-12. Rejected the same day: with the cool-down as a field the class has
nothing of the family left in it, just as the tier digester is handed `max_tier` and the penalty.
- The populations in `core/orbs/`, where `orb_factory.py` was. Rejected because the world no longer makes its orbs.
It is handed them, and the only caller of a population is `build_world`, in the scenario layer. The world owns
putting orbs on the field, which is spawning.

**Why this one:** An orb kind stays one brick. Adding the negative orb to a field adds its bundle and changes
nothing in the world or in spawning. It also puts the rule that an eaten tier orb does not return in the orb, where
no spawning rule can forget it, and it makes the lifespan a value per orb where it is a class attribute shared by
every environment in the process.

**Status:** settled on 2026-10-12 and partly built. Both populations are written, the spatial builder uses
`_tier_bundle`, `_negative_bundle` is written, and `TierOrb` and `BaseOrb` accept a cool-down of `None`. Nothing
acts on it: `BaseOrb` hands the cool-down to its timer, which needs a number, has no way to say it is ready to
return, and takes no lifespan. `NegativeOrb` is built without a lifespan. `tier_orb_expires` left the scenario's
rules on 2026-10-13, when `SpawningRules` was removed. The lifespan follows from the grid size, so a
builder has to hand it to the population.

---

## The scenario package is blocks, families, a registry and a scenario (2026-10-12)

**Decision:** `scenario/` has two folders and two files at its top level.

- `blocks/` is everything a scenario is made of that no family owns: the interfaces and rule classes as files, and
  the shared implementations in `populations/` and `recipes/`. The population protocol is in `populations/` with the
  classes that implement it.
- `families/` is the scenarios built from the blocks. A group such as `goal/` holds what its families share, and a
  family folder such as `tier_chain/` holds its config models, its builders and the strategies that are its own.
- `registry.py` selects a scenario by name, and `scenario.py` is the `Scenario`.

**Where:** `src/syn_grid/scenario/`.

**Alternatives considered:**

- `populations/` and `recipes/` at the top level beside `blocks/` and `goal/`, as they were earlier the same day.
Rejected because they are pieces a scenario is made of, the same sort of thing as `blocks/`, and they sat level with
`goal/`, which is a different sort.
- `goal/` at the top level, with `continuous/` to join it later. Rejected because the top level then mixes the pieces
with the scenarios built from them.
- The population protocol left as `blocks/population.py` beside a `blocks/populations/` folder. Rejected: two
near-identical names in one folder.

**Why this one:** The top level reads as the three roles the package has: what scenarios are made of, the scenarios,
and the table that selects one.

**Status:** done on 2026-10-12. `families/continuous/` exists since 2026-10-13 as an empty package.

A strategy that only one family uses stays with that family, as `TierChainTermination` does. The termination
interface is a file in `blocks/` with no folder of implementations, because none is shared yet.

---

## The world runs the orb field, and a scenario states only its size (2026-10-13)

**Decision:** Filling the field, taking an orb off it and bringing one back are the world's own behaviour, the same
in every scenario:

- At the start of an episode the world places orbs until the field holds `max_active_orbs`.
- An orb leaves the field when it is eaten or when its lifespan runs out.
- After a step, if the field has room and an inactive orb says it is ready to return, the world brings one back.

A scenario states one thing about the field: `max_active_orbs`, a plain value on its `WorldRecipe`. There is no class
for a scenario's orb-field rules and no hook.

The class that was there, `SpawningRules`, came from the reference with three fields and three hooks. What left it,
and why:

| Removed | Why |
|---|---|
| `tier_orb_expires` | an orb's lifespan says whether it expires |
| `spawn_max_on_episode_start` | every scenario fills the field |
| `on_reset` | filling the field is the world's, with nothing to choose |
| `after_step`, and the strategy objects behind it | "leave the field alone" and "refill" are one behaviour once an orb says whether it is ready to return |
| `on_orb_consumed` | only delay would use it, and delay is not written |
| the class | one number does not need one |

**Where:** `WorldRecipe.max_active_orbs` in `scenario/blocks/recipes/world_recipe.py`; `_tier_chain_scenario` in
`scenario/families/goal/tier_chain/builders.py`, which passes the sum of its bundles' counts.

**Alternatives considered:**

- The three hooks kept, as the reference had them, with a strategy object for the end of a step. Rejected because
only one of the three moments differs between scenarios once the orb carries its own cool-down and lifespan.
- The class kept with one hook, `on_orb_consumed(world, orb)`, that does nothing, for delay to override. It was
renamed `OrbFieldRules` and written this way for some hours on 2026-10-13. Rejected because the method existed only
for a subclass that is not written, its docstring could only describe doing nothing, and it fixed how delay plugs in
before delay was looked at. The class had been shrunk from the reference's, where the question should have been
what would be built from nothing.
- A delay length on the scenario's rules, `None` for every scenario but one. Not chosen now; it is one of the homes
listed for delay in `open-questions.md`.
- An orb that tells the world when its timer has run out, in place of the world asking each step. Rejected. A timer
here is a counter that goes down once a step, so every orb is visited every step either way and only the check is
saved. Each orb would need a reference into its own world, and training holds 16 worlds in one process. The order in
which two orbs return decides which random cells are drawn, and a loop over a list shows that order in one place.
The world is turn-based and holds a handful of orbs; a schedule of due steps would pay off with hundreds of them.

**Why this one:** Nothing is written for a scenario that does not exist. What delay needs is decided when delay is
written, with what it is for in view.

There are two kinds of thing that can happen when an orb is eaten, and neither is the field's size:

- The effect belongs to the orb. An orb that changes what other orbs pay, or what the droid can do, or what the grid
  costs, does so in every scenario that holds it. That is the kind's own digester, with what the engine offers it.
  No such orb is written, and what the engine would need for one is listed in the roadmap as not on it.
- The effect belongs to the scenario. A broken chain ending the episode is one, and it is in termination. Delay's
  field going empty is the other, and it has no home yet.

**Status:** done on 2026-10-13. `blocks/spawning.py` is deleted. The world does none of the three things above in the
new way: `grid_world.py` is still the old code.

---

## A scenario mechanic lives in its strategy when the world's public API is enough (2026-10-11)

**Decision:** A mechanic that only one scenario uses is written in that scenario's strategy, as long as it can be
written against the world's public API and the orbs' own methods. It becomes a world method only when it has to touch
the world's private bookkeeping to stay consistent.

Applied to the orb field: the delay scenario's cooldown and reactivation are loops over the active orbs that call the
orbs' own methods, so they belong beside the delay scenario. Spawning stays in the world, because it uses the world's
rng, checks for a free cell and moves orbs between the active and inactive lists.

**Where:** the strategy files beside each scenario, and
`src/syn_grid/core/grid_world.py`.

**Alternatives considered:**

- A world method per mechanic, such as `deactivate_all_orbs` and `reactivate_all_orbs`, which the strategy only
invokes. Rejected because the number of scenarios is unknown, and the world would collect methods that are irrelevant
to most of them.
- The strategy returns a recipe and the world carries it out. Rejected because the world would then have to understand
every kind of recipe, which is a scenario-specific case inside the world.

**Why this one:** The world stays the same size however many scenarios are added, and a scenario's mechanic is read in
one place, beside that scenario.

**Status:** settled as a rule. Revisit if scenarios start needing the world's public API widened one mechanic at a
time, since that is the same growth by another route.

---

## An orb kind's config class is shared, and each scenario declares the field (2026-10-11)

**Decision:** An orb kind's config is split over three places.

- `config/models/scenario/orb_models.py` holds the class that describes the orb kind itself, whichever scenario it appears
  in. It is a file of its own because it is the part of the common models that grows with every new kind.
- A family holds what describes its use of that kind, as a subclass that adds fields.
- A scenario's world model declares the field. That is where a kind becomes part of a scenario, and where it is
  optional or mandatory.

The base `WorldConf` declares no orb fields, so a scenario's config accepts only the orbs that scenario declares. No
orb kind is treated differently below the scenario. The negative orb is optional in a tier chain because the
tier-chain world model says so, and a scenario that depends on it declares it without a default.

A base class holds only the fields that are certain to be shared. For the negative orb that is `reward` and
`cool_down`. For the tier orb it is `max_tier`, the reward ladder and `chain_break_penalty`; the ladder joined
the base on 2026-10-11, when every scoring mode came to pay what the orbs are worth. Scoring mode is not a field of
the base: a scenario that lets it
vary declares it on its own block.

**Where:** `config/models/scenario/orb_models.py` for the base classes, each family's `config.py` for its subclasses and its
world model.

**Alternatives considered:**

- `neg_orb_conf` as an optional field on the base `WorldConf`, so any scenario may add a negative orb. This is what
was there. Rejected because every scenario's config then accepts the block, including one whose builder never wires
it, and because a scenario that depends on the orb can't make it mandatory.
- Each scenario defines its own class for a kind it uses. Rejected because two classes describing the same orb can
drift apart.
- A class stays in the family that first uses it and moves to common when a second family needs it. Rejected because
more scenarios using the tier orb and the negative orb are already expected.

**Why this one:** What an orb is gets written once, and each scenario states its own relationship to it in one line.

**Status:** settled and applied to the models. `TierOrbConf` is in `orb_models.py` with `max_tier`, `base_reward`,
`growth_factor` and `chain_break_penalty`, and `TierWorldConf` declares `neg_orb_conf`. Moving a field between a base class and its subclass doesn't change any
YAML, so a wrong guess about the base is cheap to correct.

---

## The reward ladder is one formula on the shared tier orb block (2026-10-11)

**Decision:** A tier orb is worth `base_reward * tier ** growth_factor`. A growth factor of 1 gives rewards that grow
linearly with the tier, and a larger one makes higher tiers worth more than that. Both settings are fields of the
shared `TierOrbConf`, so every tier-chain scenario states what its orbs are worth. The reward is not rounded.

Scaling dense has no config classes of its own. It is registered with `TierScenarioConf`, like spatial and scaling
sparse. Delay keeps its own, because it adds a real field.

Spatial is set to a base of 1 and a growth factor of 1.5. Its three tiers are worth 1, 2.83 and 5.20, about 9.0
together, which is close to the fixed 10 that legacy paid for a completed chain.

**Where:** `TierOrbConf` in `config/models/scenario/orb_models.py`; `TierOrb` in
`core/orbs/synergy/chain/tier_orb.py`; `scenario/families/goal/tier_chain/config.py`; `scenario/registry.py`;
`config/yaml/goal_tier_chain_spatial.yaml`.

**Alternatives considered:**

- A `linear_reward_growth` switch beside the growth factor (legacy). Rejected because the growth factor was then
required and ignored whenever the switch was on.
- Rounding each reward to a whole number (legacy, in the non-linear branch only). Rejected because nothing needs
whole numbers, and the two branches then disagreed for a base that is not a whole number.
- The ladder on dense's own orb block, with tier orbs worth nothing elsewhere. This was in place until 2026-10-11.
Rejected once max-tier came to pay what the orbs are worth.
- Keeping dense's own classes for their two defaults of 0. Rejected; see the termination entry above.
- A base of 2 with linear growth for spatial, which gives a chain worth 12. Not chosen: 9.0 is closer to 10, and a
growth factor of 1.5 is what the saved runs used.

**Why this one:** One formula with no setting that is ignored. In `reproduction_package/` every saved config has
`linear_reward_growth: false` with a growth factor of 1.5 or 1.685, so the capability was in use, though it only
reached the agent in the three dense runs.

**Status:** settled and applied. `TierOrb` no longer takes `linear_reward_growth`. When scenarios are locked
(`codebase-ideas.md`), a scenario's fixed values go in its builder, so locking is not a reason for dense to get
config classes of its own again.

---

## Legacy is a reference, not a target (2026-10-11)

**Decision:** The legacy strain in `src/syn_grid/legacy/` and the runs in `reproduction_package/` are references.
The rebuild does not have to reproduce legacy's rewards step for step. They are used to check that a rebuilt scenario
still learns, by comparing learning curves, and to look up what a behaviour used to be.

**Where:** phases 0, 6 and 8 of `scenario-rebuild-roadmap.md`.

**Alternatives considered:** An exact per-step match of rewards as the bar for the rebuild, which is how the roadmap
was first written. Rejected because several decisions change rewards on purpose:

- A max-tier chain pays what its orbs are worth, where legacy paid a fixed 10.
- A negative orb no longer breaks a chain.
- Step-wise has no completion bonus, uses `chain_break_penalty`, and charges it for a tier 1 eaten out of order.
- A threshold break pays the penalty plus everything the chain gathered, where legacy paid the last tier's reward.
- An ending adds to the step's reward, where legacy's timeout replaced it. A threshold timeout pays `timeout_penalty`
too, and a threshold chain is paid what it is owed when energy reaches 0.

**Why this one:** Holding the rebuild to legacy's numbers would mean keeping behaviour that was decided against.

**Status:** settled. Nothing decided so far changes the observation, so an observation trace can still be compared
exactly. If a rebuilt sparse scenario does not learn, legacy is the place to check whether it ever did.

---

## Reward normalisation stays off (2026-10-11)

**Decision:** Training keeps `norm_reward=False` on `VecNormalize`. Observations are normalised and rewards reach the
algorithm as they are.

**Where:** `runners/agent_runners/sb3/artifact_manager.py`.

**Alternatives considered:** Turning reward normalisation on, so that the size of a chain's prize stops mattering to
the learner. Not now:

- The rewards are designed by hand to sit in a small range.
- Normalisation divides by a running estimate of how much the return varies. With sparse rewards that estimate is
tiny until the first success and shifts when it comes.
- The 81 saved runs in `reproduction_package/` were trained without it, so new runs would stop being comparable with
them for a reason unrelated to the rebuild.

**Why this one:** It has been this way since `VecNormalize` was introduced on 2026-04-29 (`593fa89`). No commit on
any branch sets it to true, and 8 of the 81 saved stats files were read and all store false. The reason was not
written down; the owner recalls it as a deliberate choice.

**Status:** settled for now, and not tested either way on these scenarios. Revisit for continuous scenarios, where
the return is many completed chains added together, or if a chain's total goes well beyond a few tens.

---

## A locked scenario loads a small model, and its builder makes the full config (2026-10-12)

**Decision:** The layered models stay as the full description of a world: `TierScenarioConf`, `TierWorldConf`,
`TierOrbConf`, `GoalConf` and the rest, with their validation. A locked scenario gets one small model of its own,
frozen, strict and `extra="forbid"`, holding only what a run may choose for it. Its entry in `SCENARIOS` names that
small model as its config class, so its YAML loads into it and a YAML that states a fixed value is rejected. Its
builder takes the small model, makes the full config in code from the scenario's fixed values and what the run chose,
and hands that to the family's shared helper. Nothing below the builder knows whether a scenario is locked.

A scenario that is not locked, and the continuous sandbox, have no small model. Their entry names the full model and
their YAML states every value.

This is the third of the three sketches in `codebase-ideas.md`, with the registry and the sandbox filled in.

**Where:** not applied. It will touch `scenario/registry.py`, each family's `config.py` and `builders.py`, the
scenario's YAML, and `load_experiment_configs` in `app.py`.

**Alternatives considered:**

- Split each block into a narrow class and a tunable class that extends it (the first sketch). Rejected because no
single narrow class suits every scenario: spatial keeps the grid, scaling the grid and the chain length, delay the
delay.
- Thin the base models to what every scenario states and move the tunable fields to subclasses (the second sketch).
Rejected because `_tier_chain_scenario` and the `WorldRecipe` read a full config, and would have to take
their values from two sources.
- Fixed values as loose constants passed around the builder, with no full model made. Rejected because the values
then skip validation and everything below the builder takes a pile of arguments.
- One full model for every scenario, with a GUI choosing which fields to show. Rejected because the GUI would still
need, per scenario, the list of open fields and the fixed values for the hidden ones, which is the small model and
the builder's constants kept in the wrong layer. The YAML path could also still set anything.
- Fixed values as defaults on the models. Turned down on 2026-10-11; see `codebase-ideas.md`.

**Why this one:** The models are what the builder makes, so a locked scenario's fixed values pass the same validation
as a YAML's. The helper keeps taking one `TierScenarioConf`. The entry's config class says what a run may set for
that scenario, which is also what a GUI needs to draw a form; `model_json_schema()` on the models carries the types,
the bounds, the perception's choices and the optional orb block. Frozen models compare equal, so on the day a scenario
is locked the builder's config can be checked against the one its old YAML loaded.

**Status:** settled as the direction on 2026-10-12, and not applied. A scenario is locked after it has been rebuilt
and has reproduced its legacy results, so no model changes before then. Open, to settle with the first scenario to be
locked:

- The common type of a config class. `ScenarioConf` is named by the `TypeVar` bound in `registry.py`,
  `ScenarioEntry.build`, `build_scenario` and the return type of `load_experiment_configs`, and a small model is not a
  `ScenarioConf`.
- How the full config is reached from outside the builder. Since 2026-10-12 the world part of it is on the scenario,
  as `scenario.build_world.world_conf`; the observation part is not, and nothing reads either yet.
  `ConfigManager.save_snapshot` copies the YAML file, so a locked scenario's snapshot would hold only what the run
  chose; a GUI showing the fixed values read-only needs the same thing.
- Where a family's shared fixed values are written, so that four builders do not each state them.
- Where a rule that spans fields is checked. A check on the full model is raised in the builder and names a block and
  not a field: a positive `step_penalty` is reported on `droid_conf`, and a chain too long for the grid on
  `world_conf`.
- Whether a locked axis has a range, such as an upper bound on the grid.
- What the small model holds besides the axis. It holds the perception, limited to the ones that suit the scenario;
  see the next entry. The negative orb would be a yes or no. "Axis model" is then the wrong name for it.

---

## A locked scenario offers the perceptions that make sense for it (2026-10-12)

**Decision:** The perception stays a choice in a locked scenario, among the perceptions that fit what the scenario
tests. Spatial tests memory under a limited view, so it offers only perceptions where the droid sees part of the
world, and no fully Markovian one. A scenario whose test does not depend on how much the droid sees can offer any.

**Where:** not applied. The `Perception` enum in `config/models/scenario/common_models.py` has two values, and the
only one spatial allows is `vector_fog_of_war`. `PERCEPTIONS` in `gymnasium/observation_space/observation_handler.py`
names seven classes; the five outside the enum come back in the observation pass.

**Alternatives considered:**

- The perception fixed in the builder, as one of a locked scenario's read-only values. Rejected because how the
world is encoded is a real choice for a run. A composite observation with its own feature extractor can give the
model a clearer picture than the flat vector, at a cost in speed that matters on weaker hardware, and a user who
brings a model of their own chooses what it reads.
- Any perception for any scenario, since the run id starts with the perception. Rejected because a spatial run where
the droid sees the whole grid does not test what the scenario is named for.

**Why this one:** What a scenario tests decides how much the droid may see, and nothing else about the observation.

**Status:** settled as the rule on 2026-10-12. Open:

- How the rule is written. Either each locked scenario lists the perceptions it allows, or a perception is split into
  how much the droid sees and how that is encoded, and the scenario fixes the first; see `codebase-ideas.md`. There
  is nothing to choose between until spatial has a second allowed perception.
- The pixel observation. `GridPixel` is a docstring with no code, and whether it gets a limited view is not decided.

---

## The algorithm in the runner config is an enum (2026-10-13)

**Decision:** `alg` in `runner_config.yaml` is read as an `Algorithm` enum with one member per registered runner:
`PPO`, `FSPPO` and `RPPO`. The `RUNNER` table is keyed by it. A name that is not a member fails when the config is
loaded, and pydantic's message lists the allowed ones. The run id is built from the member's `.value`.

**Where:** `Algorithm` and `CommonConf.alg` in `config/models/runner_models.py`. The field is
`Annotated[Algorithm, Field(strict=False)]`, as `GlobalConf.scenario` is, because the models are strict and a strict
enum field rejects the plain string YAML gives. `RUNNER` and `build_runner` are in
`runners/agent_runners/runner_registry.py`, and the run id in `base_agent_runner.py`.

**Alternatives considered:**

- `alg: str`, checked against the keys of `RUNNER` in `build_runner`, which is what it was. Rejected because a typo
  was found only after the scenario had been built, and because the valid names were stated only as the keys of a
  table in another package.
- The enum's member put straight into the run id's f-string. Rejected because a `str` enum formats as its value on
  Python 3.10 and as `Algorithm.RPPO` from 3.11, and checkpoints are found by that id.

**Why this one:** It is the pattern `ScenarioName` and `SCENARIOS` already follow, so the two registries read the same
way.

**Status:** settled. The check in `build_runner` is kept. It can no longer be reached by a config and fires only when
a member has no entry in `RUNNER`, which is what the same check in `build_scenario` is for.

---

## The policy's name comes from the algorithm (2026-10-13)

**Decision:** `resolve_policy(perception, algorithm)` returns the one name in the algorithm's own `policy_aliases`
that starts with the perception's base policy. `PPO` offers `MlpPolicy`, `CnnPolicy` and `MultiInputPolicy`, and
`RecurrentPPO` offers the same three with `Lstm` in them, so naming the algorithm also names the policy family. The
base SB3 runner calls the resolver once and adds the result to the hyperparameters. A concrete runner says nothing
about the policy.

The table from a perception's leading word to a base policy stays: `vector` to `Mlp`, `composite` to `MultiInput` and
`grid` to `Cnn`.

**Where:** `runners/agent_runners/sb3/policy_resolver.py`, called from `BaseSB3Runner.__init__` in
`base_sb3_runner.py`. The resolver raises a `ValueError` that names the algorithm and what it offers when there is
not exactly one name.

**Alternatives considered:**

- A `use_lstm` flag on the resolver, which is what it was. Rejected because a shared resolver then knows one
  algorithm's variant, and the next variant needs another flag.
- The suffix handed in by the caller, as a class attribute each runner overrides. Rejected because the default in the
  base class read as "every algorithm uses `Policy`", and the override in `LstmPPO` was easy to miss.
- A variant handed to the base runner's `__init__`, `"Lstm"` or nothing, joined to a `Policy` constant. It was
  written and then replaced: the runner was still describing something the algorithm class already states.
- The resolver called in each concrete runner. Rejected because the policy depends on the perception, which is known
  only with the scenario, so every runner would repeat the same call.
- Perceptions named after SB3's policies, `mlp_fog_of_war` and so on, which would remove the table. Rejected because
  a perception's name says what the observation looks like, and SB3's wording would then reach the config and the
  Gymnasium side.

**Why this one:** It removes the LSTM special case from this code base, and a name SB3 changes is reported in one
place with the list of what the algorithm does offer. A wrong name was caught before too, by SB3 when the model is
created.

**Status:** settled. Revisit if:

- An algorithm offers two policies whose names start with the same base policy. The resolver raises on that and does
  not choose.
- A perception's name stops starting with its kind. The kind is read by splitting the name at the first underscore.

The `composite` and `grid` rows of the table are not reachable while the `Perception` enum has only vector members.

---

## Human play is not an agent runner (2026-10-13)

**Decision:** `HumanRunner` is a plain class. It takes a `Scenario` and has one method, `play()`. With
`human_control` set, `app.main` builds the scenario, runs `HumanRunner(scenario).play()` and returns, before a
`RunnerBundle` or an agent runner is built. `build_runner` takes only the bundle, and the registry maps an algorithm
to a class and nothing else.

Human control overrides everything else in the run. The snapshot and the runner config are ignored, and no check
rejects the combination.

**Where:** `runners/human_runner/human_runner.py`, `main` in `app.py`, and `build_runner` in
`runners/agent_runners/runner_registry.py`. The override is stated in the header of `config/yaml/global_config.yaml`.

**Alternatives considered:**

- `HumanRunner` as a `BaseAgentRunner` returned by `build_runner`, with `eval()` as the play loop and a `train()`
  that raises, which is what it was. Rejected because it inherited what it could not use. It skipped the base
  `__init__` to avoid making output folders, so it had no run id and the snapshot path raised on it, and whether it
  ran at all depended on `training` in the runner config.
- `GlobalConf` rejecting `human_control` together with an enabled snapshot. Rejected because a human run is for
  trying something out, and a config that has to be edited before it will start works against that.

**Why this one:** A human playing or an agent running is a choice about the whole run, so it is made once, in
`main`. It is the shape the legacy app has.

**Status:** settled. `play()` and `main` have not been run, since both need the environment.
