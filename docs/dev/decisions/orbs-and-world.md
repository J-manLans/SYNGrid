# Decisions: Orbs and the world

What an orb is, how the world runs the field, the step clock, and how a scenario acts on the world.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Two orb categories, and an orb meta that stores only kind and tier (2026-10-12)

**Decision:** There are two orb categories and there will not be a third.

- **Direct:** changes the score and nothing else, on the step it is eaten. The number may be fixed, drawn, or depend
  on the orb's own age.
- **Synergy:** everything else. It leaves state behind that changes a later outcome.

The test: does its digester need anything but the orb itself? If it needs the droid, the world or a memory of earlier
orbs, it is synergy, even if it happens at once (an orb bomb, a teleport).

Inside synergy are two groups, as folders only: **chain** (the payoff depends on which orbs of the kind came before;
the tier orb) and **effect** (it changes something other than itself).

`OrbMeta` says what an orb is: `kind`, a member of `DirectKind` or `SynergyKind`, one per orb that behaves
differently; and `tier`, an int for `SynergyKind.TIER` and `None` for every other kind. The category follows from the
enum. The routing key is `OrbKind = DirectKind | SynergyKind`, defined in `orb_meta.py`. The word is kind
everywhere, since `type` is a builtin.

**Why:** Direct stays a small closed set, which lines up with the digesters: a direct digester holds nothing a later
orb's payout depends on (its count of orbs eaten is for the metrics), while a synergy digester holds episode state that
changes what a later orb pays. A new orb is an enum member and a digester, with no new field.

**Rejected:**

- An `EffectKind` enum beside `SynergyKind`: it reads as if effect orbs were not synergy.
- One `SynergyKind.EFFECT` with a sub-field: each effect needs its own digester, so the key must tell them apart.
- Storing how an effect acts in the meta: those are capabilities of its digester.
- Direct meaning "happens at once": an orb bomb happens at once and changes more than the score.
- A stored `CATEGORY`, or a separate meta class for tier orbs: the enum already says it, and readers would check the
  kind anyway.

**Revisit if:** an orb turns up that fits neither test.

## A direct orb is one class per way of paying, and one digester per kind (2026-11-08)

**Decision:** `FixedRewardOrb` is the direct orb that pays the same reward every time. It takes its kind and its
reward, so the negative orb and the positive orb are one class: `FixedRewardOrb(DirectKind.NEGATIVE, ...)` and
`FixedRewardOrb(DirectKind.POSITIVE, ...)`. The kind is what the agent, digestion and the renderer tell them apart by.
The reward's sign is checked by the kind's config block, not by the orb.

`DirectDigester` holds the rules for every direct kind: it pays the orb's reward, counts it in `eaten`, and owes
nothing. Each kind has a subclass that only names its kind, `NegDigester` and `PosDigester`, in the same file, and a
new direct kind adds its subclass there: the file is the whole package, so there is no `direct/` folder. The
orb decides how much it pays and the digester collects it, so a drawn or an ageing direct orb would be a class of its
own beside `FixedRewardOrb` and use the same base digester. No shared base orb class is written until such an orb
exists.

**Why:** The negative and the positive orb differ only in their kind and their sign. One class per kind would be two
near copies. The engine routes on the kind and readers look a digester up by its class, so a subclass per kind fits
both without changing the engine.

**Rejected:**

- One kind for both, with the sign in the reward: the agent could not tell a hazard from a treat. Orbs that look alike
  and pay differently are a scenario of their own, and this class serves it with one kind.
- `DirectOrb` as the name: the category also covers drawn and ageing rewards, and the name would claim them.
- One `DirectDigester` instance per kind with no subclasses: `get_digester` could not tell them apart, so the engine
  would have to look digesters up by kind.

## An orb carries its own cool-down and lifespan (2026-10-18)

**Decision:** Whether an orb returns after leaving the field, and whether it expires on it, are values of the orb,
set by the population that makes it. Each is a number of steps or `None`: no cool-down is an orb that does not return
in the episode, no lifespan is one that does not expire. `BaseOrb` answers `has_expired()` and `is_ready()`, and a
timer set to `None` never completes.

The world treats every orb alike and asks no kind. An orb with no cool-down stays inactive for the episode but is not
thrown away: the next reset places it again, and the observation keeps its slot.

Populations take these as values and know no scenario. They are in `scenario/blocks/populations/`, shared by every
family, as the counterpart of `make_digester`.

An orb that despawns gets `_reachable_life_span(rows, cols)`, the longest walk between two cells, so the droid can
reach it from anywhere.

**Why:** An orb kind stays one brick: adding the negative orb adds its bundle and changes nothing in the world. The
rule that an eaten tier orb does not return is in the orb, where no spawning rule can forget it, and the lifespan is a
value per orb, not a class attribute shared by every env in the process.

**Rejected:**

- A cool-down of 0 on tier orbs with spawning deciding per kind: the world would ask each orb its kind.
- Dropping an eaten orb with no cool-down from the world: the reset and the observation need it.
- The populations in a family folder or in `core/orbs/`: one negative orb is not a tier-chain idea, and the world is
  handed its orbs; what it owns is spawning.

## A weight belongs to a sampled field, and a tier chain holds one negative orb (2026-10-12)

**Decision:** A field is filled one of two ways, a separate question from goal or continuous:

- **Fixed:** every orb is stated and present, as in a tier chain.
- **Sampled:** orbs are drawn from a pool, and each kind's `weight` sets the mix.

`weight` belongs to a kind's block only in a sampled family, through a weighted version of the block. `NegOrbConf` is
`enabled`, `reward` and `cool_down`. A tier chain with an enabled negative block holds exactly one negative orb; the
count is not config.

Every population states how many orbs it makes, through `count`. A fixed field sums its bundles' counts for the field
size, the observation slots and the sort limit. In a sampled field the field size comes from the config, the counts
are the pool, and the family's entry type apportions the pool.

**Why:** Each block holds only what its family reads. A weight means something only relative to other weights in one
pool.

**Rejected:**

- `weight` on every negative block: a fixed field had to supply a value nothing read.
- `weight` as a continuous-only idea: a goal scenario can have a sampled field.
- One pool block listing every weight: a kind's block should stay self-contained.
- A configurable number of negative orbs in a tier chain: one is enough.
- A tally kept by hand in the family's entry type: a forgotten `+= 1` leaves the observation a slot short with no error.
- `len(population.create())`: it makes orbs only to count them.

## The world runs the orb field, and a scenario states only its size (2026-10-13)

**Decision:** Filling the field, taking orbs off it and bringing one back are the world's own behaviour:

- At reset the world places orbs until the field holds `max_active_orbs`.
- An orb leaves when it is eaten or its lifespan runs out. The droid eats an orb before expiry is checked, so it wins
  an orb on its last tick.
- After a step, if there is room and an inactive orb is ready, one comes back in a random free cell.

A scenario states only `max_active_orbs`, on its `WorldRecipe`. There is no rules class for the field and no hook in
the world for a scenario to react to an orb being eaten. An effect that belongs to an orb kind goes in its digester;
an effect that belongs to a scenario is a strategy, as termination is, or a watcher run after the world's step (see
"The world's step returns a result, and a scenario's watcher reacts to it").

**Why:** Nothing is written for a scenario that does not exist. What delay needs is decided with delay in view.

**Rejected:**

- The reference's `SpawningRules`, with three fields and three hooks: once an orb carries its cool-down and lifespan,
  only the field's size differs between scenarios.
- One empty `on_orb_consumed` hook kept for delay: it existed only for a subclass not written, and fixed how delay
  plugs in before delay was looked at.
- An orb telling the world when its timer runs out: every orb is visited every step either way, each would need a
  reference to its world, and the order of returns is clearest in one loop.

## The step clock lives in the world (2026-10-23)

**Decision:** `max_steps` is in `world_conf`, and `GridWorld` counts down `steps_left`, a read-only property. Each
action takes a step and a reset restarts it. Termination, the HUD and the metrics read it from the world; nothing
else keeps a count.

**Why:** Every consumer is handed the world already, so a count there reaches them all with nothing passed.

**Rejected:** in `obs_conf` (the observation is one reader of several); at the top of the scenario config (it is
state that changes); on the environment (it would have to be handed to every consumer); in `grid_conf` (that is
geometry); in `goal_conf` (a continuous episode also ends on the clock).

**Revisit if:** the clock needs pausing or resetting by something other than the world, such as a delay.

## A scenario mechanic lives in its strategy when the world's public API is enough (2026-10-11)

**Decision:** A mechanic one scenario uses is written in that scenario's strategy, against the world's public API and
the orbs' own methods. It becomes a world method only when it must touch the world's private bookkeeping. Spawning
stays in the world, since it uses the world's rng, free cells and its own lists.

**Why:** The world stays the same size however many scenarios are added, and a mechanic is read beside its scenario.

**Rejected:** a world method per mechanic (the world collects methods most scenarios ignore); the strategy returning
a recipe the world carries out (the world must understand every recipe).

**Revisit if:** scenarios start widening the world's public API one mechanic at a time, which is the same growth by
another route.

## The world's step returns a result, and a scenario's watcher reacts to it (2026-11-05)

**Decision:** A scenario that has to act on the world after something happened in it does so through a watcher:

1. **The world's step returns a `StepResult`,** frozen, in `grid_world.py`: the reward, and `consumed`, the eaten
   orb's meta or `None`. One orb at most is eaten per step, since no two orbs share a cell. The world calls nothing
   and holds nothing for a scenario.
2. **A `Watcher`** (`scenario/blocks/watcher.py`) has `reset()` and `react(world, result)`. It acts through the
   world's public methods and the orbs' own, and holds episode state. `Scenario.make_watcher` makes one per
   environment and defaults to `NoWatcher`, which never acts, so a scenario or family that does not use it states
   nothing. A family's own watcher and its recipe sit in the family's `watcher.py`.
3. **The environment** makes the watcher, resets it with the episode, and calls `react` after the world's step and
   before termination and the observation, so both see what it did.

Delay's `DelayWatcher` is asleep until an orb is eaten. It then freezes the other orbs of that kind on the field
with `de_spawn`, which keeps their cells, counts `delay_conf.steps` steps, and releases each with `spawn` once its
cell is free by the world's own `is_cell_free`. An orb whose cell is taken waits for it to clear. Eating a tier orb
freezes only tier orbs. The negative orb runs as in any other scenario, and eating it freezes nothing, since it is
the only one of its kind. The delay is a top-level `delay_conf` block on delay's config, beside `world_conf` and
`obs_conf`, since it belongs to the watcher and not to the world.

**Why:** Everything about delay is in delay's folder. The world gains a return value that says what happened and one
general public method, `is_cell_free`, and no setting for any scenario; otherwise each scenario would add a little to
the world and a do-nothing value to every other. A watcher holds state, which a frozen `Scenario` shared by every
environment cannot, so the scenario holds a recipe, as it does for the world. Being told by the result lets the
watcher sleep: noticing on its own would mean scanning the field every step, and counting orbs mixes an eaten orb
with an expired one or a refill. Freezing after the step gives every orb the same wait; legacy froze mid-loop, so
orbs after the eaten one ticked once more. Where a frozen orb is while it waits is in "Delay's frozen orbs stay on
the field, block their cell, and bite".

The negative orb is not frozen because an agent that has learned to avoid it treats it as noise, so the chain's
silence holds, and an optional orb that makes a scenario harder gets no rules of its own in delay. Its own cycle runs
on its own clock, so it tells nothing about the gap.

**Rejected:**

- A stage on the frozen `Scenario`: it cannot hold the countdown.
- The world owning the delay, as a setting beside `max_steps`: a piece of delay in every world, and the start of
  each scenario widening the world.
- The environment holding the countdown: delay code in the generic adapter, and its state split from the world's.
- A freeze timer on each orb: the same fact stored once per orb, and still something that sees the field has to set
  it.
- A digester holding the delay: a digester sees one orb and knows nothing of the field.
- The watcher noticing an eaten orb by counting the field: never asleep, and wrong when an orb expires or returns on
  the same step.
- Freezing every orb on any eat, the negative orb with them: the negative orb has a cool-down, so frozen the same
  way it would be refilled elsewhere mid-silence, and eating it would add a second gap to delay.
- The negative orb freezing with the field and returning with it (decided 2026-10-11 and reversed here).
- The world's free-cell check counting frozen orbs: the watcher waits for a free cell instead, and the world stays
  as it was.
- The watcher as the home of effect orbs, for now: an effect that belongs to an orb kind stays with the kind (see
  the effect-orb ideas in `codebase-ideas.md`).

**Revisit if:** a scenario needs to act inside the world's step, before it ends; termination starts reading the
field, which makes the order of the watcher and termination matter; or a watcher needs to freeze an orb that has a
cool-down.

## Episode state is first set in `reset` (2026-10-16)

**Decision:** A component's episode state is first assigned in its `reset`, not in `__init__`, which holds only what
does not change during an episode. Reading episode state before the first reset raises.

**Why:** A missing reset fails loudly at once, instead of running an episode on the constructor's values or the last
episode's leftovers. The cost: a type checker sees attributes declared outside `__init__`.

**Rejected:** starting values in `__init__`, set again in `reset`.
