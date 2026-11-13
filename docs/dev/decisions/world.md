# Decisions: The world

What the world runs: the orb field, the clock, its step and what reacts to it, and episode state.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

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
the field and block their cell".

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
  `codebase-ideas/effect-orbs-magic-from-engine.md`).

**Revisit if:** a scenario needs to act inside the world's step, before it ends; termination starts reading the
field, which makes the order of the watcher and termination matter; or a watcher needs to freeze an orb that has a
cool-down.

## Episode state is first set in `reset` (2026-10-16)

**Decision:** A component's episode state is first assigned in its `reset`, not in `__init__`, which holds only what
does not change during an episode. Reading episode state before the first reset raises.

**Why:** A missing reset fails loudly at once, instead of running an episode on the constructor's values or the last
episode's leftovers. The cost: a type checker sees attributes declared outside `__init__`.

**Rejected:** starting values in `__init__`, set again in `reset`.
