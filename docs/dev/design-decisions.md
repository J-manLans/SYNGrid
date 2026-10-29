# Design Decisions

What is settled, why, and the alternatives someone might suggest again, so that nobody has to guess why something is
shaped as it is or put back what was left out on purpose.

Written in the present tense. The story of how a decision changed is in `git log`; a reversal that still matters is a
rejected alternative here. File paths, test runs and dates of work stay out, and so do open points, which go in
`open-questions.md`. When a decision changes, the entry is rewritten to say what is true now.

---

## Entry template

### Title (date last settled)

**Decision:** what is true.
**Why:** the reason.
**Rejected:** alternatives worth not suggesting again, each with its reason.
**Revisit if:** the condition that would reopen it.

---

## Contents

- **Principles**
- **Digestion:** digestion in the droid; digesters do not talk; the tier digester; scoring modes; a new mode is a new
  scenario; recipes
- **Reward and endings:** energy and score; truncation; who pays what; the reward ladder; normalisation; the
  ablations; legacy as a reference
- **Orbs and the world:** two categories; cool-down and lifespan; weights and counts; the orb field; the step clock;
  mechanics in strategies; episode state in `reset`
- **Scenarios and config:** the package; config blocks; orb config classes; the single chain; the family's name;
  locked scenarios
- **Observation:** the kept perceptions; no timer; shared readers
- **Runners and rendering:** the algorithm enum; the policy name; human play; the renderer; window input

---

## Principles

The rules most decisions below follow.

- **Everything scenario-specific is a field of `Scenario`.** The test for a field is "does the Gymnasium side need
  it?"; what only the world needs stays inside `build_world`.
- **A `Scenario` holds nothing that changes during an episode.** Training builds 16 envs from one `Scenario` in one
  process. State belongs in what `build_world()` returns; rules reach it through the `world` they are handed.
- **A conditional that exists "because scenario X works differently" does not belong in the component that holds
  it.**
- **A mechanic only one scenario uses lives in that scenario's strategy** when it can be written against the world's
  public API and the orbs' own methods. It becomes a world method only when it must touch the world's private
  bookkeeping.
- **One supplied piece per consumer,** each a static description plus a reader that is handed the world: `hud`,
  `metrics`, and the observation's global features in `ObservationRules`. The HUD and the observation are free to
  diverge.
- **Readers are module-level functions, not lambdas or nested functions,** so two scenarios built from the same
  config compare equal.
- **Termination is an interface file plus one file per strategy,** so each strategy imports only what its scenario
  needs.
- **One orb kind is one brick:** an `OrbBundle` in the scenario layer, carrying the population and a way to make the
  digester, and a `*_orb_conf` block in the world config.

---

# Digestion

## Digestion lives in the droid, with a folder of digesters (2026-10-12)

**Decision:** Digestion is the droid's own organ, in `core/droid/digestion/`:

- `digester.py`, the `OrbDigester` protocol: `KIND`, `reset`, `digest` returning the reward as a float, and
  `undigested`. `KIND` is the routing key, an `OrbKind` from `core/orbs/orb_meta.py`.
- `engine.py`, a router with no scoring rules. It hands an eaten orb to the digester for the orb's kind, and lets a
  reader fetch a digester by class with `get`, which matches with `isinstance`, so asking for `TierDigester` finds
  whichever scoring mode is registered.
- `digesters/`, one digester per orb kind, a folder for a kind with several files.

The engine raises on every sign of wrong wiring: a `ValueError` when two digesters own a kind, a `KeyError` when an
eaten orb's kind has no digester or `get` finds no digester of the class.

The droid creates the engine from the digesters it is handed. The scenario holds a way to make each digester, one
per `OrbBundle` (`make_digester`), and never a digester, since a digester holds episode state and one `Scenario` is
shared by every env in the process. `build_world` makes them, so every world gets its own.

**Why:** The engine stays a plain mechanism any family can use. The scenario decides which digesters exist and how
each is configured, but does not own their code. The droid never imports a concrete digester or learns its scenario.

**Rejected:**

- `core/digestion/`, or digesters in the scenario package: digesting is something the droid does.
- `build_world` or the scenario creating the engine: nothing outside the droid needs to.
- One scenario method that returns every digester: adding an optional orb kind would then mean editing it as well as
  adding the bundle.
- `get` returning `None`, an unrouted orb paying 0, or a second digester replacing the first: each can only come from
  a builder wired wrong, and each would go unnoticed. A metric reading 0 because its digester is missing looks like
  one reading 0 because nothing was eaten.

## Digesters do not talk to each other (2026-10-12)

**Decision:** A digester scores its own kind and knows nothing of the others. `digest` returns a float. There are no
events, no result object, no tally in the engine and no hook by which a digester hears of another kind's orb.

A tier chain is broken only by a tier orb eaten out of order. A negative orb costs its own reward and leaves the chain
alone.

A reader that needs what a digester knows asks the droid, `world.droid.get_digester(TierDigester)`, which forwards to
the engine's `get`, and reads its attributes. The engine stays private to the droid.

**Why:** Each digester can be read and tried alone, and each kind has one meaning: a tier orb is about order, a
negative orb costs what its reward says. Adding a negative orb to a scenario changes one thing, so its effect on a
learning curve can be read.

**Rejected:**

- Events returned from `digest` and counted by the engine (the reference): nothing reacted to an event, every reader
  wanted a count since reset, which a digester already keeps.
- A `notice` hook on digesters: every pair becomes a possible conversation, the result depends on which is told
  first, and a step's reward can no longer be explained from one place.
- Any non-tier orb breaking a chain (legacy): a direct orb changes the score and nothing else.
- A public `digestion_engine` attribute on the droid: every reader would walk three levels into it.

**Revisit if:** an orb has to change what another kind pays (see "Effect orbs get their magic from the engine" in
`codebase-ideas.md`), something has to react to what happened and not only to what it paid, or a scenario wants an
existing kind to break chains, in which case the builder hands the tier digester the set of breaking kinds.

## The tier digester is a template with one subclass per scoring mode (2026-10-15)

**Decision:** `TierDigester` is a base class that owns the chain. For each tier orb it decides what happened, asks the
subclass what that pays, counts it and moves the chain:

- **Progress:** the next tier, not the last. A tier 1 with no chain running starts one.
- **Completion:** the next tier, and the last. The chain is over.
- **Break:** any other tier. A tier 1 starts a new chain at 1; any other tier leaves none.

A mode writes `_score_progress`, `_score_break` and `_score_completion`, each handed the orb and called before the
chain moves, and `undigested`, what the chain has gathered and not paid. The base holds what readers take:
`chain_tier`, `chain_worth`, and the counters `chains_progressed`, `chains_broken` and `chains_completed`, as plain
public attributes. The base takes `max_tier`; each mode takes what it pays with. The builder picks the subclass.

**Why:** Adding a mode is adding a class, not editing one that works. A scenario loads only its mode, and each mode
can be tried alone.

**Rejected:**

- One digester with a `ScoringMode` enum and a branch per mode (the reference): the builder already decides the mode,
  so the enum states it twice, and it is a conditional in the wrong component.
- The orb carrying its own scoring (the old engine): orbs are the same in every mode.
- `max_tier` read off the orb (legacy): `TierOrb` does not store it.
- The last orb's reward held in the base (legacy's `_pending_reward`): no mode reads it.

## The scoring modes pay the same orbs at different times (2026-10-12)

**Decision:** A tier orb is worth the same in every mode; a mode decides only when it is paid. A break always costs
`chain_break_penalty`.

| | Progress | Break | Completion | Owed at the end |
|---|---|---|---|---|
| Step-wise | the orb's reward | the penalty, plus the orb's reward if it restarts a chain | the orb's reward | 0 |
| Threshold | 0 | the penalty, plus what the chain had gathered | what the chain had gathered, plus the orb's reward | what the chain has gathered |
| Max-tier | 0 | the penalty | what the chain had gathered, plus the orb's reward | 0 |

A chain completed without a break pays the same total in all three, and step-wise and threshold pay the same total for
any sequence once threshold has paid what it holds.

The break is one rule everywhere: a tier orb that is not the next one. A tier 1 that breaks a longer chain is the
first orb of a new one and counts towards its `chain_worth`; in step-wise it is paid its reward on top of the
penalty, so a chain completed after a restart pays its full worth in every mode. Step-wise and threshold share one
exception: a tier 1 eaten while the chain is at 1 pays the penalty alone, or tier 1 orbs could be farmed where orbs
respawn.

**Why:** The modes form one family, so comparing two isolates when the reward arrives. The agent gets one rule: eat
the tier one above the chain.

**Rejected:**

- Max-tier paying a fixed prize with tier orbs worth nothing (the reference; legacy's fixed 10): sparse against dense
  at one chain length is what the scaling scenarios compare, and with the same worth only the timing differs. The
  cost is accepted: the prize grows with the chain unless the base reward is set per scenario.
- Max-tier paying only the last orb's reward (legacy): the lower tiers would never be paid.
- A completion bonus in step-wise (legacy): a completed chain would be worth nearly double max-tier's.
- A separate `tier_consumption_penalty` (legacy): a second number for the same idea.
- A tier 1 never costing a penalty (legacy step-wise): it can be farmed.
- A threshold break paying only the last tier (legacy), or no penalty: a special case, and a free cash-out.
- A restarted chain's sum starting at 0: a completed chain should always pay the same.

**Revisit if:** a goal scenario lets tier orbs return, since a restarting tier 1 then pays for a chain that may never
grow; or long chains train worse because the prize grows (try the discount and exploration first, then the base
reward, then reward normalisation).

## A new scoring mode is a new scenario (2026-10-11)

**Decision:** Scoring mode is not config. A different mode is a new registered scenario with its own builder, reusing
the family helper. The continuous sandbox is the exception.

**Why:** The scenario name is the run's identity: run ids, result folders and comparisons hang on it. A mode in config
would let two runs of one name pay differently.

**Rejected:** the mode in the YAML with a lookup from its name to a class.

**Revisit if:** one experiment has to sweep modes. The lookup would sit at the edge, beside `SCENARIOS`, and not touch
the digesters.

## A tier bundle makes its digester from a frozen recipe (2026-10-14)

**Decision:** A tier bundle's `make_digester` is a `TierDigesterRecipe`, a frozen dataclass of `scoring_mode` (the
tier digester class), `max_tier` and `chain_break_penalty`. Calling it returns a new digester. A builder states its
mode by naming the class, through `_tier_bundle`. A digester whose constructor takes no values, as `NegDigester`, is
its own `make_digester`; there is no recipe base class.

`scenario/blocks/recipes/` holds every recipe: a frozen object a scenario holds that makes something with episode
state when a world is built. The other is `WorldRecipe`, a scenario's `build_world`, shared by every family. It holds
the world config, the bundles and `max_active_orbs`, reads only what the base `WorldConf` has, and makes the
digesters, a `SynergyDroid` (with the grid, `max_energy`, `step_penalty` and `boundary_penalty` as plain values) and
the `GridWorld`. The full world config is reachable as `scenario.build_world.world_conf`.

Because both compare by value, two scenarios built from one config compare equal. That lets `==` answer "is this the
scenario I think it is?" when a scenario is locked, when a saved agent is evaluated and when a GUI saves one. It
describes the scenario, not its behaviour; that is for a replay with one seed.

A digester sets `self.KIND: Final[OrbKind]` in `__init__`. A `Final` with a value in the class body is a class
variable to Pyright and does not satisfy the protocol's instance variable.

**Why:** It compares by value, it is the one place a mode is chosen, and the choice is a class, not a name to look up.

**Rejected:**

- A `functools.partial`: two partials never compare equal.
- A module-level function per mode: it has nowhere to get `max_tier` and the penalty.
- The recipe in the family's builders, or in `core/`: it holds nothing of the family, and core is handed digesters.
- Recipes handed down a layer at a time (the world given a droid maker, the droid given digester recipes): nothing is
  gained, and it puts makers in core's constructors.
- Inheriting from the digester protocol: Pyright refuses to assign a `Final` the parent declares, and a subclass that
  forgets a method inherits an empty one.

---

# Reward and endings

## Energy and score are two numbers, and every reward goes through the droid (2026-10-16)

**Decision:** The droid keeps both, and every reward the agent is paid goes through it.

- **Energy** is what the droid has left. It starts at `max_energy`, stays between 0 and `max_energy`, and the episode
  ends when it reaches 0, in every scenario. One config value is the start and the cap.
- **Score** is the episode's return, the sum of every reward the environment hands back. It starts at 0 in `reset`,
  has no bounds and no config, and equals what SB3's `Monitor` logs.

| Path | What it carries |
|---|---|
| `take_step` | the step penalty and the boundary penalty |
| `consume_orb` | what digestion pays for the eaten orb |
| `on_termination(ending_reward)` | what digestion is owed (the engine's `undigested`) plus what termination says the ending costs; to the score only |

The environment's step: the droid acts, termination decides, and if `outcome.terminated` the environment adds
`droid.on_termination(outcome.ending_reward)`. Termination decides on the state before anything is settled, so a
payout cannot change whether the episode ends. The observation, the HUD and the metrics are read after, so the last
frame shows the final score and the energy the episode ended with.

**Why:** With one door for every reward, energy and score cannot drift from what the agent was paid, and every
reader handed the world reaches both. The cost: a reward from somewhere new needs its own path, decided then. The
ending's payout leaves energy alone because the episode is over; with it in energy, a droid that ran out while holding
a threshold chain showed energy again on its last frame.

**Rejected:**

- One number that was both (the old `starting_score`): the two meanings want different bounds and starts.
- Only penalties moving energy: energy responds to both.
- No cap, with a `max_score` to bound the observation: a cap gives an exact bound and removed three config classes.
- The score kept by the environment: energy still missed what termination added, and the scenario's readers could not
  reach it.
- Termination adding the timeout and what is owed to the reward itself, or calling the droid in `evaluate`: the
  score missed them, and a rule that decides would also change the world.
- A general `receive(reward)` on the droid: a general door makes "every reward goes through the droid" too easy to
  satisfy. Settling and paying the ending are one event, so one call.
- `end_episode` as the name: it covers truncation, where nothing is settled.
- A held chain reviving a droid that runs out while steps remain: it lets a payout change whether the episode ends.

**Revisit if:** a continuous scenario needs a droid to bank rewards beyond the cap, which the old uncapped number
allowed.

## A truncated episode settles nothing (2026-10-25)

**Decision:** `on_termination` runs only when an episode is terminated. On a truncation nothing is settled: what a
digester holds is not paid and no `ending_reward` is taken in. Only a continuous scenario truncates.

**Why:** At a truncation the learner bootstraps from the last observation (SB3 adds `gamma * V(terminal_obs)`), and
that value stands for the rest of the episode, including what the droid holds. Paying it out as well counts it twice.
A termination is a real end; what the droid holds is paid then or never. The value counts what the droid holds only if
the agent sees or remembers it, which is the open question on chain progress.

**Rejected:** settling on either ending; settling on truncation for a learner that treats it as terminal (that learner
is wrong about Gymnasium's contract).

**Revisit if:** a continuous scenario has something that should be paid at the cut. A new algorithm does not reopen
it, but check that it bootstraps at truncation.

## Digestion pays for orbs, and termination pays for the clock (2026-10-15)

**Decision:**

| Moment | Paid by | Pays |
|---|---|---|
| Chain break | the tier digester | `chain_break_penalty`, plus what the chain had gathered in threshold |
| Completion | the tier digester | what the mode says the chain is worth |
| Any ending | the droid's `on_termination` | what its digesters say they are owed |
| Timeout | termination, through `on_termination` | `timeout_penalty`, on top of the row above |

Termination only adds to the step's reward, and returns `EpisodeOutcome.ending_reward` and changes nothing. The
clock is charged only when it alone ended the episode: a chain completed or broken on the last step, or energy running
out on it, pays no `timeout_penalty`. What is owed is settled at every ending, so threshold and step-wise total the
same over an episode. `undigested` is abstract, so every mode has to state it.

There is no `completion_reward`. `timeout_penalty` is in `goal_conf`, since any goal with a deadline has it;
`chain_break_penalty` is in `tier_orb_conf`. Both are required, with no default.

A goal scenario never truncates: its deadline is part of the task, so running out of steps is a failed attempt and
sets `terminated`. A scenario with no objective truncates on the clock, which makes it continuous.
`EpisodeOutcome.truncated` defaults to false. One termination class serves all three modes.

**Why:** A digester does the same in a goal and a continuous scenario; only whether termination ends the episode
differs. What reaching the goal pays is in one place.

**Rejected:**

- A `completion_reward` in termination: completing is caused by eating an orb, what it pays follows the mode, and a
  scenario where completion does not end the episode has no termination to pay it.
- A rule per mode inside termination, or a termination per mode: a mode conditional in termination, or the mode
  chosen in two places that can disagree.
- Nothing settled when energy reaches 0 (legacy): a threshold chain would be paid at every ending but one.
- The timeout replacing the step's reward (legacy): the last step's penalties and orb were lost.
- Threshold paying no `timeout_penalty` (legacy): the clock running out is the same event in every mode.
- An outcome type without `truncated` for goal scenarios: the environment hands Gymnasium both flags every step.
- Optional penalties or defaults of 0: a forgotten `timeout_penalty` would train with none.

**Revisit if:** a goal is something no single digester sees, such as surviving or clearing the field; a
scenario-level completion reward earns its place there.

## The reward ladder is one formula on the shared tier orb block (2026-10-11)

**Decision:** A tier orb is worth `base_reward * tier ** growth_factor`, both fields of the shared `TierOrbConf`, not
rounded. Scaling dense has no config classes of its own. Spatial uses a base of 1 and a growth factor of 1.5, so its
three tiers are worth about 9.0 together, near legacy's fixed 10.

**Why:** One formula with no setting that is ignored. The saved thesis configs all used non-linear growth.

**Rejected:** a `linear_reward_growth` switch (legacy), which left the growth factor required and ignored; rounding
(legacy, one branch only); the ladder on dense's own block; a base of 2 with linear growth for spatial.

## Reward normalisation stays off (2026-10-11)

**Decision:** Training keeps `norm_reward=False` on `VecNormalize`. Observations are normalised; rewards are not.

**Why:** The rewards are designed to sit in a small range. With sparse rewards the running estimate is tiny until the
first success and shifts when it comes. The 81 saved thesis runs were trained without it, and the owner recalls it as
deliberate.

**Revisit if:** a continuous scenario's return grows from many chains, or a chain's total goes well beyond a few tens.

## The ablations kept the step penalty, and sparse's chain break penalty (2026-10-27)

**Decision:** Spatial and scaling sparse keep `step_penalty: -0.001`, and scaling sparse keeps
`chain_break_penalty: -0.1`, the thesis values. They are locked with the scenarios' other rules.

- Spatial on 5x5 with no step penalty did not learn: mean reward about 1.2 against a ceiling of about 9.02, where
  the run with the penalty reached about 8.5.
- Scaling sparse at tier 4 with no step penalty reached its maximum faster.
- Scaling sparse at tier 4 with no chain break penalty did slightly worse.

**Why the step penalty matters although it is small:** PPO normalises advantages in each minibatch, so the learner
sees how rewards differ, not how large they are. Early on almost every step pays 0, and the step penalty is nearly
the only thing that tells a short episode from an aimless one. Where completions are frequent from the start, the
discount is enough, which fits sparse at tier 4; where they are rare, the penalty carries the agent to its first
completions, which fits spatial. This explanation fits both results and is not tested on its own.

**Why sparse keeps it although tier 4 was faster without:** `max_tier` is a run's choice, and the locked rules must
hold at every tier. Long chains make early completions rare, which is where spatial failed. The thesis ran -0.001 at
tiers 4 to 7.

**Why sparse keeps its break penalty:** a break ends the episode, so without a penalty it is a free way out, cheaper
than walking to the -1 timeout.

**Spatial's break penalty was not ablated.** It was tuned in the thesis era: under a limited view, eating an unknown
orb is how the droid explores, and a cheap break keeps that cheap. A larger penalty taught the droid to avoid orbs
and run out the clock.

**Rejected:** no step penalty in sparse on the tier 4 result alone; no step penalty in any goal scenario (refuted by
spatial); a tier 6 run before deciding (keeping the penalty is safe whatever it shows).

## Legacy is a reference, not a target (2026-10-25)

**Decision:** The legacy strain and `reproduction_package/` are references. The rebuild is held to learning in the
same general direction as legacy over several seeds, by comparing curves, not to legacy's rewards or numbers. They
are also where to look up what a behaviour used to be.

These differ from legacy on purpose:

- A max-tier chain pays what its orbs are worth, not a fixed 10.
- A negative orb does not break a chain.
- Step-wise has no completion bonus, uses `chain_break_penalty`, and charges it for a tier 1 out of order.
- A threshold break pays the penalty plus everything gathered, not the last tier's reward.
- An ending adds to the step's reward instead of replacing it; a threshold timeout pays `timeout_penalty`; a threshold
  chain is paid what it is owed when energy reaches 0.

**Why:** Holding the rebuild to legacy's numbers would keep behaviour that was decided against.

**Rejected:** an exact per-step match of rewards, or reproducing legacy's results, as the bar.

---

# Orbs and the world

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

**Why:** Direct stays a small closed set, which lines up with the digesters: a direct digester is stateless, a
synergy digester holds episode state. A new orb is an enum member and a digester, with no new field.

**Rejected:**

- An `EffectKind` enum beside `SynergyKind`: it reads as if effect orbs were not synergy.
- One `SynergyKind.EFFECT` with a sub-field: each effect needs its own digester, so the key must tell them apart.
- Storing how an effect acts in the meta: those are capabilities of its digester.
- Direct meaning "happens at once": an orb bomb happens at once and changes more than the score.
- A stored `CATEGORY`, or a separate meta class for tier orbs: the enum already says it, and readers would check the
  kind anyway.

**Revisit if:** an orb turns up that fits neither test.

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
`reward` and `cool_down`. A tier chain with a negative block holds exactly one negative orb; the count is not config.

Every population states how many orbs it makes, through `count`. A fixed field sums its bundles' counts for the field
size, the observation slots and the sort limit. In a sampled field the field size comes from the config, the counts
are the pool, and the family's builder apportions the pool.

**Why:** Each block holds only what its family reads. A weight means something only relative to other weights in one
pool.

**Rejected:**

- `weight` on every negative block: a fixed field had to supply a value nothing read.
- `weight` as a continuous-only idea: a goal scenario can have a sampled field.
- One pool block listing every weight: a kind's block should stay self-contained.
- A configurable number of negative orbs in a tier chain: one is enough.
- A tally kept by hand in the helper: a forgotten `+= 1` leaves the observation a slot short with no error.
- `len(population.create())`: it makes orbs only to count them.

## The world runs the orb field, and a scenario states only its size (2026-10-13)

**Decision:** Filling the field, taking orbs off it and bringing one back are the world's own behaviour:

- At reset the world places orbs until the field holds `max_active_orbs`.
- An orb leaves when it is eaten or its lifespan runs out. The droid eats an orb before expiry is checked, so it wins
  an orb on its last tick.
- After a step, if there is room and an inactive orb is ready, one comes back in a random free cell.

A scenario states only `max_active_orbs`, on its `WorldRecipe`. There is no rules class for the field and no hook,
and no hook for a scenario to react to an orb being eaten. An effect that belongs to an orb kind goes in its digester;
an effect that belongs to a scenario is a strategy, as termination is.

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

## Episode state is first set in `reset` (2026-10-16)

**Decision:** A component's episode state is first assigned in its `reset`, not in `__init__`, which holds only what
does not change during an episode. Reading episode state before the first reset raises.

**Why:** A missing reset fails loudly at once, instead of running an episode on the constructor's values or the last
episode's leftovers. The cost: a type checker sees attributes declared outside `__init__`.

**Rejected:** starting values in `__init__`, set again in `reset`.

---

# Scenarios and config

## The scenario package is blocks, families, a registry and a scenario (2026-10-12)

**Decision:** `scenario/` holds:

- `blocks/`: what a scenario is made of that no family owns, the interfaces and rule classes as files, and the
  shared implementations in `populations/` and `recipes/`.
- `families/`: the scenarios. A group such as `goal/` holds what its families share; a family folder such as
  `single_tier_chain/` holds its config models, builders and its own strategies.
- `registry.py`, which selects a scenario by name, and `scenario.py`, the `Scenario`.

A strategy only one family uses stays with that family.

**Why:** The top level reads as its three roles: the pieces, the scenarios, and the table that selects one.

**Rejected:** `populations/`, `recipes/` or `goal/` at the top level, mixing pieces with scenarios; the population
protocol as `blocks/population.py` beside a `populations/` folder.

## Scenario configs live with the scenario, as blocks in the world (2026-10-11)

**Decision:** A config model lives beside what it configures, at the level where it is shared.
`config/models/scenario/` holds what every scenario is built from (`common_models.py`, `orb_models.py`); a group's
`config.py` holds what the group shares; a family's `config.py` holds the family.

A scenario config is `world_conf`, everything the world acts on, and `obs_conf`, what turns the world into an
observation. Each layer adds a block to the world:

```
ScenarioConf                    world_conf, obs_conf
  WorldConf                       max_steps, grid_conf, droid_conf
    GoalWorldConf                 + goal_conf       (timeout_penalty)
      TierWorldConf               + tier_orb_conf, neg_orb_conf (optional)
        TierDelayWorldConf          tier_orb_conf + delay
```

Each world model has a scenario model that narrows `world_conf` to it. Each orb kind has its own `*_orb_conf` block,
mirroring `OrbBundle`. `SCENARIOS` in `scenario/registry.py` gives a name its config class and builder.

**Why:** The config tree and the scenario tree are one tree, so a builder and its config are found in one folder.

**Rejected:** a separate config tree mirroring the scenario package (two trees drift); goal and tier values on the
droid with the orbs in one pool (the digester and termination read them, and one leaf took four classes); goal and orb
blocks beside `world_conf` (the world acts on them); a `mode: goal` field (the name says it).

## An orb kind's config class is shared, and each scenario declares the field (2026-10-11)

**Decision:** `orb_models.py` holds the class that describes an orb kind wherever it appears. A family adds what
describes its use of the kind, as a subclass. A scenario's world model declares the field, optional or mandatory. The
base `WorldConf` declares no orb fields. A base class holds only fields certain to be shared: `reward` and
`cool_down` for the negative orb; `max_tier`, the reward ladder and `chain_break_penalty` for the tier orb.

**Why:** What an orb is gets written once, and each scenario states its relationship to it in one line.

**Rejected:** `neg_orb_conf` on the base `WorldConf` (every scenario would accept it, and none could make it
mandatory); a class per scenario (they drift); a class kept in its first family until a second needs it (a second is
already expected).

## Every scenario in the single tier chain family is a single chain (2026-10-18)

**Decision:** The family is the scenarios whose whole chain is laid out at once and left until it is eaten, whatever
`max_tier` is: spatial, scaling sparse, scaling dense and delay. The family states once, in `_tier_bundle`, that its
tier orbs have no cool-down and no lifespan, and no scenario can change it.

**Why:** A value every member shares belongs to the family; an argument nobody varies invites the question why it is
there.

**Rejected:** a `life_span` argument on `_tier_bundle` (it read as if spatial were special); keyword overrides (a
scenario whose tier orbs return is not a single chain and belongs in another family).

**Revisit if:** a scenario is added whose tier orbs expire or return.

## The family is named single tier chain (2026-10-26)

**Decision:** The family's names say what it guarantees: `scenario/families/goal/single_tier_chain/`,
`build_single_tier_chain_*`, `_single_tier_chain_scenario`, `SingleTierChainTermination`, the scenario names
`GOAL_SINGLE_TIER_CHAIN_*` and `goal_single_tier_chain_*.yaml`. The mechanic's names describe the orbs and stay:
`digesters/tier_chain/`, `TierChainPopulation`, the HUD label `current tier chain`.

The scenario name is also the runner's output folder, so runs from before the rename are under `goal_tier_chain_*`.

**Why:** The guarantee and the mechanic together name the family, and each half can vary: `multi_tier_chain`,
`single_color_chain`.

**Rejected:** `tier_chain` (the mechanic's name too); `single_chain` (it leaves out which chain); waiting for the naming
pass (phase 6 adds three builders to the family).

## A locked scenario loads a small model, and its builder makes the full config (2026-10-12)

**Decision:** The layered models stay the full description of a world. A locked scenario gets one small model of its
own, frozen, strict and `extra="forbid"`, holding only what a run may choose. Its `SCENARIOS` entry names the small
model, so a YAML that states a fixed value is rejected. Its builder makes the full config from the scenario's fixed
values and the run's choices, and hands it to the family helper. Nothing below the builder knows whether a scenario
is locked. A scenario that is not locked, and the sandbox, load the full model.

A scenario is locked once runs over several seeds learn in the same general direction as legacy.

**Why:** A locked scenario's fixed values pass the same validation as a YAML's. The entry's config class says what a
run may set, which is also what a GUI needs to draw a form. Frozen models compare equal, so the builder's config can
be checked against the one the old YAML loaded.

**Rejected:**

- Fixed values as defaults on the models: a default can be overridden, so two runs of one name could differ.
- A narrow class and a tunable subclass per block: no single narrow class suits every scenario.
- Thinning the base models: the helper and `WorldRecipe` would read from two sources.
- Loose constants passed around the builder: they skip validation.
- One full model for all, with a GUI choosing fields: the per-scenario list would live in the wrong layer, and the
  YAML could still set anything.

## A locked scenario is what was run, and changing it makes a different scenario (2026-10-27)

**Decision:** A scenario's name promises one test, so a locked scenario is locked to what was run. Spatial is tier 3,
seen through the fog-of-war window, with the penalties and orb rewards the ablations kept, fixed in its builder. The
grid size is its only dial on the world. The other locked scenarios follow with their own axis: the chain length and grid for scaling sparse,
the delay for delay. `max_energy` is a rule fixed in the builder with the penalties, not a size: with a step penalty
of -0.001 energy is mostly spent on boundary hits, so it caps pointless moves.

Tier 3 is a reference point, not a limit. Spatial at tier 4 is a new scenario: a short builder and a registry entry.

**Why:** A benchmark is useful only if two people who say "I ran spatial" mean the same test. The limit need not be
drawn in the YAML: a GUI can open a locked scenario read-only and hand out named copies (see "A GUI forks a locked
scenario instead of editing it" in `codebase-ideas.md`).

**Rejected:** `max_tier` tunable with the tag recording it ("spatial" would name a family, not a test); every value
left in the YAML (the file is both the reference and the thing edited).

## A locked scenario fixes the visibility, and a run chooses the encoding (2026-10-29)

**Decision:** A locked scenario fixes how much the droid sees, and a run may pick any perception with that
visibility, whatever its encoding. The scaling scenarios take any fully observable perception: vector, composite or
pixel. Spatial takes any partially observable perception with the same hindrance as `vector_fog_of_war`, the window
around the droid.

**Why:** What a scenario tests decides how much the droid may see, and nothing else about the observation. Two
spatial runs with different encodings face the same task; a spatial run that sees the whole grid does not. Encoding
is a real choice for a run: a composite observation with its own extractor may suit a model better, at a cost in
speed.

**Rejected:** one perception fixed per locked scenario, `vector_fog_of_war` for spatial included (it takes away a
real choice and ties the test to one encoding); any perception for any scenario (a spatial run that sees the whole
grid does not test spatial).

---

# Observation

## The kept perceptions observe only what earned its place (2026-10-23)

**Decision:** `vector_markovian` and `vector_fog_of_war` observe the droid's position and one block per orb slot
(active flag, row, column, identity), and no global values: no steps left, score, chain tier or orb timer. The single
tier chain builders set `global_features=()`.

**Why:** The owner ablated an observation like `vector_markovian_easy` one value at a time and removed every value
without which the agent did as well or better. A value the agent does not need costs input and gives the policy
something to fit.

**Rejected:** the global values in front with an optional timer per orb (the ablation's starting point).

**Revisit if:** a scenario has to be solved from a value these leave out, such as threshold's gathered sum. When the
legacy perceptions are rewritten, this is the reason not to add their global values to the two kept ones.

## An orb's timer is not observed (2026-10-23)

**Decision:** The agent is not shown how long an orb stays or how long until it returns. It sees an orb when it is on
the field. A rewritten perception gets no timer by default.

**Why:** Delay tests whether an agent can bridge a gap it cannot see the end of. A countdown would let it read the gap
off the input, and the memory question would shrink to nothing; withholding it is the measurement. Spatial works the
same way: the agent remembers what it is not shown. That a stateless agent cannot see an orb vanish is the
stateless-versus-memory difference the benchmark measures. Chain progress is not a timer but the agent's own history.

**Rejected:** showing the time left on an orb, and for a missing orb the time until it returns.

## The HUD and the observation share readers and keep their own lists (2026-10-23)

**Decision:** A number the HUD shows and the observation carries is read by the same module-level reader, and each
consumer lists it in its own tuple: a `HudBar` or `HudStat` in `hud`, a `GlobalFeature` in
`ObservationRules.global_features`. Readers every scenario can use are in `scenario/blocks/readers.py`; a family's own
sit beside its builder.

**Why:** The two diverge on purpose: the HUD shows energy, score, steps left and the chain, and the observation none.
The HUD needs a label and a bound only for a bar; the observation always needs a bound and a fixed order, and score
has no upper bound for a `Box`.

**Rejected:** one list with a flag per consumer (the flags grow); a shared base type (it couples the two files for two
saved arguments; worth it if a third consumer appears); the readers in `hud.py` (the observation would seem to depend
on the HUD).

---

# Runners and rendering

## The algorithm in the runner config is an enum (2026-10-13)

**Decision:** `alg` in the runner config is an `Algorithm` enum (`PPO`, `FSPPO`, `RPPO`), and the `RUNNER` table is
keyed by it, as `ScenarioName` and `SCENARIOS` are. A wrong name fails when the config loads. The field is
`Annotated[Algorithm, Field(strict=False)]`, since a strict enum field rejects YAML's plain string. The run id uses
the member's `.value`.

**Why:** A typo is caught at load, and the valid names are stated where they are used.

**Rejected:** `alg: str` checked in `build_runner` (a typo found only after the scenario was built); the member in an
f-string (a `str` enum formats differently from Python 3.11, and checkpoints are found by that id).

## The policy's name comes from the algorithm (2026-10-13)

**Decision:** `resolve_policy(perception, algorithm)` returns the one name in the algorithm's `policy_aliases` that
starts with the perception's base policy, and raises if there is not exactly one. The base SB3 runner calls it; a
concrete runner says nothing about the policy. The table from a perception's leading word to a base policy stays:
`vector` to `Mlp`, `composite` to `MultiInput`, `grid` to `Cnn`.

**Why:** It removes the LSTM special case, and naming the algorithm names the policy family.

**Rejected:** a `use_lstm` flag (the next variant needs another); a suffix or variant per runner (it restates what the
algorithm class says); perceptions named after SB3's policies (SB3's wording would reach the config).

**Revisit if:** an algorithm offers two policies with one base, or a perception's name stops starting with its kind.

## Human play is not an agent runner (2026-10-13)

**Decision:** `HumanRunner` is a plain class with `play()`. With `human_control` set, `app.main` runs it and returns
before an agent runner is built. Human control overrides the snapshot and the runner config, with no check.

**Why:** Human or agent is a choice about the whole run, made once, in `main`.

**Rejected:** `HumanRunner` as a `BaseAgentRunner` with a `train()` that raises (it inherited what it could not use);
rejecting `human_control` with a snapshot on (a human run is for trying something, and should start without edits).

## The renderer takes plain values, and the HUD is the scenario's list (2026-10-21)

**Decision:** The renderer is handed plain values and never reads the world or a config.

- From the world, through the environment: the droid's position, the orbs on the field and `grid_dimensions`.
- From the scenario, through `scenario.hud`: everything else the HUD shows, as static elements (a label, a kind, a
  maximum for a bar, a reader). The environment calls the readers and hands the values over.
- A scenario says what to show and in what order, never a pixel. Placement belongs to the renderer and its theme
  file. The images are constants in the renderer, the frame rate comes from the env's metadata; there is no
  `RendererConf`.
- The fog overlay dims cells farther from the droid than a view radius on `ObservationRules`, the same one the
  perception reads.
- The HUD is to be stacked from drawn pieces (a top cap, a tiling row piece, a bottom cap), drawn by element kind,
  so a new element adds a row and no number. The theme file holds anchors inside each piece.
- The game HUD stays in plain pygame; pygame_gui is for the config GUI.

**Why:** The renderer stays free of `core/` and config, the HUD's look is edited as data, and a new scenario adds
elements without a change in the renderer.

**Rejected:** a generic labelled-rows layout (the HUD is part of the look and feel); a placement per scenario
(rendering in the scenario layer); energy, score and steps as fixed furniture; pygame_gui for the game HUD (built for
interactive UI); slots painted into one picture (it caps the count).

## Runners own the window's input, the renderer draws, and the env closes the window (2026-10-25)

**Decision:** The renderer reads no input. It draws a frame, an episode summary or the pause screen, sets the title
and closes the window. Each runner reads its own keys.

- `HumanRunner` blocks on pygame's event queue, takes the arrow keys as actions and quits on Escape or the close
  button. When an episode ends it shows a summary of the HUD values (from the env's `get_hud_data()`), the metrics and
  the last reward; Space plays again, Escape quits, other keys are ignored.
- Agent runs with a human renderer share run controls in `BaseAgentRunner`: Space pauses, Escape or the close button
  quits. Eval calls them before each step; training through `RunControlsCallback`, added only when training renders
  to a window.
- Quitting returns, it does not raise: eval returns from its loop, and the callback returns False from `_on_step`,
  SB3's way to end `learn()`, after which training saves.
- `SYNGridEnv.close()` closes the window, and every runner closes its env in a `finally`.

**Why:** The renderer has one reason to change, how things look. Stopping and closing use what SB3 and Gymnasium
already offer, so the window closes the same way however a run ends.

**Rejected:** pause read in the renderer's update with a flag for the human runner (input back in the renderer); a
summary between eval episodes (the `VecEnv` resets first, and the CSV is eval's summary); `pygame.quit()` in each quit
path (a crash left pygame running); `sys.exit()` (it skipped SB3's `on_training_end` and the last flush); the human
runner reading `scenario.hud` itself (two places to read the HUD); polling for keys (a busy CPU core).
