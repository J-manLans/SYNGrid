# Ideas About the Codebase

Ideas worth returning to that are not problems to solve now. Half-formed is fine.

Decisions stay out: once an idea is decided, it moves to `design-decisions.md` and leaves here, or keeps only what is
still an idea.

---

## Entry template

**Idea:**
**Where it came from:**
**Why it might matter:**

---

## A GUI on top of the registry (2026-10-12)

**Idea:** The owner's picture of the GUI the project is heading for.

- The first step is picking a scenario: one of the registered ones, or the continuous sandbox.
- For a locked scenario only its axis can be edited. Its other values are shown read-only.
- The values are on one side and a preview of the scenario on the other. The preview follows the form, so a larger
  grid for spatial shows as a larger grid.
- In the sandbox every value that is meant to be tunable can be edited, and orbs are added from a list of every
  existing kind.
- Version 1 stops there. A user who finds a good setup in the sandbox turns it into a scenario by hand, in the code.
- Version 2 has a "create scenario" step: choose continuous or goal, define the goal, and save the result as a
  scenario.

**Where it came from:** Talking through whether the locked-scenario models would also serve a GUI, on 2026-10-12.
Nothing in the repo is written for a GUI; the only mention is a TODO in `config/config_manager.py`.

**Why it might matter:** It is a second consumer of the config models, and it asks a few things of them.

What already fits:

- `SCENARIOS` is the list of scenarios, and an entry's config class is what a user may set for that scenario. A form
  can be made from it without a special case for locked scenarios or the sandbox.
- `ConfigManager.load_config` ends in `model_class(**raw)`, so a dict from a form takes the same path as a dict from
  a YAML.
- A `Scenario` holds nothing that changes and `build_world()` returns a new world on each call, so a preview can
  build a world from the form on every edit.
- Each orb kind is a `*_orb_conf` block on a world model. A sandbox world model that declares every kind as optional
  is the list of existing orbs, and adding an orb is switching a block on. It allows one block per kind.

What it asks for:

- An error has to name a field the user can see. See the open points in the design decision.
- A range on a locked axis would serve the form, the validation and the preview at once.
- The explanations of the fields are in class docstrings. A form would want them per field.
- The renderer's settings have no place in the new models yet, and a preview needs them.

Version 2 is the only part that goes against the current design. A scenario is an entry in code, in its family's
catalogue, and a member of `ScenarioName`, and a scenario made in a GUI would have to be data that a general entry
composes from. Nothing is shaped for that now. The blocks (termination as a protocol with a file per strategy,
`OrbBundle`) are what such a composer would choose from.

Turning a sandbox setup into a goal scenario is not a copy of its config: the sandbox is continuous and has its own
world model, so the goal and the termination are decided by hand.

The owner wants to learn pygame_gui and would use it for this GUI. It is built for interactive widgets and themes
them through a JSON file, which is what a form of fields, dropdowns and buttons needs. The game HUD stays in plain
pygame, see "The renderer takes plain values, and the HUD is the scenario's list" in `design-decisions.md`.

**Picking a checkpoint (2026-11-04).** For eval and for continuing a run, such as a curriculum stage, the user picks
a checkpoint from a list. What it is named does not matter, only whether it fits the run being set up. Today a
checkpoint is found by rebuilding the run id from the config, so a continued stage on a new grid means renaming the
parent's files by hand. Found while talking it through:

- What has to fit is the observation's shape and the algorithm, with the policy it brings. Bounds that differ, as a
  new grid gives, already pass: the loaded `VecNormalize` keeps the space it was saved with and checks shapes only.
  Accepting that on purpose is a choice to write down then. The `VecNormalize` statistics saved beside the
  checkpoint come with it.
- A curriculum over the grid fits today. Over tiers in a continuous scenario it fits, since the slot count is
  `max_active_orbs`, if identities stay put when the pool grows: the world renumbers them by sorting, and a kind that
  sorts earlier would shift every number after it. Over tiers in a single chain it does not fit, since the slot count
  follows the orbs; it needs a fixed slot count, as legacy's `tiers` setting gave.
- A list can grey out the checkpoints that do not fit.

See "A run that continues another gets its own id, and records its parent" in `design-decisions.md`.

---

## A bundle that carries everything its orb kind brings (2026-10-12)

**Idea:** An `OrbBundle` holds a population and a way to make a digester. A kind brings more than that to a scenario:
the metrics that read its digester, a tag suffix, HUD elements and observation features. The bundle could carry some
of these, and a family's entry type would collect them from its bundles as it now sums their counts.

**Where it came from:** Adding the negative orb to the single tier chain family. Its `if` block in `SingleTierChainEntry._compose`
adds the bundle, extends the tag and adds the `negatives_eaten` metric: three lines that together mean "add this
brick".

**Why it might matter:**

- The contributions are of three sorts. The population and the digester must exist together: an orb with no digester
  makes the engine raise. The count and the metrics follow from the kind being present, with nothing to decide. What
  the HUD shows and what the agent observes are the scenario's choice, and a kind should not make it.
- Only the first sort is in the bundle, and the count is taken from the population. That is enough while one family
  uses the negative orb, because its `if` block is one readable place.
- The moment to act is a second family using the same kind. Each would then repeat the lines, and they could drift.
  The first candidate is a `metrics` field on the bundle.
- A metric's reader is the same in every family: take the kind's digester from the engine and read a counter. So the
  readers would move out of a family's `builders.py` and live with the kind. Followed to the end, that is one module per
  kind, which groups by kind where `scenario/blocks/populations/` and `scenario/blocks/recipes/` group by role.
- The HUD's form and the observation's features are open, in phases 9 and 7. Putting either in the bundle decides
  them early.

---

## A perception is two things: what the droid sees and how it is encoded (2026-10-12)

**Idea:** Split the perception into two settings. One is the visibility: the whole world, or a window around the
droid. The other is the encoding: a flat vector, a composite of named parts, a grid of cells, or pixels. That a
locked scenario fixes the visibility and the encoding is a tunable is decided (see "A locked scenario fixes the
visibility, and the encoding is a tunable" in `design-decisions.md`); making them two settings is the idea.

**Where it came from:** Deciding which perceptions a locked scenario offers, on 2026-10-12. The fog of war in
`vector_fog_of_war.py` is one filter, which keeps the orbs within one cell of the droid, and nothing ties it to the
vector encoding. The class names already read as a pair: `vector`, `composite`, `composite_grid` and `grid_pixel` on
one side, and `markovian`, `fog_of_war` and `fully_pomdp` on the other.

**Why it might matter:**

- "Spatial offers only the perceptions with a limited view" then needs no list per scenario. With one enum, every new
  perception has to be added by hand to each scenario that allows it.
- The two halves may belong to different sides. The visibility follows from the task, which is the scenario's. The
  encoding follows from the policy and its feature extractor, which is the runner's. This bears on the shape of
  `obs_conf`, which is open.
- The composite perceptions return `Dict` spaces that SB3 flattens into one vector, so they give the model nothing a
  vector does not until a custom extractor is written. Their own notes say so.
- A limited pixel observation can be made in two ways. A crop of the window is the same size on every grid and says
  nothing about where the droid is. The full grid with everything outside the window blacked out keeps the position.
  `vector_fog_of_war` gives the droid its position beside the orbs it sees, so the second is the closer match.
- `composite_grid_markovian` and a pixel observation are not the same thing. The first is a rows x cols x channels
  array where a cell holds facts (an active flag, the orb kind, the tier). The second is a picture the model has to
  read. Both suit a CNN; the first is small and exact, the second large and slower.

---

## A held chain that keeps the droid alive (2026-10-16)

**Idea:** In threshold scoring a held chain is worth something not yet paid. A droid that runs out of energy while
holding one could have the chain cashed into energy and go on, so the chain doubles as a reserve.

If it is ever built, it happens in the world and not at termination: after the droid acts and eats, a droid at 0
asks its digestion to cash out, and termination then reads energy as usual. Seen as a revive at termination it needs
a check for steps left (`terminated and steps_left > 0`), which reads poorly and lets a payout undo an ending. Seen as
a cash-out before termination looks, no clock check is needed: if the clock ran out too, the episode ends on the clock
and nothing is owed.

**Where it came from:** Deciding on 2026-10-16 that the ending's payout goes to the score and not to energy, so a
droid that ran out of energy stays at 0 on its last frame. See "Energy and score are two numbers" in
`design-decisions.md`.

**Why it might matter:** Probably as a reason not to build it. The droid can already cash out without the rule, by
eating a tier orb out of order while it still has moves left: a threshold break pays `chain_break_penalty` plus what
the chain had gathered. Learning when to do that is the long-horizon trade-off the benchmark wants to test, and an
automatic cash-out at 0 would hand the agent the answer. That deliberate cash-out only matters where a break does not
end the episode, in a family where chains repeat. In a goal tier chain a break ends it, and running out of energy pays
the chain without the penalty, so breaking on purpose never helps there.

If it is built anyway, it is a new path for a reward into the droid and changes what threshold teaches, so it would be
a scenario of its own. It fits a continuous family better than a goal one: a cash-out would have to end the chain
without counting as a break or a completion, and a goal episode would then go on with a chain that cannot finish.

**Where it earns its place (2026-10-16):** a highly stochastic goal scenario, with a continuous threshold tier chain
running beside the goal and the cash-out at 0 turned on. Orb rewards are uncertain, so the droid has to take chances,
and one payout can take it to 0 on a single step. A deliberate break cannot save it then, because the droid cannot
know beforehand which orb is the bad one. The only defence is to already carry an undigested chain, so the cash-out is
no longer an answer handed to the agent but what makes insurance a strategy. The agent learns how much chain is worth
carrying against the step costs of building it, and that a large held chain makes risky orbs affordable.

The goal ends the episode, and a break or a completion of the side chain ends nothing, which is a termination of the
scenario's own. Open when it is built:

- Whether a cash-out pays the whole chain, with anything above `max_energy` lost or left undigested, or only what
  keeps the droid alive. Paying only what is needed makes the chain a battery and changes what carrying it is worth.
- What a cash-out counts as. It is not a break or a completion, so it likely needs a counter of its own.
- What "stochastic" means: a reward drawn per orb, or kinds whose values are hidden. That decides whether the agent
  can learn to read the risk at all.

---

## A GUI forks a locked scenario instead of editing it (2026-10-29)

**Idea:** The owner's picture of forking in the GUI. What a fork is, and where it is saved, is decided (see "A fork
is a full config with a name and a parent" in `design-decisions.md`); this is how the screen could go.

- Picking spatial shows its axis as editable: the grid size, the negative orb's switch, and the perception as a
  dropdown of the perceptions with its visibility. The rest of the base config is grayed out, or hidden.
- An unlock button makes every value editable, or visible. The perception stays a dropdown with the same list, and
  the negative orb becomes its switch with its two values under it.
- Cancel throws the edits away, and nothing is written. Save asks for a name and writes the fork.

**Where it came from:** Talking through where to put the limit on what is tunable, by voice on 2026-10-26, and the
owner picturing the screen on 2026-10-29.

**Why it might matter:** Tunable becomes a property of the mode and not of the scenario: locked for runs that can be
compared, unlocked for exploring. Until the GUI exists, a fork is a YAML written by hand.

---

## Effect orbs get their magic from the engine (2026-10-27)

**Idea:** When effect orbs are built, the combining happens in the digestion engine and nowhere else. Digesters keep
not knowing about each other. The engine already sees all of them, so it is the one place that can let one kind's orb
change what another kind's orb pays. This is a lean, not a decision. Effect orbs are not built, and the roadmap lists
them under "Not on this roadmap".

**Where it came from:** The owner remembers deciding this in an earlier talk, of which no record was found. It was
talked through again on 2026-10-27. It answers the "Revisit if" of "Digesters do not talk to each other" in
`design-decisions.md`, and the roadmap's list of what effect orbs need from the engine.

**How it could work.** The engine's step becomes a short pipeline with a fixed order:

1. **Tick.** Once per step, the engine calls an optional `tick()` on every digester that has one. This is how a timed
   effect counts down or wears off.
2. **Route.** The consumed orb goes to the digester of its kind, as today, and that digester returns a reward.
3. **Modify.** The engine hands the reward to each digester that has the optional modifier capability, in a fixed
   order, and each returns the adjusted reward. A "double the next tier payout" orb is an effect digester that holds
   the doubling and applies it here. The tier digester never hears of it.
4. **Reach the world.** An effect that has to touch the field (clear it, freeze it, move an orb) is handed a narrow,
   one-way view of the world, to read from and to ask through. It does not hold the world.

The capabilities are small optional protocols next to `OrbDigester`, such as `Tickable` and `RewardModifier`. The
engine checks which a digester has when it is built. A digester with none of them is the digester of today, so
nothing already written changes. This is the list in the roadmap: a tick, a stage before routing, a way to change
what another kind's orb pays, a way to reach the world.

**Why it fits the rule.** A digester can still be read and tried alone. The only thing that knows about combinations
is the engine, and it knows them as stages in a fixed order, not as digesters calling each other. A step's reward can
be explained from one place by reading the pipeline top to bottom.

**Two things to settle first:**

- **Payouts that skip the pipeline.** `undigested()` is settled by the droid when the episode ends. If a modifier is
  meant to scale a held chain payout too, that payout has to pass through the modify stage as well. Today it does
  not.
- **Order matters.** Two modifiers do not commute: doubling and then subtracting is not subtracting and then
  doubling. The order has to be written down, by scenario or by a fixed rule, and not left to how the list was built.

Open when it is built: whether modify runs before routing (so an effect can change which digester gets the orb) or
only after (so it can only change the reward); where the order of modifiers is stated; and what the narrow view of
the world exposes.

**Field effects through a watcher (2026-11-05).** Step 4 above, an effect that touches the field, may fit a watcher
better than a view of the world handed through the engine: the engine lives inside the droid, which holds no world,
while a watcher is handed the world and the eaten orb's kind every step. A reward effect could also be paid by a
watcher the way termination pays its ending, through the droid and added to the step's reward, but a held payout
(`undigested()`) never reaches the step result, and two homes for reward effects would need one chosen. The owner's
direction: an orb kind's effects stay contained to the kind, with explicit channels where something has to
communicate. A kind could carry its own watcher in its bundle (see "A bundle that carries everything its orb kind
brings"), so its field effect travels with it and the scenario never has to know. See "The world's step returns a
result, and a scenario's watcher reacts to it" in `design-decisions.md`.

---

## Orbs that block without being frozen (2026-11-06)

**Idea:** An orb kind whose orbs block their cell for good: a wall. With blocking a property of its own, a wall is an
orb that blocks and is never eaten, and walking into it costs nothing. A layout of them can shape a scenario, such
as the vision's wall of deadly orbs around a goal orb.

**Where it came from:** The owner and the agent, while deciding that delay's frozen orbs block their cell. See
"Delay's frozen orbs stay on the field, block their cell, and bite" in `design-decisions.md`.

**Why it might matter:** The vision says a scenario's character may come from how its orbs are arranged, and that
nothing can describe a layout today. Blocking cells are the first piece of that. Placing walls on purpose, and not at
random, is the larger part and is not touched.

---

## Clamping energy once per tick (2026-11-08)

**Idea:** The droid clamps energy on each reward as it comes in, so a step that would take energy below 0 is cut at 0
and the orb eaten in the same tick pays into a clean slate. At 0.03 energy and a step penalty of -1, the droid ends
the tick with the orb's payout R and not R - 0.97: the overdraft is forgiven. The score, which has no bounds, is
charged all of it. Clamping once per tick, on the sum of the step's cost and the orb's payout, would charge the
overdraft against the orb, and the droid would live only if the orb paid more than it.

**Where it came from:** Writing the droid's tests on 2026-11-08, after recording that an orb eaten in the draining
step keeps the droid alive; see "Energy and score are two numbers" in `design-decisions.md`.

**Why it might matter:** Probably not, and the owner leans against it: energy and score measure different things, so
keeping them in step is not a goal, and they already part at the cap on purpose. What the forgiven overdraft gives
is a droid alive with a little more energy than it paid for, at most one step's cost each time it eats on the step
that drains it. A wall hit never comes before an eat, since a blocked move leaves the droid where it is and no orb on
the field sits under it, so only the step penalty can be forgiven: at most 0.001 in spatial and the scaling
scenarios, and nothing in delay. Under max-tier and threshold the only positive payouts end the episode anyway.

It would matter with a large step penalty and an orb that pays while the episode goes on, such as step-wise scoring's
progress or an orb that pays on its own. Clamping once per tick also needs someone to close the tick: the world knows
where it ends and the droid does not, so either the world asks the droid to clamp at the end of
`perform_droid_action`, or the droid's two paths stop clamping, which sits close to the rejected general
`receive(reward)` door. It would change when episodes end, so it would go into `landscape-changes.md`.

---

## A positive direct orb (2026-11-08)

**Idea:** A direct orb that pays a positive reward when it is eaten, with no order to keep: the negative orb's
counterpart, with its own digester and config block.

**Built so far:** the orb and its digester: `FixedRewardOrb(DirectKind.POSITIVE, ...)` and `PosDigester`, tested, and
the renderer draws it. What is left waits for a scenario that holds one: a population, a config block that checks the
reward is above 0, and a bundle in that family's builders.

**Where it came from:** Deciding on 2026-11-08 that a tier chain has at least two tiers. A max tier of 1 was the only
way to get an orb that simply pays, and this orb is what replaces it. See "A tier chain has at least two tiers" in
`design-decisions.md`.

**Why it might matter:** It is the plainest reward a scenario can hold, and a building block for a continuous
scenario or a stochastic one, where orbs pay different amounts and the agent has to read the risk.

---

## A guide entity that walks the chain (2026-11-09)

**Idea:** A second body on the grid, a ghost, that shows the agent the tier order and does nothing else. The agent
cannot see the tiers, so the ghost is the only place the order can come from.

The rules as talked through:

- The orbs land in new positions every episode, and their tiers are hidden from the observation. Everything else is
  visible, the ghost included, on every step.
- The ghost walks the shortest path from orb to orb in tier order. It only passes over them and never eats one, so
  it touches no orb, no digester and no score.
- It walks every tier but one, stops on the last tier it walks, and leaves one orb untouched. Which tier is left alone
  is the ghost's one dial: tier 1 or tier 4.
- With tier 1 left alone, the ghost walks 2, 3 and 4. The agent has to find the orb nobody visited and eat it first,
  then follow the ghost's route. An agent that only copies the ghost eats a tier 2 first, which breaks the chain and
  ends the episode, so imitation earns nothing.
- With tier 4 left alone, the ghost walks 1, 2 and 3. Copying gets the agent through three tiers, and the test is
  that it must then finish the chain on the orb the ghost never touched. It is the easier setting.
- The ghost's behaviour is locked, so it is the same in every episode and the dial is the only thing that changes
  what the scenario asks. It always takes the shortest path with a fixed tie rule, never eats, and does the same
  thing when it stops. Whether it then stays or leaves is decided once and not tunable.
- The ghost's trail can be switched on as an option. With it off, the agent has to remember what the ghost did, which
  suits a recurrent policy. With it on, the history is in the observation and a stateless policy can read it. One
  scenario, two questions: can the agent learn the order, and can it remember it. Results must say which setting
  they came from.

Leaving tier 1 untouched is the harder setting: the first eat is the test. Leaving tier 4 untouched is easier,
because following the ghost already gets the agent through three tiers.

**Where it came from:** The owner, by voice on 2026-11-09, after asking what a rival or ghost droid could be besides
a score fight (the obvious design: contact decided by who has the higher score). The idea moved from enemy to
teacher once the tiers were hidden.

**Why it might matter:**

- It asks for something no scenario does today: learning a hidden order by watching it, not by trial and error with
  the end reward as the only teacher. The untouched first orb is what stops it being plain imitation.
- The trail is a vector, since the observation is one. A cell the ghost never visited reads 0. A visited cell reads a
  value that falls with age, down to a floor above 0, so the oldest cell is still readable when the ghost leaves.
  How fast it falls is a dial. Listing only the visited cells, as position and age, keeps the size fixed on a large
  grid. Order numbers in the cells would do the same with less shading.
- Companions, rivals and guides would share the same machinery: a second entity with its own movement rule. The owner
  would file this under an entity family, but one scenario does not show the shape yet. A second or third entity
  scenario is the moment to build it.

**Open points:**

- The ghost needs a fixed rule when a shortest path has several routes, such as rows before columns, or the walk is
  not reproducible and the replay probe cannot hold it.
- Where the ghost starts: the centre, where the droid starts, or somewhere else.
- Whether the ghost stays on the grid after it stops or leaves. It is decided once, and a leaving ghost adds a memory
  test on top.
- How long the ghost walks, and how many steps the agent has after it stops. Locked with the rest of its behaviour.
- What the ghost needs from the engine: a place in the world's step, a place in the observation, and a way to
  leave. A watcher may fit the first, see "Field effects through a watcher" above.
- The word "trace" in the glossary is the replay probe's recording. The ghost's trail needs another name so the two
  do not mix.
- With four orbs a blind guess hits tier 1 one time in four. It is a weak shortcut, since a wrong pick ends the
  episode, but a small agent may still settle for it. Training one early and watching what it does would show.
- What the agent learns is a rule, the unvisited orb first and the rest in the ghost's order. Whether a larger chain
  makes that harder or easier is open.
