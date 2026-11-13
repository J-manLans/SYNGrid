# Open Questions

What is not decided yet, grouped by the roadmap phase that needs the answer.

An answered question leaves: the answer becomes an entry in its area's file in `decisions/`, with its title in the
`design-decisions.md` index, and the question is removed. A
question that is half answered stays with what is left of it. Keep each entry to what is known and the options; a
long argument goes in `codebase-ideas/` or is settled.

---

## Entry template

- **The question.** What is known, and the options if any have come up. Raised on (date).
  See: where it is discussed.

---

## Phase 6: Rebuild, verify and lock the scenarios

- **`reward_multiplier`.** Legacy threshold scoring scaled and rounded each orb's reward with it. Not carried over and
  not decided. If it returns, it is a setting of the orb's reward and not of the mode.
- **How a scenario adds what only it shows,** such as delay's countdown on the HUD. Delay's watcher is a field of
  its entry; the HUD is left for when the HUD
  gets its other elements. Raised on 2026-10-21.
- **Whether a layout that walls off the chain is part of the task or a flaw of the spawn.** Orbs are placed at random
  with no check that a chain can be completed, and an orb the droid is not ready for works as a wall, since stepping
  on it breaks the chain. The owner played 10 episodes at tier 7 on 5x5, and 4 could not be solved because the next
  orb was walled off. That is a small sample, and 6x6 has not been measured. It also bears on what a run's ceiling
  means in the phase's "done when", since a completed chain's pay is not reachable on every layout. Whether legacy
  placed orbs the same way was not checked. Raised on 2026-11-02.
- **Whether `base_reward` has a limit.** `TierOrbConf` accepts any, so a tier chain can be worth nothing or cost.
  Raised on 2026-11-09, writing the config tests.
- **Whether the runner's counts must be at least 1.** `timesteps`, `iterations` and `num_eval_episodes` accept 0.
  `timesteps: 0` trains nothing, and the run still saves a model and finishes normally. Raised on 2026-11-09.

### Locking a scenario

To settle with the first scenario that is locked, which is spatial.
See: "A locked scenario is a base config and its tunables in front of its entry" and "A fork is a full config with
a name and a parent" in `design-decisions.md`.

- **Where a family's shared fixed values are written,** so that four builders do not each state them.
- **Where a rule that spans fields is checked.** A check on the full model is raised as the entry validates it and names
  a block, not a field the user can see.
- **Whether a locked axis has a range,** such as an upper bound on the grid.
- **Where the user folder for forks is, and how the code finds it.** `user_scenarios/` at the repo root, or
  `output/user_scenarios/`, which keeps one ignored folder but is lost when `output/` is cleared, although forks are
  read by later runs and not produced by them. Either way it is found the way `output/` is. Raised on 2026-10-29.

## Phase 7: The observation pass

- **Which global features an observation carries.** The two kept perceptions carry none. Open for the rewritten ones.
  See: "The kept perceptions observe only what earned its place" in `design-decisions.md`.
- **Whether chain progress is observed where tier orbs return.** In a single chain the orbs left on the field show
  how far the agent got, so dropping chain progress cost nothing. Where tier orbs return, the field no longer says
  it. Chain progress is the agent's own history, not a fact of the world: an LSTM can learn to keep a count, a
  stateless agent cannot, so leaving it out would make the task close to unsolvable for one architecture instead of
  measuring credit assignment. It also matters at truncation, where what a running chain holds is paid only through
  the bootstrapped value. It would go in `ObservationRules.global_features`, per scenario. Raised on 2026-10-23.
  See: "A truncated episode settles nothing" and "An orb's timer is not observed" in `design-decisions.md`.
- **What the observation shows for an orb that never expires,** if a rewritten perception brings the timer back. A
  tier orb's timer reads `None` on the field. A 0, a -1 or a separate flag. Raised on 2026-10-18.
- **The shape of `obs_conf`.** It is two levels of nesting around one field, `perception`, which now names only a
  format; the view radius and the feature lists are the scenario's.
  See: "A perception is a format, and the scenario says what the agent knows" in `design-decisions.md`.
- **Where the identity table lives.** The scenario builds it; whether it is a field of `Scenario` or of
  `ObservationRules` is not said. `world.max_identity`, the identity's bound, moves with it. Needed by 7a's identity
  step. Raised on 2026-11-13.
  See: "An orb's identity is looked up in a table the scenario builds" in `design-decisions.md`.
- **How the perception reads the identity in 7a, before orb features exist.** The decision has the identity orb
  feature read the table, and 7a adds nothing new. Either the base perception reads the table itself in 7a and the
  orb feature takes over later, or orb features come in with the identity step. Raised on 2026-11-13.
- **Which step brings orb features in.** That a perception is a format and a slot's columns are orb features is
  decided; when the lists are built is not. If 7a½'s frozen mark is an orb feature, the lists exist before it or the
  mark is built as a one-off. Raised on 2026-11-11.
- **Whether the composite is a thin layer over the vector,** building the same parts and handing them out as a
  `Dict` in place of joining them. To decide at the code. Raised on 2026-11-11.
- **Whether the pixel observation gets a lantern, and which kind.** A crop of the window loses the droid's
  position; the full grid with the rest blacked out keeps it.
- **Whether non-tier synergy orbs despawn like negative orbs,** or stay on the field.

- **What an orb kind's bundle carries.** An `OrbBundle` holds a population and a way to make a digester. A kind also
  brings the metrics that read its digester, a tag suffix, HUD elements and observation features, which a family's
  entry adds by hand today: the negative orb's `if` block in `SingleTierChainEntry._compose` adds the bundle, extends
  the tag and adds the `negatives_eaten` metric. Three sorts: the population and the digester must exist together;
  the count and the metrics follow from the kind being present; what the HUD shows and what the agent observes are
  the scenario's choice, and a kind should not make it. Options: the bundle carries its metrics, as a `metrics` field,
  with their readers moving out of a family's `builders.py` to live with the kind, which ends at one module per kind;
  or it carries what it carries today, and a second family using the same kind repeats the lines. Putting the HUD or
  the observation in the bundle decides 7's and 8b's forms early. Raised on 2026-10-12.

### Delay's frozen orbs

Built between 7a and 7b, see the roadmap.

See: "Delay's frozen orbs stay on the field and block their cell" in `design-decisions.md`. Raised on
2026-11-06.

- **How the droid's move learns that a cell is blocked.** Today the move knows only the grid's edges.
- **How an orb holds the frozen state.** A state of its own (on the field, cooling down, frozen) in place of
  `is_active`, or a flag beside it. Two flags allow pairs that mean nothing, such as frozen and off the field.
  Blocking is a property apart from it either way.
- **How the observation marks a frozen orb.** Talked through on 2026-11-11, not decided. The lean: a frozen flag, 0
  or 1, as an orb feature listed only by scenarios in which orbs can freeze, so spatial and scaling do not carry it.
  A frozen tier-3 orb stays a tier-3 orb, so its identity is unchanged; a 0 or 1 is the plainest input there is and
  keeps the state out of the identity's magnitude; as an orb feature it reaches the vector, the composite and the
  grid at once. Against it: one flag per state allows pairs that mean nothing, such as frozen and burning, so a
  second state would call for a single state feature. The other option raised: a state digit at the bottom of the
  radix identity, which keeps the slot narrower but makes a frozen orb read as another kind.

## Phase 8b: The modular HUD

- **What the renderer is handed about the HUD.** The idea is a frozen `HudSpec` (a label, a kind and a maximum that
  is `None` for a stat) made from the HUD tuple without the readers, so the renderer never holds a reader, and the
  environment passes only the numbers each frame. Open: whether it lands before 8b, what a spec carries beyond the
  three fields, and what the renderer does with a spec it has no place for. Raised on 2026-10-22.
- **Where a bar sits.** A stack grows one way, so a bar across it lives in a fixed cap or stretches with the stack. Or
  a row piece per element kind, a stat row and a bar row, of one width with the same connector edges. Raised on
  2026-10-23.
- **What the window does when the HUD and the grid differ in height.** The lean is that the taller sets it.

## Phase 9: The GUI

- **Whether the GUI starts a run, or writes the config a run reads.** Version 1 picks and sets a scenario; how the
  run that follows is started is not said. Raised on 2026-11-13.
- **How a run points at the checkpoint it loads, and where a continued run records its parent.** Waits for the GUI,
  which picks a checkpoint from a list. Until then a checkpoint is found by the run id, and a curriculum stage renames
  its parent's files by hand. Raised on 2026-11-04.
  See: "A run that continues another gets its own id" in `design-decisions.md`.

  What has to fit, found on 2026-11-04: the observation's shape and the algorithm, with the policy it brings. Bounds
  that differ, as a new grid gives, already pass: the loaded `VecNormalize` keeps the space it was saved with and
  checks shapes only, and accepting that on purpose is a choice to write down then. The `VecNormalize` statistics
  saved beside the checkpoint come with it. A curriculum over the grid fits today. Over tiers in a continuous
  scenario it fits, since the slot count is `max_active_orbs`, if identities stay put when the pool grows. Over tiers
  in a single chain it does not, since the slot count follows the orbs; it needs a fixed slot count.

  Continuing gets its tests when this is settled, not before: curriculum runs come after the GUI and a few new
  scenarios. The rules to test then, as talked through on 2026-11-09: the continued model trains on the run's env;
  it has the saved weights (already tested) and the saved normalization statistics, and goes on updating them; its
  step count carries on, so the next checkpoint is named with the total. The statistics are the silent one: a fresh
  `VecNormalize` crashes nothing and only learns worse. Also open: which seed a continued run uses. `load_model`
  prints that the checkpoint's seed is authoritative but sets nothing, so whether a stage takes its parent's seed
  or its own is undecided.

## Phase 10: The sandbox

- **Whether a broken chain ends a continuous episode.** In the single tier chain family a break ends it. A continuous
  scenario has no objective and truncates on the clock, so a break that ended it would be an objective of its own.
  Raised on 2026-11-13.
- **Whether a `ScoringMode` enum survives** as a router for the sandbox, so its scoring mode can be set.
- **Whether a held chain keeps the droid alive.** In threshold scoring a held chain is worth something not yet paid,
  and a droid that runs out of energy while holding one could have it cashed into energy and go on. If built, the
  cash-out happens in the world after the droid acts and eats, and termination reads energy as usual, so a payout
  never undoes an ending. Against it: the droid can already cash out by breaking the chain on purpose while it has
  moves left, which is the long-horizon trade-off the benchmark tests, and an automatic cash-out hands the agent the
  answer. Where it would earn its place: a highly stochastic scenario with a continuous threshold chain beside a
  goal, where one payout can take the droid to 0 and only a held chain insures against it. Open if built: whether it
  pays the whole chain or only what keeps the droid alive, what a cash-out counts as, and what stochastic means.
  Raised on 2026-10-16.
  See: "Energy and score are two numbers, and every reward goes through the droid" in `design-decisions.md`.

## Phase 11: Effect orbs

- **How effect orbs get their magic from the engine.** The lean, talked through on 2026-10-27: the combining happens
  in the digestion engine and nowhere else, so digesters still know nothing of each other. Each step is a pipeline in
  a fixed order: tick (an optional `tick()` on each digester that has one), route (the eaten orb to its kind's
  digester, as today), modify (the reward handed to each digester with the modifier capability, in a fixed order),
  and reach the world (an effect that touches the field gets a narrow view to read and ask through). The capabilities
  are small optional protocols beside `OrbDigester`, such as `Tickable` and `RewardModifier`, so a digester with none
  is today's. To settle first: payouts that skip the pipeline, since `undigested()` is settled at the episode's end
  and does not pass modify; and the order of modifiers, which do not commute. Open: whether modify runs before
  routing, where the order is stated, and what the view of the world exposes. The other home raised, on 2026-11-05:
  field effects through a watcher, which is handed the world and the eaten orb's kind every step, where the engine
  lives in the droid and holds no world; a reward paid by a watcher would never reach a held payout. The owner's
  direction: a kind's effects stay with the kind, with explicit channels where something has to communicate, and a
  kind could carry its own watcher in its bundle.
  See: "Digesters do not talk to each other" and "The world's step returns a result, and a scenario's watcher reacts
  to it" in `design-decisions.md`.
- **Which effect orbs make the first set.** The vision names synergies, converters and hazards, and asks that each
  adds a distinct mechanic and its own metric. Raised on 2026-11-13.

## Not tied to a phase

- **Whether the episode logger moves into `runners/`.** It is a Gymnasium wrapper in `gymnasium/utils/`, and the
  runners decide to wrap the environment with it and where its CSV goes. It imports nothing from `runners/` (checked
  on 2026-11-13), and both the agent runners and the human runner use it, which speaks for where it is. The test: it
  belongs with the runners if it changes mostly when run outputs, folders or run ids change. Raised on 2026-11-13.
- **Properties or plain attributes for what readers take from a digester.** Plain public attributes now; the owner
  may return to properties.
- **Whether `on_termination` guards against a second call.** It runs once per episode; a second call would pay the
  owed sum twice, and nothing stops it.
