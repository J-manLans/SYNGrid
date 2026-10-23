# Open Questions

Design questions that have come up during the rebuild and have no answer yet, grouped by the roadmap phase that
needs the answer. `design-decisions.md` records what was chosen; this file is the list of what has not been.

When a question is answered, the answer becomes an entry in `design-decisions.md` and the question is removed from
here. A question that is only half answered stays, with what is left of it.

---

## Entry template

- **The question.** What is known, and what the options are if any have come up. Raised on (date).
  See: where it is discussed.

---

## Phase 1: Digestion

- **Properties or plain attributes for what readers take from a digester.** They are plain public attributes now, and
  the owner may return to properties.
  See: "The tier digester is a template with one subclass per scoring mode" in `design-decisions.md`.

## Phase 3: Core takes pieces, not config

- **Whether the world is handed its orbs or the populations that make them.** `WorldRecipe` is written to join the
  orbs of every bundle and hand the world a list. The reference did the other thing: `GridWorld` took one
  `orb_population` and called `create()` in its constructor, because it had to call `BaseOrb.set_life_span` first,
  a class attribute shared by every world in the process. If the lifespan becomes a value each orb is handed, that
  reason is gone and a list is the simpler choice. If it does not, the world has to act before the orbs exist, and it
  would take the bundles' populations and call each. There is no single population any more, since each bundle has
  its own. Raised on 2026-10-12. Since 2026-10-18 the lifespan is a value each orb is handed, so a list is enough;
  the question is closed at step 6 of phase 3.
  See: "An orb carries its own cool-down and lifespan" and "A tier bundle makes its digester from a frozen recipe"
  in `design-decisions.md`; `reference/syn_grid/core/grid_world.py`.

## Phase 5: The Gymnasium side

- **Whether the droid settles when a continuous episode is truncated.** A goal scenario never truncates, so today
  every ending is a termination and calls `on_termination`. At truncation SB3 bootstraps the value of the remaining future from the terminal
  observation, so paying out what a running chain has gathered there would count it twice: once as reward and once
  in the bootstrapped value. Leaning towards settling on `terminated` only, which is what the name
  `on_termination` already says; a truncation would then need its own path only if it pays something. Raised on 2026-10-15.
  See: `design-decisions.md`, "Energy and score are two numbers, and every reward goes through the droid".

## Phase 7: The observation pass

- **Which global features an observation carries.** The two kept perceptions carry none, after the owner's
  ablation. The question is open for the perceptions rewritten in phase 7.
  See: the roadmap's phase 7; "The kept perceptions observe only what earned its place" and "The HUD and the
  observation share readers and keep their own lists" in `design-decisions.md`.
- **Whether chain progress is observed where tier orbs return.** In the thesis scenarios one chain lies on the board,
  so the orbs still there tell the agent how far it got: tier 1 gone and tier 2 on the field means tier 2 is next.
  That is why dropping chain progress cost nothing. Where tier orbs spawn and despawn, as in a continuous scenario,
  the field no longer says where the agent is in the chain, and the progress has to come from somewhere.
  It is a different kind of value from an orb's timer. A timer is about the world, and an agent can learn it by
  watching. Chain progress is the agent's own history. An LSTM can keep a count in its hidden state, though it has to
  learn to, and probably slowly. A stateless agent cannot carry anything between steps, so without it the next tier
  is a guess from the field when several tiers are present. It would do badly, not nothing. Leaving it out would
  then stop measuring credit assignment and make the task close to unsolvable for one architecture.
  It goes in `ObservationRules.global_features`, decided per scenario, in phase 7 or when a continuous family is
  built. Raised on 2026-10-23.
  See: the roadmap's phase 7; "An orb's timer is not observed" and "The kept perceptions observe only what earned
  its place" in `design-decisions.md`.
- **Where an orb's dense identity lives, and whether the radix formula becomes a plain sort on kind and tier.**
  See: the roadmap's phase 7; "Two orb categories, and an orb meta that stores only kind and tier" in
  `design-decisions.md`.
- **What the observation shows for an orb that never expires,** if a rewritten perception brings the timer back.
  The default is no timer. A tier orb has no lifespan, so its timer's `remaining` is `None` while it is on the field.
  Neither perception in `src` writes the timer; legacy's composite perceptions did, and a rewritten one would need a
  number. A 0, a -1 or a separate flag are the options. Raised on 2026-10-18.
  See: the roadmap's phase 7; "An orb's timer is not observed" and "An orb carries its own cool-down and lifespan"
  in `design-decisions.md`.
- **The shape of `obs_conf`.** It is two levels of nesting around one field, `perception`. If a perception is split
  into what the droid sees and how it is encoded, the two halves may belong to different sides.
  See: "Scenario configs live with the scenario, as blocks in the world" in `design-decisions.md`; "A perception is
  two things" in `codebase-ideas.md`.
- **How "the perceptions that make sense for a scenario" is written.** Either each locked scenario lists the
  perceptions it allows, or a perception is split into two settings and the scenario fixes how much the droid sees.
  Spatial allows one perception today, so there is nothing to choose between until it has a second.
  See: "A locked scenario offers the perceptions that make sense for it" in `design-decisions.md`.
- **Whether the pixel observation gets a limited view, and which kind.** `GridPixel` is a docstring with no code in
  `legacy/`, and is not in `src`. A
  crop of the window loses the droid's position; the full grid with the rest blacked out keeps it.
  See: "A perception is two things" in `codebase-ideas.md`.
- **Whether non-tier synergy orbs despawn like negative orbs,** or stay on the field.
  See: "Two orb categories, and an orb meta that stores only kind and tier" in `design-decisions.md`.

## Phase 7c: The modular HUD

- **What the renderer is handed about the HUD.** Raised on 2026-10-22. The renderer is handed `scenario.hud` when it
  is built, keeps each bar's maximum by label and picks energy, steps left and the chain by label. The idea is a
  frozen `HudSpec` (a label, a kind and a maximum that is `None` for a stat) in `hud.py`, made from the HUD tuple
  without the readers, so the renderer never holds a reader. Each frame the environment would pass only the numbers,
  in the order of the specs, and the HUD's height could be fixed when the renderer is built. Open: whether it lands
  before 7c, what a spec carries beyond the three fields (a block type or a size for the modular HUD), and what the
  renderer does with a spec it has no place to draw.
  See: "The renderer takes plain values, and the HUD is the scenario's list" in `design-decisions.md`.
- **Where a bar sits.** A stack grows in one direction, so a bar across it either lives in a cap at a fixed size or
  stretches with the stack. Or a row piece per element kind, a stat row and a bar row, of one width and with the
  same connector edges, so a bar is a row like any other. Raised on 2026-10-23.
  See: the roadmap's phase 7c.
- **What the window does when the HUD and the grid differ in height.** The roadmap leans towards the taller of the
  two setting it.
  See: the roadmap's phase 7c.

## Phase 8: The rest of the tier-chain family

- **Where delay's mechanic lives, and how a scenario triggers it.** Delay tests long-term credit assignment by
  stretching the time between the moments the droid can act on an orb. When the droid eats an orb, every other orb
  disappears from the observation for `delay` steps, so the droid sees an empty field, and then they return in the
  same cells. The eaten orb is gone. It has to be the field that empties: with the orbs left in sight and the droid
  unable to eat, the agent would first have to learn that it cannot eat yet, which is a second problem on top of the
  one the scenario measures. Nothing for it is written, and the scenario has no hook to plug into. Homes that have
  come up:
  - In the delay scenario's own code, and not on the world. The owner's starting point, and what an entry from
    2026-10-11 says. The reference had the two mechanics as world methods, `deactivate_all_orbs(delay)` and
    `reactivate_all_orbs()`. Only delay uses them, so they would move out of the world's API and be written beside
    the scenario, as loops over the orbs on the field that call the orbs' own methods. The world then offers only
    what they read, the orbs on the field, and a way for the scenario to be told an orb was eaten and a step is
    done.
  - A number on the world, 0 in every scenario but delay.
  - A freeze each orb carries, as it carries a cool-down and a lifespan, with something triggering it for all of
    them. It fits with the first: the scenario's loop would call it, and an orb that counts its own freeze down
    needs no call after the step.

  Whatever is chosen, a hidden orb is not the same as an inactive one. The world brings an inactive orb that is
  ready back in a random cell, and delay needs the same cells. A rule that reacts to an orb being eaten also needs
  to be told, which means a way for a scenario to hear of it. Two ways have come up:
  - The world is handed callbacks and calls them: a tuple of callables per moment, passed through `WorldRecipe`,
    empty for a scenario with nothing to say. The world then holds scenario code. It is the only way if something
    has to happen in the middle of the world's step.
  - The scenario acts after the world's step, as termination does. The environment calls the scenario's rule, which
    asks the world what was eaten. The world is handed nothing and only has to say what was eaten on the last step.
    The observation is built after the step, so orbs hidden then are already gone from what the agent sees. Leaned
    towards on 2026-10-13, on the condition that nothing in a delay step depends on the field emptying before the
    step is over. That was not traced in legacy.

  Raised on 2026-10-13.
  See: "The world runs the orb field, and a scenario states only its size" and "A scenario mechanic lives in its
  strategy when the world's public API is enough" in `design-decisions.md`.
- **The negative orb under delay.** It is meant to disappear and return with the field, which it does if hiding acts
  on every orb on the field. Still open: whether eating the negative orb itself empties the field.
  See: "A weight belongs to a sampled field, and a tier chain holds one negative orb" in `design-decisions.md`.
- **`reward_multiplier`.** Legacy threshold scoring applied it to each orb's reward. It was not carried over and not
  discussed to a decision.

- **How a builder adds what only its scenario shows.** Raised on 2026-10-21 for delay's countdown on the HUD. The
  family's own elements are listed in `_tier_chain_scenario`. Passing a builder's extras in as a parameter, as the
  tier bundle is, keeps the helper the one place a scenario is composed. The other option is a `replace` on the
  finished `Scenario`. Delay also has its own termination, so this is settled together with the helper's other
  parameters.

## Phase 9: Logging

Nothing open yet. The logger reads `scenario.metrics`.

## Phase 10: Locking a scenario

All of these are to settle with the first scenario that is locked.
See: "A locked scenario loads a small model, and its builder makes the full config" in `design-decisions.md`.

- **The common type of a config class.** `ScenarioConf` is named by the `TypeVar` bound in `registry.py`,
  `ScenarioEntry.build`, `build_scenario` and the return type of `load_experiment_configs`. A small model is not a
  `ScenarioConf`.
- **How the full config is reached from outside the builder.** The world part is on the scenario since 2026-10-12,
  as `scenario.build_world.world_conf`. The observation part is not kept anywhere, and nothing uses either yet.
  `ConfigManager.save_snapshot` copies the YAML file, so a locked scenario's snapshot would hold only what the run
  chose. A GUI that shows the fixed values read-only needs the same thing.
- **Where a family's shared fixed values are written,** so that four builders do not each state them.
- **Where a rule that spans fields is checked.** A check on the full model is raised in the builder and names a block
  and not a field.
- **Whether a locked axis has a range,** such as an upper bound on the grid.
- **What the small model holds besides the axis, and what it is called.** It holds the perception. The negative orb
  would be a yes or no, since its `reward` and `cool_down` are fixed values.

## Not tied to a phase

- **What the tier chain family and its scenarios are called.** Every scenario in the family is a single chain, laid
  out whole, so `single_chain` may describe it better than `tier_chain`, which is also the mechanic's name. Left to
  one naming pass at the end of the rebuild. Raised on 2026-10-18.
  See: the roadmap's phase 10; "Every scenario in the tier chain family is a single chain" in `design-decisions.md`.
- **Whether a `ScoringMode` enum survives** as a router for the sandbox scenario. It is written as a reservation
  inside an entry marked settled. The sandbox is not on the roadmap.
  See: "A new scoring mode is a new scenario" in `design-decisions.md`.
- **What a GUI needs from the models.** A GUI is where the project is heading, and nothing is written for it.
  See: "A GUI on top of the registry" in `codebase-ideas.md`.
