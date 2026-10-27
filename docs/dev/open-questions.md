# Open Questions

What is not decided yet, grouped by the roadmap phase that needs the answer.

An answered question leaves: the answer becomes an entry in `design-decisions.md` and the question is removed. A
question that is half answered stays with what is left of it. Keep each entry to what is known and the options; a
long argument goes in `codebase-ideas.md` or is settled.

---

## Entry template

- **The question.** What is known, and the options if any have come up. Raised on (date).
  See: where it is discussed.

---

## Phase 6: Rebuild, verify and lock the scenarios

- **Where delay's mechanic lives, and how a scenario triggers it.** When the droid eats an orb, every other orb
  disappears from the observation for `delay` steps and returns in the same cells. It has to be the field that
  empties: with orbs in sight and the droid unable to eat, the agent would first have to learn it cannot eat yet.
  Homes that have come up:
  - In the delay scenario's own code, as loops over the orbs on the field that call the orbs' own methods. The
    owner's starting point. The world offers only what they read and a way to hear that an orb was eaten.
  - A number on the world, 0 in every scenario but delay.
  - A freeze each orb carries, as it carries a cool-down and a lifespan, triggered for all of them.

  A hidden orb is not an inactive one: the world brings an inactive orb back in a random cell, and delay needs the
  same cells. How the scenario hears of a consume:
  - The world is handed callbacks per moment through `WorldRecipe`. The world then holds scenario code, but it is the
    only way if something must happen mid-step.
  - The scenario acts after the world's step, as termination does, and asks the world what was eaten. Leaned towards
    on 2026-10-13, if nothing in a delay step depends on the field emptying before the step ends, which was not
    traced in legacy.

  Raised on 2026-10-13. The two parked methods in `GridWorld` (`# TODO(delay)`) move into whichever is chosen.
  See: "The world runs the orb field, and a scenario states only its size" and "A scenario mechanic lives in its
  strategy when the world's public API is enough" in `design-decisions.md`.
- **The negative orb under delay.** It is meant to disappear and return with the field. Open: whether eating the
  negative orb itself empties the field.
- **`reward_multiplier`.** Legacy threshold scoring scaled and rounded each orb's reward with it. Not carried over and
  not decided. If it returns, it is a setting of the orb's reward and not of the mode.
- **How a builder adds what only its scenario shows,** such as delay's countdown on the HUD. Either the builder's
  extras are a parameter of `_single_tier_chain_scenario`, as the tier bundle is, or a `replace` on the finished
  `Scenario`. Settled with the helper's other parameters, since delay also has its own termination. Raised on
  2026-10-21.

### Locking a scenario

To settle with the first scenario that is locked, which is spatial.
See: "A locked scenario loads a small model, and its builder makes the full config" in `design-decisions.md`.

- **The common type of a config class.** `ScenarioConf` is named by the `TypeVar` bound in `registry.py`,
  `ScenarioEntry.build`, `build_scenario` and the return type of `load_experiment_configs`. A small model is not a
  `ScenarioConf`.
- **How the full config is reached from outside the builder.** The world part is `scenario.build_world.world_conf`;
  the observation part is kept nowhere. `ConfigManager.save_snapshot` copies the YAML, so a locked scenario's snapshot
  would hold only what the run chose. A GUI that shows or forks the fixed values needs the same hook.
- **Where a family's shared fixed values are written,** so that four builders do not each state them.
- **Where a rule that spans fields is checked.** A check on the full model is raised in the builder and names a block,
  not a field the user can see.
- **Whether a locked axis has a range,** such as an upper bound on the grid.
- **What spatial's small model holds besides the grid size.** Whether `max_steps` grows with the grid or is fixed,
  whether the negative orb is a yes or no, and what the model is called.

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
- **Where an orb's dense identity lives, and whether the radix formula becomes a plain sort on kind and tier.** The
  world writes the dense id back onto the meta, so the meta cannot be frozen until the id lives elsewhere. Negative
  has to keep sorting before the tiers, or existing observations change.
- **What the observation shows for an orb that never expires,** if a rewritten perception brings the timer back. A
  tier orb's timer reads `None` on the field. A 0, a -1 or a separate flag. Raised on 2026-10-18.
- **The shape of `obs_conf`.** It is two levels of nesting around one field, `perception`. If a perception is split
  into what the droid sees and how it is encoded, the halves may belong to different sides.
  See: "A perception is two things" in `codebase-ideas.md`.
- **How "the perceptions that make sense for a scenario" is written** for the locked scenarios other than spatial.
  Either each lists the perceptions it allows, or a perception is split in two and the scenario fixes how much the
  droid sees.
- **Whether the pixel observation gets a limited view, and which kind.** A crop of the window loses the droid's
  position; the full grid with the rest blacked out keeps it.
- **Whether non-tier synergy orbs despawn like negative orbs,** or stay on the field.

## Phase 8b: The modular HUD

- **What the renderer is handed about the HUD.** The idea is a frozen `HudSpec` (a label, a kind and a maximum that
  is `None` for a stat) made from the HUD tuple without the readers, so the renderer never holds a reader, and the
  environment passes only the numbers each frame. Open: whether it lands before 8b, what a spec carries beyond the
  three fields, and what the renderer does with a spec it has no place for. Raised on 2026-10-22.
- **Where a bar sits.** A stack grows one way, so a bar across it lives in a fixed cap or stretches with the stack. Or
  a row piece per element kind, a stat row and a bar row, of one width with the same connector edges. Raised on
  2026-10-23.
- **What the window does when the HUD and the grid differ in height.** The lean is that the taller sets it.

## Phase 9: Logging

- **What the plot helper's base class looks like:** what it owns (reading the CSV, smoothing, saving) and what a
  subclass states (columns and how to draw them), whether it takes its paths as arguments, and where the thesis plots
  live. Raised on 2026-10-26.

## Not tied to a phase

- **Properties or plain attributes for what readers take from a digester.** Plain public attributes now; the owner
  may return to properties.
- **Whether a `ScoringMode` enum survives** as a router for the sandbox. The sandbox is not on the roadmap.
- **Whether `on_termination` guards against a second call.** It runs once per episode; a second call would pay the
  owed sum twice, and nothing stops it.
- **What a GUI needs from the models.** See "A GUI on top of the registry" in `codebase-ideas.md`.
