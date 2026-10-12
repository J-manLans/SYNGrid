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

## Phase 2: The blocks the spatial builder names

- **What a chain holds when the steps run out.** A threshold scenario's termination is meant to pay the digester's
  `chain_reward`, and a max-tier one must not. How the tier-chain termination tells them apart is not decided. One
  suggestion: each scoring mode says what its chain is owed when it is cut off, and the family keeps one termination
  class. The owner's position is that paying at a timeout is termination's job.

  The answer also decides how cleanly the sandbox can choose a scoring mode. If termination holds the rule, a
  sandbox whose mode comes from config needs a conditional on the mode in its termination, or one termination per
  mode. If each mode says what its chain is owed, the sandbox's termination is the same for every mode. Raised on
  2026-10-12.
  See: "Digestion pays for orbs, and termination pays for the clock" and "A new scoring mode is a new scenario" in
  `design-decisions.md`.

## Phase 4: The builders

- **How `make_digester` gets the values a digester needs.** `OrbBundle.make_digester` takes no arguments, and a tier
  digester needs `max_tier` and `chain_break_penalty` from the config. `_make_tier_digester` takes none and raises.
  Two options have come up. A `functools.partial` works, but two partials do not compare equal, which is what
  already keeps two scenarios from comparing equal through `build_world`. A small frozen dataclass with a
  `__call__`, holding the digester class and its values, compares by value; it would also be where a sandbox
  builder puts the class it chose from config. Raised on 2026-10-12.
  See: "The tier digester is a template with one subclass per scoring mode" in `design-decisions.md`; the roadmap's
  phase 4.

## Phase 5: The Gymnasium side

- **Where the renderer's settings come from.** The old `RendererConf` has no counterpart in the new models. A preview
  in a GUI would need them too.
  See: the roadmap's phase 5.

## Phase 7: The observation pass

- **Which global features an observation carries, and whether the HUD and the observation share readers.**
  See: the roadmap's phase 7.
- **Where an orb's dense identity lives, and whether the radix formula becomes a plain sort on kind and tier.**
  See: the roadmap's phase 7; "Two orb categories, and an orb meta that stores only kind and tier" in
  `design-decisions.md`.
- **The shape of `obs_conf`.** It is two levels of nesting around one field, `perception`. If a perception is split
  into what the droid sees and how it is encoded, the two halves may belong to different sides.
  See: "Scenario configs live with the scenario, as blocks in the world" in `design-decisions.md`; "A perception is
  two things" in `codebase-ideas.md`.
- **How "the perceptions that make sense for a scenario" is written.** Either each locked scenario lists the
  perceptions it allows, or a perception is split into two settings and the scenario fixes how much the droid sees.
  Spatial allows one perception today, so there is nothing to choose between until it has a second.
  See: "A locked scenario offers the perceptions that make sense for it" in `design-decisions.md`.
- **Whether the pixel observation gets a limited view, and which kind.** `GridPixel` is a docstring with no code. A
  crop of the window loses the droid's position; the full grid with the rest blacked out keeps it.
  See: "A perception is two things" in `codebase-ideas.md`.
- **Whether non-tier synergy orbs despawn like negative orbs,** or stay on the field.
  See: "Two orb categories, and an orb meta that stores only kind and tier" in `design-decisions.md`.

## Phase 8: The rest of the tier-chain family

- **The negative orb under delay.** It is meant to freeze and return with the field. How is not worked out.
  See: "A weight belongs to a sampled field, and a tier chain holds one negative orb" in `design-decisions.md`.
- **`reward_multiplier`.** Legacy threshold scoring applied it to each orb's reward. It was not carried over and not
  discussed to a decision.

## Phase 9: Logging and HUD

- **The form of the HUD.** A small vocabulary of elements the renderer can draw, or a drawer per scenario.
  See: the roadmap's phase 9.

## Phase 10: Locking a scenario

All of these are to settle with the first scenario that is locked.
See: "A locked scenario loads a small model, and its builder makes the full config" in `design-decisions.md`.

- **The common type of a config class.** `ScenarioConf` is named by the `TypeVar` bound in `registry.py`,
  `ScenarioEntry.build`, `build_scenario` and the return type of `load_experiment_configs`. A small model is not a
  `ScenarioConf`.
- **How the full config is reached from outside the builder.** It exists only inside the builder and the `partial`.
  `ConfigManager.save_snapshot` copies the YAML file, so a locked scenario's snapshot would hold only what the run
  chose. A GUI that shows the fixed values read-only needs the same thing.
- **Where a family's shared fixed values are written,** so that four builders do not each state them.
- **Where a rule that spans fields is checked.** A check on the full model is raised in the builder and names a block
  and not a field.
- **Whether a locked axis has a range,** such as an upper bound on the grid.
- **What the small model holds besides the axis, and what it is called.** It holds the perception. The negative orb
  would be a yes or no, since its `reward` and `cool_down` are fixed values.

## Not tied to a phase

- **Whether a `ScoringMode` enum survives** as a router for the sandbox scenario. It is written as a reservation
  inside an entry marked settled. The sandbox is not on the roadmap.
  See: "A new scoring mode is a new scenario" in `design-decisions.md`.
- **What a GUI needs from the models.** A GUI is where the project is heading, and nothing is written for it.
  See: "A GUI on top of the registry" in `codebase-ideas.md`.
