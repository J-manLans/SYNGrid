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

### Locking a scenario

To settle with the first scenario that is locked, which is spatial.
See: "A locked scenario is a base config and its tunables in front of its entry" and "A fork is a full config with
a name and a parent" in `design-decisions.md`.

- **What a locked run's snapshot holds.** `ConfigManager.save_snapshot` copies the run's YAML, which for a locked
  scenario is only the tunables. Either the merged full config, or the tunables beside the name of the base
  config they were merged onto. Not to settle now, and not to raise again before the GUI: the snapshot is a quick file
  copy, and the owner expects the GUI scenario builder to need something completely different. Until then, a base
  config edit is committed before training on it, so a run's values can be looked up in git by its date. Noted on
  2026-11-01.
- **Where a family's shared fixed values are written,** so that four builders do not each state them.
- **Where a rule that spans fields is checked.** A check on the full model is raised as the entry validates it and names a block,
  not a field the user can see.
- **Whether a locked axis has a range,** such as an upper bound on the grid.
- **Where the user folder for forks is, and how the code finds it.** `user_scenarios/` at the repo root, or
  `output/user_scenarios/`, which keeps one ignored folder but is lost when `output/` is cleared, although forks are
  read by later runs and not produced by them. Either way it is found the way `output/` is. Raised on 2026-10-29.

## Phase 7: The observation pass

- **Whether a perception returns a new array each step.** Every perception fills the same buffer, so
  `obs_t is obs_{t+1}`. Gymnasium 1.4.0 warns about it, and `check_env` now fails its reset-determinism check on it:
  it copies the first two observations but not the third, which a later unseeded reset overwrites. With every
  observation copied, the seeded resets match. Copying costs an allocation each step; keeping the buffer leaves
  `check_env` failing. Raised on 2026-11-07.
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
- **Whether the pixel observation gets a limited view, and which kind.** A crop of the window loses the droid's
  position; the full grid with the rest blacked out keeps it.
- **Whether non-tier synergy orbs despawn like negative orbs,** or stay on the field.

### Delay's frozen orbs

Built between 7a and 7b, see the roadmap.

See: "Delay's frozen orbs stay on the field, block their cell, and bite" in `design-decisions.md`. Raised on
2026-11-06.

- **Where a penalty for moving into a blocking orb is set.** Blocking costs nothing of itself. Either a penalty is a
  setting of a blocking orb, or a scenario adds it where it needs one, as delay adds the bite to its frozen orbs,
  and the default is nothing. The owner leans towards the scenario adding it, which keeps the bite with the frozen
  state and apart from blocking. The bite starts at around 0.1.
- **How the droid's move learns that a cell is blocked.** Today the move knows only the grid's edges.
- **How an orb holds the frozen state.** A state of its own (on the field, cooling down, frozen) in place of
  `is_active`, or a flag beside it. Two flags allow pairs that mean nothing, such as frozen and off the field.
  Blocking is a property apart from it either way.
- **How the observation marks a frozen orb,** in each perception.

## Phase 8b: The modular HUD

- **What the renderer is handed about the HUD.** The idea is a frozen `HudSpec` (a label, a kind and a maximum that
  is `None` for a stat) made from the HUD tuple without the readers, so the renderer never holds a reader, and the
  environment passes only the numbers each frame. Open: whether it lands before 8b, what a spec carries beyond the
  three fields, and what the renderer does with a spec it has no place for. Raised on 2026-10-22.
- **Where a bar sits.** A stack grows one way, so a bar across it lives in a fixed cap or stretches with the stack. Or
  a row piece per element kind, a stat row and a bar row, of one width with the same connector edges. Raised on
  2026-10-23.
- **What the window does when the HUD and the grid differ in height.** The lean is that the taller sets it.

## Not tied to a phase

- **Properties or plain attributes for what readers take from a digester.** Plain public attributes now; the owner
  may return to properties.
- **Whether a `ScoringMode` enum survives** as a router for the sandbox. The sandbox is not on the roadmap.
- **Whether `on_termination` guards against a second call.** It runs once per episode; a second call would pay the
  owed sum twice, and nothing stops it.
- **How `reproduction_package/` connects to the thesis.** It records the thesis runs and stays in the repo until
  this is decided. Raised on 2026-11-04.
- **What a GUI needs from the models.** See "A GUI on top of the registry" in `codebase-ideas.md`.
- **How a run points at the checkpoint it loads, and where a continued run records its parent.** Waits for the GUI,
  which picks a checkpoint from a list. Until then a checkpoint is found by the run id, and a curriculum stage renames
  its parent's files by hand. See "A run that continues another gets its own id" in `design-decisions.md` and
  "Picking a checkpoint" under "A GUI on top of the registry" in `codebase-ideas.md`. Raised on 2026-11-04.
