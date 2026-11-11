# Design Decisions

What is settled, why, and the alternatives someone might suggest again, so that nobody has to guess why something is
shaped as it is or put back what was left out on purpose.

The decisions live in `decisions/`, one file per area. This file holds the principles most of them follow and an index
of every decision by title, so a decision can be found by its name. Read the area's file in full before giving a design
opinion in that area.

Written in the present tense. The story of how a decision changed is in `git log`; a reversal that still matters is a
rejected alternative. File paths, test runs and dates of work stay out, and so do open points, which go in
`open-questions.md`. When a decision changes, the entry is rewritten to say what is true now. A new decision goes in
its area's file, and its title in the index below.

---

## Entry template

### Title (date last settled)

**Decision:** what is true.
**Why:** the reason.
**Rejected:** alternatives worth not suggesting again, each with its reason.
**Revisit if:** the condition that would reopen it.

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
- **The core grows only by general capabilities.** The world, the droid and the orbs are not locked, but what is
  added to them is stable and usable by any scenario, and is named for what it does, not for the scenario that
  first needs it. A core addition named for a scenario, or a branch on which scenario is running, is a workaround.
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

## Index

### Digestion: [`decisions/digestion.md`](decisions/digestion.md)

- Digestion lives in the droid, with a folder of digesters (2026-10-12)
- Digesters do not talk to each other (2026-10-12)
- The tier digester is a template with one subclass per scoring mode (2026-10-15)
- The scoring modes pay the same orbs at different times (2026-10-12)
- A new scoring mode is a new scenario (2026-10-11)
- A tier bundle makes its digester from a frozen recipe (2026-10-14)

### Reward and endings: [`decisions/reward-and-endings.md`](decisions/reward-and-endings.md)

- Energy and score are two numbers, and every reward goes through the droid (2026-10-16)
- A truncated episode settles nothing (2026-10-25)
- Digestion pays for orbs, and termination pays for the clock (2026-10-15)
- The reward ladder is one formula on the shared tier orb block (2026-10-11)
- Reward normalisation stays off (2026-10-11)

### Orbs and the world: [`decisions/orbs-and-world.md`](decisions/orbs-and-world.md)

- Two orb categories, and an orb meta that stores only kind and tier (2026-10-12)
- A direct orb is one class per way of paying, and one digester per kind (2026-11-08)
- An orb carries its own cool-down and lifespan (2026-10-18)
- A weight belongs to a sampled field, and a tier chain holds one negative orb (2026-10-12)
- The world runs the orb field, and a scenario states only its size (2026-10-13)
- The step clock lives in the world (2026-10-23)
- A scenario mechanic lives in its strategy when the world's public API is enough (2026-10-11)
- The world's step returns a result, and a scenario's watcher reacts to it (2026-11-05)
- Episode state is first set in `reset` (2026-10-16)

### Locked scenarios: [`decisions/locked-scenarios.md`](decisions/locked-scenarios.md)

- Dense locks with sparse's reward values, so only the scoring mode differs (2026-11-04)
- The ablations kept the step penalty, and sparse's chain break penalty (2026-10-27)
- Delay ends as the family does (2026-11-06)
- Delay's values: a free silence, and a break at half the timeout (2026-11-06)
- Delay's episode length follows from the delay and the grid (2026-11-06)
- Delay's frozen orbs stay on the field and block their cell (2026-11-06, revised 2026-11-11)

### Scenarios and config: [`decisions/scenarios-and-config.md`](decisions/scenarios-and-config.md)

- The scenario package is blocks, families, an entry, a registry and a scenario (2026-11-06)
- Scenario configs live with the scenario, as blocks in the world (2026-10-11)
- An orb kind's config class is shared, and each scenario declares the field (2026-10-11)
- A tier chain has at least two tiers (2026-11-08)
- The negative orb has a switch beside its values (2026-10-29)
- Every scenario in the single tier chain family is a single chain (2026-10-18)
- A chain leaves a cell for the negative orb, on or off (2026-11-07)
- The family is named single tier chain (2026-10-26)
- A locked scenario is a base config and its tunables in front of its entry (2026-11-06)
- A scenario is one entry in its family's catalogue (2026-11-06)
- A locked scenario is what was run, and changing it makes a different scenario (2026-10-29)
- A fork is a full config with a name and a parent (2026-10-29)
- A locked scenario fixes the visibility, and the encoding is a tunable (2026-10-29)

### Observation: [`decisions/observation.md`](decisions/observation.md)

- The kept perceptions observe only what earned its place (2026-10-23)
- An orb's timer is not observed (2026-10-23)
- The HUD and the observation share readers and keep their own lists (2026-10-23)
- The environment hands out a copy of each observation (2026-11-07)
- `vector_markovian_easy` is not rewritten (2026-11-11)
- The composite perceptions are rewritten, as one (2026-11-11)
- An orb's identity is the radix scalar, and tier order is meant to be seen (2026-11-12)
- An orb's identity is looked up in a table the scenario builds (2026-11-12)
- A perception is a format, and the scenario says what the agent knows (2026-11-11)
- An orb slot's columns are orb features the scenario lists (2026-11-11)

### Runners and rendering: [`decisions/runners-and-rendering.md`](decisions/runners-and-rendering.md)

- The algorithm in the runner config is an enum, and a user's runner is added to it (2026-11-06)
- An SB3 algorithm is a spec in the runner table (2026-11-06)
- The policy's name comes from the algorithm (2026-10-13)
- Human play is not an agent runner (2026-10-13)
- The renderer takes plain values, and the HUD is the scenario's list (2026-10-21)
- Runners own the window's input, the renderer draws, and the env closes the window (2026-10-25)
- A run that continues another gets its own id, and records its parent (2026-11-04)
- No config snapshot until the GUI builds one (2026-11-09)

### Project: [`decisions/project.md`](decisions/project.md)

- Legacy is a reference, not a target (2026-10-25)
- The thesis runs live at the `v0.4.0` tag (2026-11-10)
- SYNGrid moves to Python 3.12, through Pop!_OS 24.04 (2026-11-07)
- Legacy stays until the perceptions are rewritten (2026-11-07)

### Testing: [`decisions/testing.md`](decisions/testing.md)

- The regression test runs the replay probe on each locked scenario's base config (2026-11-07)
- The suite is measured by coverage.py alone, and mutmut (2026-11-07)
- mutmut forks from a clean server (2026-11-09)
- Hypothesis checks the world's invariants, and CI runs the tests and ruff (2026-11-07)
- CI fails below a coverage floor of 85% (2026-11-10)
- Tests come first, from the owner's rules (2026-11-07)
