# Decisions: Locking and forks

What a locked scenario is, what can be tuned on it, and how a fork of it is made.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## A locked scenario is a base config and its tunables in front of its entry (2026-11-06)

**Decision:** A scenario is composed from its full config. A locked scenario is three things:

- **Its entry:** what no value changes, such as the scoring mode, the termination and the HUD.
- **Its base config:** the full config that was run, as a read-only YAML in the package. Where the thesis ran a
  scenario at several sizes, its tunables hold the smallest: spatial on 5x5, scaling sparse at tier 4, scaling dense
  at tier 6, delay at a delay of 30.
- **Its tunables:** a model, frozen, strict and `extra="forbid"`, holding only what a run may set, under the full
  model's field names and nesting.

A locked run's YAML is validated against its tunables and merged onto the base config, and the result is validated
as the full model and composed by the entry. A YAML that states a fixed value is rejected. Nothing below the registry
knows whether a scenario is locked. A scenario that is not locked, the sandbox and a fork load the full model.

A scenario's entry holds its full model, the perceptions with its visibility, an optional lock and what its family
composes it from: the
tunables and the base config, as `Lock.scenario_tunables` and `Lock.base_config_path`. A tunables model is named after
its scenario, as `SpatialScenarioConf`; scaling sparse and
dense are configured alike and share `ScalingScenarioConf`. The entry does the merge, hands the base config out as
`base_conf()` for a test or a GUI to build from, and checks the perception on every
config it builds, so a fork is held to its parent's visibility too. The tunables model never leaves the entry, so
everything after it takes a `ScenarioConf`. `ConfigManager` reads YAML and knows nothing of locks; a lock is a scenario
idea.

A scenario is locked once its runs learn in the same general direction as legacy.

A run's YAML must state every tunable, so the base config's tunables reach no run. They are what the regression
test records.

**Why:** The values that make a scenario are data, so they live in a file that a person and a GUI can read, with
comments, and one merge serves every locked scenario. An entry that composes from the full config serves a locked run
and a fork alike. The tunables model is what a GUI draws the form from, and the base config is what it shows read-only
and copies into a fork. The fixed values pass the same validation as any YAML's.

**Rejected:**

- The full config made by a function per scenario from constants: the values are data written as code, the
  functions repeat each other, and the fixed values cannot be read without a run's tunables.
- The entry composing from the tunables model: a fork would have nothing that accepts it.
- "Small model" and "a run's choices" as the words, with `choices_class` on the lock: three names for one
  thing, and none says what it is for.
- The merge in `ConfigManager`: it would have to know what a lock is.
- The perceptions on the lock: a fork, which has no lock, would see the whole grid.
- The base config as a Python object in the family: a YAML is easier for a person and a GUI to read.
- JSON for the configs: it loses the comments, and a GUI goes through the models whatever the file format.
- Fixed values as defaults on the models: a default can be overridden, so two runs of one name could differ.
- A narrow class and a tunable subclass per block: no single narrow class suits every scenario.
- Thinning the base models: the family's entry type and `WorldRecipe` would read from two sources.
- One full model for all, with a GUI choosing fields: the per-scenario list would live in the wrong layer, and the
  YAML could still set anything.

## A locked scenario is what was run, and changing it makes a different scenario (2026-10-29)

**Decision:** A scenario's name promises one test, so a locked scenario is locked to what was run. Spatial is tier 3,
seen through the fog-of-war window, with the penalties and orb rewards the ablations kept, fixed in its base config. The
grid size is its dial on the world. The other locked scenarios follow with their own axis: the chain length and grid for
scaling sparse and dense, the delay for delay. Beside the axis, `max_steps` is a tunable of every locked scenario, so
a larger grid can be given the steps it needs, and the scenario tag records it as `_steps100`. `max_energy` is a rule
fixed with the penalties, not a size: with a step penalty of -0.001 energy is mostly spent on boundary hits, so it caps
pointless moves.

Tier 3 is a reference point, not a limit. Spatial at tier 4 is a fork of spatial.

**Why:** A benchmark is useful only if two people who say "I ran spatial" mean the same test. The limit need not be
drawn in the YAML: a fork changes what it likes under a name of its own. `max_steps` does not change what is tested,
only how long the droid has, and the tag keeps two runs that differ in it apart.

**Rejected:** `max_tier` tunable with the tag recording it ("spatial" would name a family, not a test); every value
left in the YAML (the file is both the reference and the thing edited); an entry for each
variant (a change of values needs no code); `max_steps` fixed at 100 (a large grid cannot be searched in it);
`max_steps` worked out from the grid (the thesis sparse runs had 100 steps on both 5x5 and 6x6, which no per-cell rule
gives, and the full model would hold a rule where it holds a number).

## A fork is a full config with a name and a parent (2026-10-29)

**Built by:** nothing on the roadmap yet. What is open is under "Locking a scenario" in `open-questions.md`.

**Decision:** Changing a locked scenario beyond its axis makes a fork: its base config copied, any value edited, and
saved with a name of its own and the scenario it came from. A fork is built by its parent's entry, so it keeps the
parent's scoring mode, termination and HUD, and its perception is held to the parent's visibility. Its runs are filed
under its own name.

A fork is data. Saving one writes a YAML into a user folder outside `src/`, gitignored by default, and the registry
reads that folder beside the families' catalogues. A fork whose parent is not registered is an error that names the
file. Changing a mechanic, such as the scoring mode, the termination or an orb kind, makes a new scenario, in code.

**Why:** A fork lets someone push a model past a scenario without changing what the scenario's name means. Values are
data and mechanics are code, so a user makes a fork without touching the package. A fork of spatial that saw the
whole grid would no longer test memory, and that world is scaling sparse already. A fork is small and quick to
recreate, so ignoring it by default costs little, and a user who wants it in git removes one line.

**Rejected:**

- A fork choosing any perception: it would no longer test what its parent tests.
- A general entry that takes any values: the scoring mode would become config (see "A new scoring mode is a new
  scenario").
- Forks under `src/syn_grid/config/yaml/`: they would read as part of the benchmark, and an installed package is no
  place to write.

## A locked scenario fixes the visibility, and the encoding is a tunable (2026-10-29)

**Built by:** phase 7b, for the composite and pixel encodings a run can pick. The rule is built, with the vector as the
only encoding.

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
