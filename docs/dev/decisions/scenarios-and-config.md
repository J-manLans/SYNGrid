# Decisions: Scenarios and config

The scenario package, config blocks, entries, locks and forks.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## The scenario package is blocks, families, an entry, a registry and a scenario (2026-11-06)

**Decision:** `scenario/` holds:

- `blocks/`: what a scenario is made of that no family owns, the interfaces and rule classes as files, and the
  shared implementations in `populations/` and `recipes/`.
- `families/`: the scenarios. A group such as `goal/` holds what its families share; a family folder such as
  `single_tier_chain/` holds its config models, its entry type and how it composes (`builders.py`), its catalogue
  of entries (`scenarios.py`) and its own strategies.
- `entry.py`, the entry every family subclasses; `registry.py`, which collects the catalogues and selects a scenario
  by name; and `scenario.py`, the `Scenario`.

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
      TierWorldConf               + tier_orb_conf, neg_orb_conf
```

Each world model has a scenario model that narrows `world_conf` to it. Each orb kind has its own `*_orb_conf` block,
mirroring `OrbBundle`. A scenario's entry gives it its config class (see "A scenario is one entry in its family's catalogue").

**Why:** The config tree and the scenario tree are one tree, so a builder and its config are found in one folder.

**Rejected:** a separate config tree mirroring the scenario package (two trees drift); goal and tier values on the
droid with the orbs in one pool (the digester and termination read them, and one leaf took four classes); goal and orb
blocks beside `world_conf` (the world acts on them); a `mode: goal` field (the name says it).

## An orb kind's config class is shared, and each scenario declares the field (2026-10-11)

**Decision:** `orb_models.py` holds the class that describes an orb kind wherever it appears. A family adds what
describes its use of the kind, as a subclass. A scenario's world model declares the field, optional or mandatory. The
base `WorldConf` declares no orb fields. A base class holds only fields certain to be shared: `enabled`, `reward` and
`cool_down` for the negative orb; `max_tier`, the reward ladder and `chain_break_penalty` for the tier orb.

**Why:** What an orb is gets written once, and each scenario states its relationship to it in one line.

**Rejected:** `neg_orb_conf` on the base `WorldConf` (every scenario would accept it, and none could make it
mandatory); a class per scenario (they drift); a class kept in its first family until a second needs it (a second is
already expected).

## A tier chain has at least two tiers (2026-11-08)

**Decision:** `TierOrbConf.max_tier` is at least 2, as `Field(ge=2)`. The config is the front door: the tier
population and the tier digester take a plain number and do not check it.

**Why:** A chain of one tier is an orb that pays when it is eaten, with nothing to keep in order. That is a positive
direct orb's job, not a chain's.

**Rejected:** a validator with a message of its own: every simple bound in the config models is a `Field`, and
pydantic's error names `max_tier`. A check in the digester or the population: the config already rejects it.

## The negative orb has a switch beside its values (2026-10-29)

**Decision:** `NegOrbConf` holds `enabled` beside `reward` and `cool_down`, and `neg_orb_conf` is mandatory on the
tier world: an orb that is off is `enabled: false`, never a missing block. A locked scenario's tunables offer only
the switch, at the same path, `neg_orb_conf.enabled`, and the values come from its base config. A fork can change all
three.

**Why:** The tunables and the full model then name the orb by one path, so a locked run's tunables merge onto the
base config with no special case. A GUI shows a toggle whose fields keep their values while it is off.

**Rejected:** a `negative_orb` yes or no in the tunables with no counterpart in the full model (the merge needs a
case of its own to turn it into a block); the standard orb's values as defaults on `NegOrbConf` (a default can be
overridden; they belong in the base config); the block kept optional beside the switch (a missing block and a
disabled one would mean the same).

## Every scenario in the single tier chain family is a single chain (2026-10-18)

**Decision:** The family is the scenarios whose whole chain is laid out at once and left until it is eaten, whatever
`max_tier` is: spatial, scaling sparse, scaling dense and delay. The family states once, in `_tier_bundle`, that its
tier orbs have no cool-down and no lifespan, and no scenario can change it.

**Why:** A value every member shares belongs to the family; an argument nobody varies invites the question why it is
there.

**Rejected:** a `life_span` argument on `_tier_bundle` (it read as if spatial were special); keyword overrides (a
scenario whose tier orbs return is not a single chain and belongs in another family).

**Revisit if:** a scenario is added whose tier orbs expire or return.

## A chain leaves a cell for the negative orb, on or off (2026-11-07)

**Decision:** A single tier chain's grid has a cell for each tier orb, one for the negative orb and one for the
droid, and a config with fewer is rejected. The negative orb's cell is counted whether it is switched on or not, so
a world without it has one free cell at the tightest.

**Why:** A world with an orb and no cell for it hangs on its first reset, waiting for room that never comes. Counting
the negative orb only when it is on would let a locked run's switch, which is a tunable, turn an accepted config into
one that hangs. A field packed to the last cell is not something an agent could solve most of the time, so the spare
cell costs nothing worth having.

**Rejected:** counting the negative orb only when it is enabled (the switch would decide whether a config is valid,
and the tightest accepted grid would have no free cell).

**Revisit if:** the family gets more orbs than the chain and the negative orb, or obstacles that take cells.

## The family is named single tier chain (2026-10-26)

**Decision:** The family's names say what it guarantees: `scenario/families/goal/single_tier_chain/`,
`SingleTierChainEntry`, `SINGLE_TIER_CHAIN_SCENARIOS`, `SingleTierChainTermination`, the scenario names
`GOAL_SINGLE_TIER_CHAIN_*` and `goal_single_tier_chain_*.yaml`. The mechanic's names describe the orbs and stay:
`digesters/tier_chain/`, `TierChainPopulation`, the HUD label `current tier chain`.

The scenario name is also the runner's output folder, so runs from before the rename are under `goal_tier_chain_*`.

**Why:** The guarantee and the mechanic together name the family, and each half can vary: `multi_tier_chain`,
`single_color_chain`.

**Rejected:** `tier_chain` (the mechanic's name too); `single_chain` (it leaves out which chain).

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
dense are configured alike and share `ScalingScenarioConf`. The entry does the merge, and checks the perception on every
config it builds, so a fork is held to its parent's visibility too. The tunables model never leaves the entry, so
everything after it takes a `ScenarioConf`. `ConfigManager` reads YAML and knows nothing of locks; a lock is a scenario
idea.

A scenario is locked once its runs learn in the same general direction as legacy.

A run's YAML must state every tunable, so the base config's tunables reach no run. They are what the regression
test records.

**Why:** The values that make a scenario are data, so they live in a file that a person and a GUI can read, with
comments, and one merge serves every locked scenario. An entry that composes from the full config serves a locked run and a
fork alike. The tunables model is what a GUI draws the form from, and the base config is what it shows read-only and
copies into a fork. The fixed values pass the same validation as any YAML's.

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

## A scenario is one entry in its family's catalogue (2026-11-06)

**Decision:** A scenario is one entry, which states the whole of it: its full config model, the perceptions with its
visibility, its lock, and what its family composes it from. `ScenarioEntry` in `scenario/entry.py` is the base: it
holds what every family shares, does the merge and the checks, and leaves `_compose` to the family. Each family
subclasses it once, with the fields its scenarios differ by: `SingleTierChainEntry` takes a scoring mode, a tag
function and an optional watcher recipe, and composes everything the family shares. A family's catalogue,
`scenarios.py` in its folder, holds its entries, each with what the scenario tests as a comment above it. The
registry collects the catalogues into `SCENARIOS` and looks a name up there; it knows nothing of any family.

A family's `builders.py` is how the family composes: its entry type, its tag functions, its readers and its bundle
helpers. Its catalogue is which scenarios it has. A watcher that needs a value from the config is named by a
function of the config, as `DelayWatcherRecipe.from_conf`.

**Why:** A scenario was split three ways: a builder held its tag and scoring, the registry entry its visibility and
lock, and the base config its values. The builders were nearly copies, and the visibility sat in the registry only
because a fork, which has no lock, must still be held to it. One entry per scenario says what a scenario is in one
place, and the per-scenario functions go. A family's catalogue grows with the family; a registry that held every
family's entries would import every family's internals and grow without bound. The base class lives outside the
registry, since the catalogues build entries and the registry imports the catalogues. The catalogue is apart from the
machinery because they change for different reasons: a scenario added or locked, against how the family composes.

**Rejected:**

- A builder function per scenario: the bodies repeat each other, and the scenario is split from its visibility.
- The family's fields on a generic entry: a scoring mode means nothing to another family.
- A generic entry holding a family record in a builder slot: two objects per scenario where one does.
- Every family's entries in `registry.py`: the registry imports every family and grows with each scenario.
- The entry types in `registry.py`: the catalogues and the registry would import each other.
- The entry types in `scenario.py` or `scenario/blocks/`: `scenario.py` is the built `Scenario`, and the blocks are
  the parts a scenario is made of, not what stands in front of one.
- The catalogue at the bottom of `builders.py`: it mixes which scenarios exist with how they are composed.

**Revisit if:** a family's scenarios differ by more than a few fields, or a second family's entry type repeats the
first's.

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

**Decision:** Changing a locked scenario beyond its axis makes a fork: its base config copied, any value edited, and
saved with a name of its own and the scenario it came from. A fork is built by its parent's entry, so it keeps the
parent's scoring mode, termination and HUD, and its perception is held to the parent's visibility. Its runs are filed
under its own name.

A fork is data. Saving one writes a YAML into a user folder outside `src/`, gitignored by default, and the registry
reads that folder beside the families' catalogues. A fork whose parent is not registered is an error that names the file.
Changing a mechanic, such as the scoring mode, the termination or an orb kind, makes a new scenario, in code.

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
