# Decisions: Scenarios and config

How the scenario package is laid out, how a scenario's config is shaped, and the single tier chain family's rules.

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
mirroring `OrbBundle`. A scenario's entry gives it its config class (see "A scenario is one entry in its family's
catalogue").

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

## The sandbox is a fixed field, its tier orbs whole chains (2026-11-13)

**Built by:** phase 10, the sandbox step.

**Decision:** The sandbox's field is fixed: every orb is stated and present, as in a tier chain. Each orb kind is set
by a count. Tier orbs are set by a max tier and a number of chains, and each chain is one orb of every tier up to the
max, so every chain on the field is whole: two chains at max tier 4 are two orbs each of tiers 1 to 4.

**Why:** A count is what the user means and sees on the grid, and every population already takes one. Legacy's
weights only arrived at counts by another road: each kind's weight became a ratio, the ratios were scaled to a pool
size and rounded, with a ranking step and a minimum pool size, and that many orbs were built once. Nothing was drawn
at random per spawn. Whole chains keep every chain completable.

**Rejected:**

- Weights per kind, as legacy had: an indirect way to a count, with rounding rules of its own.
- A count per tier: it allows uneven chains, such as three tier 1 orbs and one tier 3.

**Revisit if:** a scenario needs the mix to vary within an episode, each returning orb drawn from a pool. That is a
sampled field (see "A weight belongs to a sampled field, and a tier chain holds one negative orb" in `orbs.md`).

## In the sandbox every orb despawns and returns (2026-11-13)

**Built by:** phase 10, the sandbox step.

**Decision:** By default every orb in the sandbox despawns and returns, tier orbs included. Its lifespan is
`_reachable_life_span(rows, cols)`, the longest walk between two cells, as the negative orb's is. Once it has
despawned or been eaten, it returns after a cool-down, so the field refills and chains repeat. Both are values of
each kind's block, so a user can change them, as every value in the sandbox can be.

**Why:** The sandbox is continuous: it has no objective and ends only on the clock, so a field that emptied would
leave the rest of the episode with nothing to do. The lifespan and the cool-down are the orb's own values, set by the
population that makes it (see "An orb carries its own cool-down and lifespan" in `orbs.md`), so the sandbox's
populations pass them where the single tier chain's pass `None`. A lifespan the droid can always cover keeps an orb
reachable from anywhere before it goes.

**Rejected:**

- Tier orbs that stay until eaten and never return, as in the single tier chain: that is the family's guarantee, a
  whole chain laid out at once, and a sandbox built on it would run dry.

**Revisit if:** an orb kind has to stay on the field to work, as an effect orb might.

## The family is named single tier chain (2026-10-26)

**Decision:** The family's names say what it guarantees: `scenario/families/goal/single_tier_chain/`,
`SingleTierChainEntry`, `SINGLE_TIER_CHAIN_SCENARIOS`, `SingleTierChainTermination`, the scenario names
`GOAL_SINGLE_TIER_CHAIN_*` and `goal_single_tier_chain_*.yaml`. The mechanic's names describe the orbs and stay:
`digesters/tier_chain/`, `TierChainPopulation`, the HUD label `current tier chain`.

The scenario name is also the runner's output folder, so runs from before the rename are under `goal_tier_chain_*`.

**Why:** The guarantee and the mechanic together name the family, and each half can vary: `multi_tier_chain`,
`single_color_chain`.

**Rejected:** `tier_chain` (the mechanic's name too); `single_chain` (it leaves out which chain).

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
