# Glossary

What the words used in the rebuild mean, in the code and in the other files in this folder. A term is defined once
here; the reasons behind it are in `design-decisions.md`.

Terms are grouped by area. The last section lists the words that are used in two senses or are not settled.

---

## Scenarios

- **Scenario:** one registered benchmark task, and the `Scenario` object that describes it: a name, a tag, the pieces
  the Gymnasium side needs and a way to build its world. It holds nothing that changes during an episode.
- **Goal scenario:** a scenario with an objective to reach and a deadline to miss.
- **Continuous scenario:** a scenario where reaching nothing ends the episode. None is written.
- **Family:** the scenarios that share a folder, a config hierarchy and a helper. The tier chain family is spatial,
  delay, scaling sparse and scaling dense. The families are in `scenario/families/`, grouped under `goal/` and,
  when one is written, `continuous/`.
- **Single chain:** what every scenario in the tier chain family is: one chain, all of it laid out on the field at
  once and left there until it is eaten. So its tier orbs have no lifespan and no cool-down. A scenario whose tier
  orbs expire or return would not belong to this family.
- **Variant:** a scenario in a family that needs a config field the family lacks, and so has models of its own.
  Delay is the only one.
- **Sandbox:** the planned continuous scenario where every value can be set and orb kinds can be added. It is not on
  the roadmap.
- **Spatial:** the whole chain is on the grid and the droid sees a 3x3 window around itself. Its axis is the grid
  size.
- **Scaling sparse, scaling dense:** the same world with max-tier scoring (sparse) and threshold scoring (dense).
  Their axis is the chain length and the grid size.
- **Delay:** eating an orb makes every other orb vanish from the observation for `delay` steps, after which they
  return in the same cells. It tests long-term credit assignment. Its axis is the delay.
- **Axis:** the one thing that varies between runs of a scenario and makes it harder or easier.
- **Scenario tag:** where a run sits on its scenario's axis, as text for the run id, such as `5x5`.
- **Locked:** a scenario whose values other than its axis are fixed in its builder and cannot be set from a YAML. A
  scenario is locked after it has reproduced its legacy results. None is locked.
- **Full model:** a config model that describes a whole world, such as `TierScenarioConf`.
- **Small model:** a locked scenario's own config model, holding only what a run may choose.

## Building a scenario

- **Registry:** `SCENARIOS` in `scenario/registry.py`. It gives a scenario name its entry.
- **Entry:** a scenario's config class and its builder.
- **Builder:** the function that turns a scenario's config into a `Scenario`. One per registered scenario.
- **Helper:** the function a family's builders share, which composes what the family has in common.
- **Block:** see the last section; the word has two senses.
- **Rules:** a frozen description of one part of a scenario's behaviour: `ObservationRules`,
  `TerminationRules`.
- **Strategy:** a family's own implementation of an interface from `scenario/blocks/`, kept in the family's folder,
  such as `TierChainTermination`.
- **Population:** what makes one orb kind's orbs for a world. It decides which orbs exist, once per world. Putting
  them on the field is spawning, which the world does. The populations are shared, in `scenario/blocks/populations/`.
- **Count:** how many orbs a population makes, known before any world exists. In a fixed field the counts of the
  bundles add up to the field size, the observation's slots and its sort limit, so no orbs are made only to count
  them. In a sampled field they add up to the pool, and the field size is a config value.
- **Bundle:** an `OrbBundle`: one orb kind's population and a way to make its digester.
- **Recipe:** a frozen object a scenario holds that makes something with episode state when it is called, once per
  world. It holds the values the thing is made from and compares by value. There are two. A tier bundle holds a
  `TierDigesterRecipe`, which makes a tier digester from the scoring mode's class and the two values every mode
  takes. A scenario holds a `WorldRecipe` as `build_world`, which makes a world from the world config, the bundles
  and the field's size. Recipes are in `scenario/blocks/recipes/`.
- **Brick:** one orb kind as a unit: its bundle in the scenario layer and its `*_orb_conf` block in the config.
- **Reader:** a module-level function that is handed the world and returns a number. The HUD, the metrics and the
  observation's global features are each a description plus a reader, and one reader can serve all three. The
  readers every scenario can use are in `scenario/blocks/readers.py`; a family's own sit beside its builder.
- **`build_world`:** the scenario's field that returns a new world on each call. Everything that changes during an
  episode is made there.

## The world

- **World:** the `GridWorld`: the grid, the droid, the orbs and the step clock.
- **Droid:** what the agent moves. It eats orbs and owns digestion.
- **Field:** the orbs that are on the grid.
- **Fixed field:** every orb is stated and present, as in a tier chain.
- **Sampled field:** orbs are drawn at random from a pool, and each kind's weight sets the mix. None is written.
- **Energy:** what the droid has left. It starts at `max_energy`, stays between 0 and `max_energy`, and the episode
  ends when it reaches 0.
- **Score:** how the episode went: the episode's return, the sum of every reward the environment has handed back,
  termination's included. The droid keeps it beside energy, since every reward goes through the droid. It starts at
  0 and has no bounds.
- **Settle:** the droid taking in what its digestion has left undigested when the episode is terminated, in `on_termination`.
- **Ending reward:** what termination says the ending itself costs or pays, `EpisodeOutcome.ending_reward`. The
  environment hands it to the droid's `on_termination`. In a goal scenario it is the timeout penalty or 0.
- **Reward:** what one step pays. A negative reward is a penalty.
- **Penalty:** a config value that must be 0 or negative. Every field whose name ends in `_penalty` is checked.
- **Step clock:** the count of steps in an episode, up to `max_steps`.
- **Timeout:** the step clock running out. It ends the episode and costs `timeout_penalty`.
- **Terminated, truncated:** Gymnasium's two ways for an episode to end. Terminated means the task is over, reached
  or failed. Truncated means the clock stopped an episode that had no task to fail. A goal scenario never truncates:
  its deadline is part of the task, so running out of steps sets `terminated`. A continuous scenario is where
  `truncated` is used. `EpisodeOutcome.truncated` defaults to false, so a goal strategy does not name it.

## Orbs

- **Orb:** something on the grid the droid can eat.
- **Eat, consume:** the same thing. The code says `consume`, the documents mostly say eat.
- **Kind:** what an orb is: a member of `DirectKind` or `SynergyKind`. Digestion routes on it, and each kind has its
  own block in the config.
- **Category:** direct or synergy. It is not stored; it follows from which enum the kind belongs to.
- **Direct orb:** pays a reward and nothing else, on the step it is eaten. The negative orb is one.
- **Synergy orb:** leaves something behind that changes a later outcome. The tier orb is one.
- **Chain orb, effect orb:** the two groups inside synergy. A chain orb's payoff depends on which orbs of its kind
  came before. An effect orb changes something other than itself. They are folders, not kinds, and no effect orb is
  written.
- **Tier:** a tier orb's place in the chain, from 1 up to `max_tier`. Every other orb has no tier.
- **Identity:** the number the agent sees for an orb. Orbs of the same kind and tier share it.
- **Lifespan:** how many steps an orb stays on the field before it despawns, or `None` for an orb that stays until
  it is eaten. It belongs to the orb.
- **Despawn:** an orb leaving the field without being eaten.
- **Cool-down:** how many steps an orb is off the field after it is eaten or despawns, or `None` for an orb that
  does not return in the episode. The next episode's reset places every orb again.
- **Expired, ready:** the two questions an orb answers about its timer, `has_expired()` and `is_ready()`. An orb
  has expired when it is on the field and its lifespan has run out, and is ready when it is off the field and its
  cool-down has run out. A timer set to `None` never completes, so such an orb is never expired or ready.

## Digestion

- **Digestion:** turning an eaten orb into a reward.
- **Digester:** the rules for one orb kind. It keeps its own episode state and knows no other digester.
- **Engine:** the router that hands an eaten orb to the digester of its kind.
- **Chain:** the tiers eaten in order so far. It starts at tier 1 and grows one tier at a time.
- **Progress:** an eaten tier orb that is the next tier and not the last.
- **Completion:** an eaten tier orb that is the next tier and the last. The chain is over.
- **Break:** an eaten tier orb that is any other tier. A tier 1 starts a new chain; any other tier leaves none.
- **Chain worth:** the summed reward of the orbs in the running chain, `chain_worth`, kept in every mode. Whether
  it has been paid depends on the mode.
- **Undigested:** what a digester has gathered and not paid, and still pays if the episode ends now. Every
  digester states it through `undigested()`, and the droid settles the sum when the episode ends. Max-tier holds an
  unpaid chain and returns 0, since an unfinished chain is lost. Only threshold has anything undigested today.
- **Scoring mode:** when a tier orb's worth is paid. An orb is worth the same in every mode.
- **Max-tier:** nothing is paid until completion, which pays the whole chain.
- **Step-wise:** each orb is paid as it is eaten.
- **Threshold:** nothing is paid on progress; a break or a completion pays what the chain had gathered.
- **Reward ladder:** what each tier is worth: `base_reward * tier ** growth_factor`.

## Observation

- **Observation:** what the agent is given each step.
- **Perception:** the class that turns the world into an observation, and the config value that names it, such as
  `vector_fog_of_war`.
- **Visibility:** how much of the world a perception shows: all of it, or a window around the droid. It is a way of
  talking about perceptions and not a setting.
- **Encoding:** the form the observation takes: a flat vector, a composite of named parts, a grid of cells, or
  pixels. It is also not a setting.
- **Markovian:** a perception that shows the whole world, so one observation is enough to act on.
- **Fog of war:** a perception that shows only the orbs within one cell of the droid.
- **POMDP:** a task where one observation is not enough, so the agent needs memory. A perception with a limited view
  makes one.
- **Slot:** the place for one orb in an observation.
- **Global feature:** one value in the observation that is not about an orb, such as energy or steps left.
- **HUD:** what the renderer shows beside the grid. A scenario lists its elements.
- **Metric:** one named number a scenario reports at the end of an episode.

## Running

- **Runner:** what trains or evaluates an agent on a scenario.
- **Run id:** the name of a run's outputs, made of the perception, the seed, the scenario tag, the runner's optional
  tag and the algorithm.
- **Snapshot:** a copy of the scenario's YAML saved under the run id.

## The rebuild

- **Strain:** one of the two copies of the code. The working strain is `src/syn_grid/`; the legacy strain is the
  frozen copy in `src/syn_grid/legacy/`.
- **Legacy:** the stable code from before the rebuild. It is a reference and not a target.
- **Reference:** the `reference/` folder, and the branch `refactor-of-scenario-refactor` it was copied from.
- **Skeleton:** `reference/skeleton/scenario/`: the intended structure of the scenario layer, with no logic.
- **Reproduction package:** `reproduction_package/`, the configs and results of the thesis runs.
- **Phase:** a step of `scenario-rebuild-roadmap.md`.

---

## Words with two senses, or not settled

- **Block.** It means a `*_conf` model nested in a world config, such as `tier_orb_conf`. It also names the folder
  `scenario/blocks/`, which holds the pieces a `Scenario` is made of: the interfaces, the rule classes, and the
  populations and recipes every family shares. The two are different things.
- **Axis model.** A name used for the small model. The small model also holds the perception, so the name is
  expected to change.
- **Perception.** One config value that fixes both the visibility and the encoding. Whether it is split into two
  settings is in `open-questions.md`.
- **Pool.** The population file says a scenario "fills the orb pool"; the decisions use pool only for what a sampled
  field is drawn from.
- **Sparse and dense.** They name the two scaling scenarios. The scoring modes behind them are max-tier and
  threshold.
