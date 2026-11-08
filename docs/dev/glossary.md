# Glossary

What the words mean, in the code and in the other files in this folder. A term is defined once here, and a new term
is added when it is coined. Reasons stay out (`design-decisions.md`), and so does the state of the code (the
roadmap's table).

Terms are grouped by area. The last section lists the words that are used in two senses or are not settled.

---

## Scenarios

- **Scenario:** one registered benchmark task, and the `Scenario` object that describes it: a name, a tag, the pieces
  the Gymnasium side needs and a way to build its world. It holds nothing that changes during an episode.
- **Goal scenario:** a scenario with an objective to reach and a deadline to miss.
- **Continuous scenario:** a scenario where reaching nothing ends the episode. None is written.
- **Family:** the scenarios that share a folder, a config hierarchy and an entry type. The single tier chain family is spatial,
  delay, scaling sparse and scaling dense. The families are in `scenario/families/`, grouped under `goal/` and,
  when one is written, `continuous/`.
- **Single tier chain:** what every scenario in the family of that name is: one tier chain, all of it laid out on the
  field at once and left there until it is eaten. So its tier orbs have no lifespan and no cool-down. A scenario whose
  tier orbs expire or return would not belong to this family. The family's folder is `single_tier_chain/`; the
  mechanic keeps the name tier chain.
- **Variant:** a scenario in a family that needs a config field the family lacks, and so has models of its own.
  Delay is the only one.
- **Sandbox:** the planned continuous scenario where every value can be set and orb kinds can be added. It is not on
  the roadmap.
- **Spatial:** the whole chain is on the grid and the droid sees a 3x3 window around itself. Its axis is the grid
  size.
- **Scaling sparse, scaling dense:** the same world with max-tier scoring (sparse) and threshold scoring (dense).
  Their axis is the chain length and the grid size.
- **Delay:** eating a tier orb freezes the other tier orbs for `delay_conf.steps` steps, after which they return in
  the same cells. The negative orb is not frozen. It tests whether an agent can bridge a gap it cannot see the end
  of. Its axis is the delay.
- **Silence:** the steps between eating a tier orb and the other tier orbs thawing, while nothing in the chain can
  be eaten.
- **Spare steps:** the steps an episode has beyond the silences its chain needs: in delay, `max_steps` minus
  `(max_tier - 1) * delay`. They pay for the walking and the mistakes.
- **Axis:** the one thing that varies between runs of a scenario and makes it harder or easier.
- **Scenario tag:** where a run sits on its scenario's axis, as text for the run id, such as `5x5_steps100`.
- **Locked:** a scenario whose values other than its tunables (its axis, `max_steps`, the perception and the
  negative orb's switch) are fixed in its base config and cannot be set from a run's YAML. A scenario is locked once
  runs learn in the same general direction as legacy.
- **Full model:** a config model that describes a whole world, such as `TierScenarioConf`.
- **Tunables:** what a run of a locked scenario may set, held in the scenario's own config model. Older commit
  messages call this the small model, or a run's choices.
- **Base config:** a locked scenario's full config as it was run, a read-only YAML in the package. A run's tunables
  are merged onto it.
- **Fork:** a copy of a locked scenario's base config with any value changed, saved under a name of its own with the
  scenario it came from, its parent. It is built by its parent's entry.

## Building a scenario

- **Registry:** `SCENARIOS` in `scenario/registry.py`, every family's catalogue collected. It gives a scenario name its
  entry.
- **Entry:** what a scenario name stands for: its full model, the perceptions it allows, its lock if it has one, and
  what its family composes it from. One object per scenario, a subclass of `ScenarioEntry` (`scenario/entry.py`).
- **Entry type:** a family's subclass of `ScenarioEntry`, such as `SingleTierChainEntry`. It adds the fields the
  family's scenarios differ by and composes everything they share.
- **Catalogue:** a family's entries, in `scenarios.py` in its folder, each with what the scenario tests above it.
- **Lock:** a locked scenario's tunables and base config, held by its entry. The entry merges a run's tunables onto
  the base config.
- **Block:** see the last section; the word has two senses.
- **Rules:** a frozen description of one part of a scenario's behaviour: `ObservationRules`,
  `TerminationRules`.
- **Strategy:** a family's own implementation of an interface from `scenario/blocks/`, kept in the family's folder,
  such as `SingleTierChainTermination`.
- **Step result:** what one tick of the world reports, `StepResult`: the reward, and the meta of the orb eaten that
  step or `None`. `GridWorld.perform_droid_action` returns it.
- **Watcher:** what a scenario does after the world's step. It is handed the world and the step result in `react`,
  holds episode state, and is made once per environment by the scenario's `make_watcher`. `NoWatcher`, which never
  acts, is the default; delay's is `DelayWatcher`.
- **Freeze, frozen:** a delay watcher taking an orb off the field while keeping its cell, so it returns there. A
  frozen orb is not on the field, so it is neither seen nor eaten.
- **Blocking:** an orb property that keeps the droid out of the orb's cell, the way the grid's edge keeps it on the
  grid. It costs nothing of itself. Freezing an orb turns it on.
- **Bite:** the droid moving into a frozen orb. The move is blocked, the orb stays frozen, and the bite costs a
  penalty of its own, which belongs to the frozen state and not to blocking.
- **Population:** what makes one orb kind's orbs for a world. It decides which orbs exist, once per world. Putting
  them on the field is spawning, which the world does. The populations are shared, in `scenario/blocks/populations/`.
- **Count:** how many orbs a population makes, known before any world exists. In a fixed field the counts of the
  bundles add up to the field size, the observation's slots and its sort limit, so no orbs are made only to count
  them. In a sampled field they add up to the pool, and the field size is a config value.
- **Bundle:** an `OrbBundle`: one orb kind's population and a way to make its digester.
- **Recipe:** a frozen object a scenario holds that makes something with episode state when it is called, once per
  world. It holds the values the thing is made from and compares by value. A tier bundle holds a
  `TierDigesterRecipe`, which makes a tier digester from the scoring mode's class and the two values every mode
  takes. A scenario holds a `WorldRecipe` as `build_world`, which makes a world from the world config, the bundles
  and the field's size. The shared recipes are in `scenario/blocks/recipes/`; a recipe for one scenario's strategy
  sits with it, as `DelayWatcherRecipe` does in delay's `watcher.py`.
- **Brick:** one orb kind as a unit: its bundle in the scenario layer and its `*_orb_conf` block in the config.
- **Reader:** a module-level function that is handed the world and returns a number. The HUD, the metrics and the
  observation's global features are each a description plus a reader, and one reader can serve all three. The
  readers every scenario can use are in `scenario/blocks/readers.py`; a family's own sit in its `builders.py`.
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
- **Terminal reward:** what termination says the ending itself costs or pays, `EpisodeOutcome.terminal_reward`. The
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
- **Fixed reward orb:** a direct orb that pays the same reward every time, `FixedRewardOrb`. The negative and the
  positive orb are both one, told apart by their kind.
- **Negative orb, positive orb:** a fixed reward orb of `DirectKind.NEGATIVE` or `DirectKind.POSITIVE`. The positive
  orb is written but no scenario holds one yet.
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
- **Break:** an eaten tier orb that is any other tier. It leaves no chain, whatever its tier.
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
- **SB3 spec:** an `SB3Spec` in `RUNNER`: one SB3 algorithm as SYNGrid runs it, its class, device, extra
  hyperparameters and frame stack. Calling it makes the SB3 runner.
- **Run id:** the name of a run's outputs, made of the perception, the seed, the scenario tag, the runner's optional
  tag and the algorithm.
- **Snapshot:** a copy of the scenario's YAML saved under the run id.
- **Human play:** a person plays a scenario by hand through `HumanRunner`. It trains nothing and records nothing,
  and shows a summary of the episode when it ends.
- **Run controls:** the keys an agent run with a human renderer reads between two steps: Space pauses, and Escape or
  the close button quits. Eval and training share them.

## The rebuild

- **Strain:** one of the two copies of the code. The working strain is `src/syn_grid/`; the legacy strain is the
  frozen copy in `src/syn_grid/legacy/`.
- **Legacy:** the stable code from before the rebuild. It is a reference and not a target.
- **Reference:** the branch `refactor-of-scenario-refactor`, read for how things were done and not built on.
- **Skeleton:** the intended structure of the scenario layer, with no logic, that the files in `src/syn_grid/scenario/`
  started from. It was kept in a `reference/` folder, removed on 2026-10-23.
- **Reproduction package:** `reproduction_package/`, the configs and results of the thesis runs.
- **Phase:** a step of `scenario-rebuild-roadmap.md`.
- **Replay probe:** `scripts/replay/probe.py`. It plays the tape through a scenario and records what comes back.
  `diff.py` compares two recordings, and `sweep.sh` runs the probe at each commit in a range.
- **Runner probe:** `scripts/runner_probe.py`. It trains each algorithm briefly with a seed, runs its eval loop, and
  saves and loads it through `train()` and `eval()` (the round trip), and prints digests of the weights, the eval
  actions and the CSV rows. It is the check for the runners, which the replay probe does not reach.
- **Tape:** the fixed, seeded action sequence the replay probe plays, read by one cursor across episodes.
- **Trace:** one recording of the replay probe: every observation, reward, end flag and episode length. A trace
  recorded by hand is named by the commit it was recorded at.
- **Baseline:** a locked scenario's reference trace, recorded from its base config, which the regression test checks
  against. It is committed in `tests/regression/baselines/`, named after the scenario, and its `.json` records the
  commit it was recorded at.

---

## Words with two senses, or not settled

- **Block.** It means a `*_conf` model nested in a world config, such as `tier_orb_conf`. It also names the folder
  `scenario/blocks/`, which holds the pieces a `Scenario` is made of: the interfaces, the rule classes, and the
  populations and recipes every family shares. The two are different things.
- **Axis model.** A name used for the tunables model. It also holds the perception, so the name is
  expected to change.
- **Perception.** One config value that fixes both the visibility and the encoding. Whether it is split into two
  settings is in `open-questions.md`.
- **Pool.** The population file says a scenario "fills the orb pool"; the decisions use pool only for what a sampled
  field is drawn from.
- **Sparse and dense.** They name the two scaling scenarios. The scoring modes behind them are max-tier and
  threshold.
