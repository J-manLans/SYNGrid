# Scenarios

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
