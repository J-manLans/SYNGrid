# Building a scenario

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
