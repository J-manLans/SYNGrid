# Words with two senses, or not settled

- **Block.** It means a `*_conf` model nested in a world config, such as `tier_orb_conf`. It also names the folder
  `scenario/blocks/`, which holds the pieces a `Scenario` is made of: the interfaces, the rule classes, and the
  populations and recipes every family shares. The two are different things.
- **Axis model.** A name used for the tunables model. It also holds the perception, so the name is
  expected to change.
- **Perception.** Today one config value that fixes both the visibility and the encoding. It is decided to name only
  the format, with the visibility in the scenario's view radius; see "A perception is a format, and the scenario
  says what the agent knows".
- **Visibility and encoding.** Today neither is a setting of its own: the one perception value fixes both. It is
  decided that the visibility is the scenario's view radius and the encoding is the format a run picks; see the same
  decision.
- **Pool.** The population file says a scenario "fills the orb pool"; the decisions use pool only for what a sampled
  field is drawn from.
- **Sweep.** The test sweep ends a roadmap step (`procedures/test-sweep.md`); the docs sweep ends a session;
  `sweep.sh` runs the replay probe at each commit in a range; and the scenario sweep is a test.
- **Sparse and dense.** They name the two scaling scenarios. The scoring modes behind them are max-tier and
  threshold.
