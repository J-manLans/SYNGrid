# The rebuild

- **Strain:** one of the two copies of the code. The working strain is `src/syn_grid/`; the legacy strain is the
  frozen copy in `src/syn_grid/legacy/`.
- **Legacy:** the stable code from before the rebuild. It is a reference and not a target.
- **Skeleton:** the intended structure of the scenario layer, with no logic, that the files in `src/syn_grid/scenario/`
  started from. It was kept in a `reference/` folder, removed on 2026-10-23.
- **Reproduction package:** `reproduction_package/`, the configs and results of the thesis runs.
- **Phase:** a step of the roadmap, one file in `scenario-rebuild-roadmap/`.
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
- **Scenario sweep:** `tests/integration/test_scenario_sweep.py`. Every registered scenario, built from its base
  config and made the way a run makes its env, goes through Gymnasium's checker and plays an episode to its end.
- **Rule in a value:** a rule the code keeps through a value it passes rather than through logic, such as `None` for
  no cool-down, a constant or a fixed count. Coverage counts its line as run and mutmut never changes it into another
  value, so the test sweep reads each file for them.
