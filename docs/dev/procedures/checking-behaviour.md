# Checking behaviour

How a change is checked against the stable version: the replay probe, the regression test and the runner
fingerprint. When to read it is in `AGENTS.md`, "Verify before asserting": before a change under `src/` meant to
leave behaviour alone, and before any change under `src/syn_grid/runners/`.

What stays out: the test sweep, which is in `test-sweep.md`, and comparing with legacy, which is in `legacy.md`.

- The replay probe (`scripts/replay/probe.py`) records a locked scenario from its base config, so a run's tunables do
  not move it; `--run-yaml` records it as a run sees it. The regression test checks each locked scenario against its
  baseline in `tests/regression/baselines/`. Comparisons are exact, so read the size of a difference before calling
  it behavioural.
- **A change meant to leave behaviour alone is checked against the stable version.** For a locked scenario that is
  the regression test, which plain pytest runs against the baseline. A scenario that is not locked, or a run YAML
  (`--run-yaml`), has no baseline, so it is probed before and after and compared with `scripts/replay/diff.py`. Say
  whether the traces matched and what the tape does not reach: rendering, logging, the runners, and any branch its
  random walk does not hit.
- **A change to the runners is fingerprinted before and after** with `scripts/runner_probe.py`, which trains each
  algorithm briefly with a seed, saves and loads it through `train()` and `eval()`, and digests the weights, the eval
  actions and the CSV rows. The replay probe does not reach the runners, and neither probe reaches the run controls
  or video recording.
