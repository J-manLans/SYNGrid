# v1.0.0 TODO

Ordered by dependency, not severity — each item unblocks the ones below it.
Grouped into phases by urgency, numbered continuously across all of them.

Unrelated in-flight work: see `rppo-regression.md`.

## Make the numbers mean something

1. ~~Seed the load path.~~ **Not a defect — closed, premise was wrong.**
   `ArtifactManager.load_model` does not pass a seed, but it does not need to.
   SB3 persists `seed` in the checkpoint, `load()` restores it via
   `__dict__.update(data)`, and `OnPolicyAlgorithm._setup_model` then calls
   `set_random_seed(self.seed)`, which re-seeds torch, numpy, the action space
   and the VecEnv. Verified on 2.9.0 and against the declared 2.7.1 floor: two
   independent unseeded loads of one checkpoint produce identical trajectories
   under a fixed action tape. Passing `seed=` to `load()` is *not* a no-op — it
   overrides the checkpoint's seed and changes the rollout, which would decouple
   eval from the seed the checkpoint was trained with. One real residual, much
   smaller than the original claim: the seed comes from the checkpoint, so
   `agent_conf.seed` is silently ignored for `eval` and `continue_training`.
   Worth a warning, not a behaviour change.

2. Fix the reward engine. Two defects, both with regression tests now in place:
   - `OrbFactory._normalize_counts` sorts the count list before distributing the
     remainder, destroying the index-to-orb-type mapping. Weights invert whenever
     the first enabled type outweighs the second.
   - `DigestionEngine._max_reward_bonus` is never cleared on a chain break, so a
     later completion pays out a bonus accumulated on an abandoned chain.
     **Update:** ruled off the spatial path — it lives on `_step_wise_scoring`
     and the spatial config uses `max_tier_scoring`.
   ~~The `_normalize_counts` tests are `xfail(strict=True)` and will flip to XPASS once fixed.~~

3. ~~Pin every reward-shaping ambiguity to one documented value.~~ **Timeout
   penalty: done.** It is now `DroidConf.timeout_penalty` (default `-1.0`),
   replacing a hardcoded `-1` inside `episode_termination.py`. The old hardcode
   shadowed a parameter, and that parameter was itself being fed
   `chain_break_penalty` — so the timeout and chain-break rewards were pinned at
   a 1:1 ratio and the interesting ratios were unreachable by config. They are
   independent fields now; the spatial scenario runs at
   `chain_break_penalty: -0.01` / `timeout_penalty: -1.0` (100:1). A related
   floor-division bug on the delay-mode path (`timeout_penalty // 2`, which
   collapsed any small negative penalty to `-1.0`) is fixed too. **Remaining
   open:** the value is now tunable but not yet *established* — the sweep in
   `rppo-regression.md` has to be redone one variable at a time, because every
   run recorded before this change is a confounded comparison.

4. ~~Decide `truncated` semantics.~~ **Done: timeouts terminate.** Running out of
   steps is a punishment and sets `terminated`, so no value function is
   bootstrapped at the horizon. Pinned by a test so it is not later "fixed" by
   accident. The open cost of that choice — no bootstrapping at the horizon — is
   now a documented trade-off rather than an oversight. Continuous mode is a
   separate, still-open question and deliberately untouched.

## Make it run

5. `ConfigDict(extra="forbid")` on the config models. Unknown keys are currently
   swallowed by pydantic's default, so a typo in any shipped YAML trains on a
   different config than the file describes.

6. Add `configs.yaml` / `test_configs.yaml` to `[tool.setuptools.package-data]`.
   Only `assets/**/*` is declared, so a non-editable install cannot find its
   default config and dies at startup.

7. Add a `--config` flag so a run can point at a config outside the installed
   package.

## Make it honest

8. Every remaining NOTE/TODO → resolved, or pinned with a tracked issue. A
   documented frozen quirk is fine; an undocumented one is a defect, because a
   reader cannot tell which behaviour produced the published numbers.

9. CI step that loads every shipped config against `FullConf`. This catches dead
   config keys, invalid perception names and typos permanently.

10. CHANGELOG marking 1.0.0 as the frozen reference for the thesis figures.

11. Before deleting `reproduction_package/`, deal with its four inbound
    references, or they become dead links and lost evidence:
    - `README.md:93` links to `reproduction_package/info.md`.
    - `scripts/replay_probe.py:39` documents
      `--config-src reproduction_package/spatial_scenario/5x5/rppo.yaml` as its
      worked "a specific scenario's own config" example; that command stops
      working the moment the directory goes.
    - `docs/dev/rppo-regression.md` used the archived `5x5/rppo.yaml` as the
      *only* record of the one-variable diff behind its central claim. That has
      been transcribed into the doc, but verify it against git history first.
    - `scripts/replay_sweep.sh:8` mentions it in a comment.
    `timeout_penalty` was added retroactively to the seven archived scenario
    YAMLs so they still load; each carries a comment saying the value is not
    proof of what was configured at run time.

## Adoption, if there is room

12. Widen the Python ceiling. `requires-python = ">=3.10,<3.11"` blocks 3.11+
    for no reason the code depends on.

13. Content-hash run ids, so hand-written `id_tag`s stop being load-bearing.
