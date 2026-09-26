# v1.0.0 TODO

Ordered by dependency, not severity — each item unblocks the ones below it.
Grouped into phases by urgency, numbered continuously across all of them.

Unrelated in-flight work: see `rppo-regression.md`.

## Make the numbers mean something

1. Seed the load path. `ArtifactManager.load_model` never receives the seed, so
   eval and `continue_training` runs are unseeded. Fresh training is fine — SB3's
   constructor seeds the VecEnv. Without this the `info.md` reproducibility claim
   is not true.

2. Fix the reward engine. Two defects, both with regression tests now in place:
   - `OrbFactory._normalize_counts` sorts the count list before distributing the
     remainder, destroying the index-to-orb-type mapping. Weights invert whenever
     the first enabled type outweighs the second.
   - `DigestionEngine._max_reward_bonus` is never cleared on a chain break, so a
     later completion pays out a bonus accumulated on an abandoned chain.
   The `_normalize_counts` tests are `xfail(strict=True)` and will flip to XPASS
   once fixed.

3. Pin every reward-shaping ambiguity to one documented value, then re-run only
   what changed. The timeout penalty is the open one — the suite is currently red
   on it (`assert -1 == -10.0`).

4. Decide `truncated` semantics and record which choice produced the thesis
   figures. Timeouts are reported as terminations, so no value function is ever
   bootstrapped at the horizon.

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

## Adoption, if there is room

11. Widen the Python ceiling. `requires-python = ">=3.10,<3.11"` blocks 3.11+
    for no reason the code depends on.

12. Content-hash run ids, so hand-written `id_tag`s stop being load-bearing.
