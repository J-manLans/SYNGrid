# Scenario Rebuild Roadmap

What is left of the rebuild, in order, and the state of each part: written, a stub or legacy, and whether it has run.

Nothing done stays here: when a step is done it is removed, and what was done and run goes in its commit message.
Reasons go in `design-decisions.md`, undecided points in `open-questions.md`. No dates, except where a fact needs
one to be read correctly.

**Next:** rebuild the test suite (phase 6b).

**Tests from now on:** every piece of new code ships with its tests, and code and tests are reviewed together. The
suite is rebuilt in phase 6b, and phase 7 does not start until that is finished.

---

## Where things stand

*Has it run* is one of: **written** (never run), **run** (executed, by a script, a human or the environment) or
**learned** (an agent trained on it and learned).

| Part | State | Has it run |
|---|---|---|
| Spatial: entry, base config, run YAML | written | learned: RPPO and PPO on 5x5, with and without a negative orb, before the lock; RPPO on seed 3 reached about 8.5 of a ceiling of about 9.02, with episodes of about 15 steps; the locked path run by a script, step for step the same as before |
| Scaling sparse: entry, base config, run YAML | written | learned: tier 4, in the ablations, before the lock; checked by the owner at more than one `max_tier`; the locked path run by a script, step for step the same as before |
| Scaling dense: entry, base config, run YAML | written | run, by a script through the environment: random steps, and scripted chains that pay the thesis ceilings; played in human mode; learned: PPO, one seed, on sparse's reward values at tier 7 on 5x5 and on 6x6, and on the thesis values at tier 7 on 6x6 |
| The lock: entries, tunables, merge, perception check | written; entries moved into each family's catalogue, the four traces unchanged | run, by a script; played in human mode; learned: dense, through its tunables and base config |
| Delay: entry, base config, run YAML, `DelayWatcher`, `max_steps` from the delay | written; locked, with the delay, the negative orb's switch and the perception as its tunables | run: through the probe (200 episodes, and 50 with `max_steps` from the delay), `check_env`, and 300 random-action episodes (silences of 30 to 33 steps, no orb back in another cell); trained on the thesis-era termination at a delay of 30, and behaves like the thesis delay-30 run; learned: PPO, seed 3, a break ending the episode, 50 spare steps: at a break penalty of −0.1 at delays of 30, 70, 130, 260 and 390, and at −0.5 at 130, 260 and 390; the family's termination in place of `DelayTermination` matched it step for step in the probe |
| `StepResult`, `Watcher`, `NoWatcher` | written | run, for every scenario through the environment; the three locked traces unchanged |
| Max-tier digester | written | learned |
| Step-wise digester | written | written |
| Threshold digester | written | run, by a script through the environment |
| Negative orb: digester, population, bundle | written | learned; played in human mode in spatial, sparse and dense; `negatives_eaten` read by the owner at the end of a human run, and correct |
| `SingleTierChainTermination` | written; every scenario in the family uses it | learned before the refactor; the three locked traces unchanged after it |
| Droid, engine, `GridWorld`, orbs, timer | written | learned |
| Environment | written | learned; `check_env` passes |
| `vector_markovian`, `vector_fog_of_war` | written | learned |
| The other five perceptions | legacy, in `legacy/` | — |
| Renderer | written | run, in human mode |
| `HumanRunner` | written | run |
| SB3 runner: `SB3Runner`, and an `SB3Spec` each for PPO, FSPPO and RPPO in `RUNNER` | written | learned: PPO and RPPO, before the specs; run: all three by the runner probe, 512 steps through `train()`, saved, loaded and evaluated through `eval()`, the same as before the specs |
| `BaseAgentRunner`, the extension point for users' runners | written | run, through the SB3 runner; a user's own runner built in a test; the run controls played by hand, video recording not run since `_wrap_episode_stats` came in |
| CSV logger | written | run, through the environment, and by the runner probe on 512 training steps on two envs; never on a full training run |
| `runners/agent_runners/utils/extractors.py` | legacy: imported by nothing, kept as phase 7b's starting point | — |
| `scenario/families/continuous/` | empty package | — |
| Replay scripts, `scripts/replay/` | written | run, by hand on spatial: two recordings identical, a step-penalty change caught at step 0; the sweep run on one commit; the locked traces recorded for spatial, scaling sparse, scaling dense and delay; the probe building from the base config: the same digests as the old traces wherever the run YAML held the base config's values, and with `--run-yaml` the same as dense's old trace, which its tier-7 6x6 run YAML made; the sweep run on one commit after the change |
| Runner probe, `scripts/runner_probe.py` | written | run: the same digests on repeated runs, and the same as the scratch version it came from; on Python 3.12 the same digests as on 3.10 |
| Tests | `tests/unit/` and `tests/integration/`, kept to as few as cover the behavior: the world's step and free cells, the watchers, the environment's step, the observation handler's copy, entries and the catalogue, delay's config, the SB3 runner and `BaseAgentRunner` | run: passes; every unit and integration test file worked through with coverage and mutmut |
| Regression test, `tests/regression/`, and the four baselines | written | run: 4 pass; red on a change to a tier orb's worth, to a base config's value and on a missing baseline |

---

## Phase 6b: Rebuild the test suite

**Produces:** a test suite written from scratch against the rebuilt strain, tight and complete, not large.

The tests written while the scenarios were locked are the start of it. Go over what exists, fill the gaps and tidy
the rest. The owner wants tooling that shows untested paths, such as branch coverage, and mutation testing.
The tools are coverage.py on its own and mutmut, to find the gaps, and Hypothesis for the world's invariants; see "The
suite is measured by coverage.py alone, and mutmut" and "Hypothesis checks the world's invariants" in
`design-decisions.md`.

- **Test files for the source files with none,** written test-first: the owner states each file's rules, and the
  tests are reviewed before anything else (see "Writing tests" in `AGENTS.md`). Paths are under `src/syn_grid/`, in
  order:
  - Game rules:
    1. `core/droid/digestion/digesters/direct_digester.py`
    2. `core/droid/digestion/engine.py`
    3. `core/orbs/base_orb.py`, with `synergy/chain/tier_orb.py` and `direct/fixed_reward_orb.py`
    4. `core/utils/timer.py`
    5. `scenario/blocks/populations/tier_chain_population.py` and `neg_population.py`, with `population.py`
    6. `scenario/blocks/recipes/world_recipe.py` and `tier_digester_recipe.py`
  - Config:
    7. `config/config_manager.py`
    8. the validators in `config/models/runner_models.py`, `scenario/common_models.py`, `scenario/orb_models.py`
       and `global_models.py`
  - Plumbing, where the agent proposes the rules for the owner to confirm:
    9. `gymnasium/utils/episode_logging/csv_episode_logger.py`
    10. `runners/agent_runners/sb3/artifact_manager.py`
    11. `runners/agent_runners/sb3/run_controls_callback.py`
    12. `runners/agent_runners/sb3/policy_resolver.py`
    13. `runners/agent_runners/runner_registry.py`

  Left without a test file:
  - `core/orbs/orb_meta.py`: phase 7 moves the orb identity.
  - The observation and render code, rebuilt in phases 7 and 8.
  - Interfaces and data holders with no logic of their own.
  - The entry points and glue the probes run.
  - `extractors.py`.
  - `gymnasium/environment.py`, whose step `tests/integration/test_env_step.py` covers.
- **A scenario sweep in `tests/integration/`:** every registered scenario builds, resets, steps and passes
  `check_env`, as one parametrized test.

**Done when:** the suite passes, covers the rebuilt strain's paths, and each test fails when the behavior it checks is
broken. Phase 7 starts after this. The observation and render packages will be rebuilt, so skip tests for them.

---

## Phase 7: The observation pass

**Produces:** an observation the scenario describes, and the new orb meta.

**Files:** `core/orbs/orb_meta.py`, `scenario/blocks/observation.py`, `gymnasium/observation_space/`, the
observation part of the config.

**Settle first** (see `open-questions.md`, phase 7): which global features an observation carries; whether non-tier
synergy orbs despawn like negative orbs; the shape of `obs_conf`; what a perception that writes an orb's timer shows
for an orb that never expires.

### 7a: Align what is there today

The two perceptions in the `Perception` enum brought in line with the new rules. Nothing new is added.

1. The orb identity: where the dense id lives, and whether the radix formula becomes a plain sort on kind and tier.
2. Global features supplied by the scenario, as `GlobalFeature`s in `ObservationRules.global_features`, each with
   its bound and a reader. The two kept perceptions carry none. Chain progress is needed where tier orbs return.
3. The view radius, a field on `ObservationRules` that the limited-view perception reads in place of its hardcoded
   distance, and that the renderer's fog overlay reads in phase 8a.

**One change at a time.** Each of the three is its own commit, and the regression test is run after each against the
baselines. A difference is explained, and its baselines recorded again, or reverted before the next change starts. The reward must not change.
Check whether anything scales the observation by its upper bound. When 7a is done, train spatial.

**Done when:** the observation differs from the baselines only where a change was meant to, the reward is identical,
and spatial still learns.

### 7a½: Delay's frozen orbs

Delay's next version, as decided in "Delay's frozen orbs stay on the field, block their cell, and bite" in
`design-decisions.md`. It comes after 7a, so the mark lands on the aligned observation and the rebuilt suite, and
before 7b, so the new perceptions are written with the frozen state from the start. The questions are in
`open-questions.md`, under phase 7.

1. An orb state for "on the field, but not edible", and blocking as an orb property of its own.
2. The droid's move stopped by a blocking orb, with no charge of its own; the bite's penalty on a frozen orb.
3. `DelayWatcher` freezing orbs in their cells in place of taking them off the field, and the rule that a released
   orb waits for its cell going.
4. The frozen mark in the two kept perceptions, and the frozen skin in the renderer.
5. Relock delay, as every scenario was locked: train it and compare the curves with today's delay, ablate what is in
   doubt while its base config still changes, then lock it and record its baseline again with `probe.py --baseline`. Anything
   that moves the observation or reward goes in `landscape-changes.md`.

**Done when:** the delay spread at −0.5 is run again on it and learns, and the bite's 0.1 is checked against the
curves. The baselines of spatial and scaling are unchanged.

### 7b: New perceptions, one at a time

Rewritten from their copies in `legacy/` against the new rules, each added alone and checked before the next. The
extractors start from `extractors.py`, which the owner has read, and are reworked from there rather than written from
scratch.

**Settle first:** how an extractor is chosen. The policy name comes from the perception and the algorithm in
`policy_resolver.py`, and the extractor would be a second thing the perception implies.

1. **Composite.** The observation as a dictionary of groups. SB3's `MultiInputPolicy` flattens every key by default,
   so it needs a features extractor of its own. `GroupedMetaExtractor` in `extractors.py` is a first attempt to
   rewrite with this step.
2. **Pixel.** The grid as an image, for a CNN. SB3's default CNN expects larger images, so it likely needs a small
   extractor too. `TinyGridCNN` in the same file trains but is unchecked.

3. **Delete legacy.** Once the perceptions are done: `legacy/`, the `USE_LEGACY` switch and `landscape-changes.md`.
   The owner deletes it. Then the rebuild merges to `main`, with `version` in `pyproject.toml` at 0.5.0 and the tag
   `v0.5.0`.

**Done when:** each perception stays inside its observation space, checked by the last of the world's Hypothesis
invariants, added to `TestInvariants` in `tests/integration/test_env_step.py`; an agent trains on each new one; and
nothing imports `legacy/`.

---

## Phase 8a: The renderer rework

**Produces:** a renderer split by reason to change, and a fog overlay that shows what the agent sees.

**Files:** `rendering/pygame_renderer.py`, `rendering/world_view.py`, `rendering/hud_view.py`, `rendering/assets/`,
`pyproject.toml`.

1. Move the renderer's images and font into `rendering/assets/` with `git mv`: `sprites/`, `tiles/` and `fonts/`.
   `sounds/` stays in `assets/`, since the runner uses it. Add the folder to the package data. A pure move, in its
   own commit.
2. Split `pygame_renderer.py` into a coordinator, `world_view.py` and `hud_view.py`, as pure moves, with no base
   classes. The coordinator keeps the window, the summary and pause screens, the dimming and text helpers, the title
   and `close()`. The world view takes the floor, the orbs and the droid. The HUD view takes the HUD.
3. The world view takes the view radius from phase 7 and dims every cell farther from the droid, with the same
   distance the perception uses. No radius, no overlay.

**Done when:** a human run looks as before apart from the fog, and the observation and the reward are unchanged.

---

## Phase 8b: The modular HUD

**Produces:** a HUD stacked from drawn pieces and drawn by element kind, so a scenario can list any number of
elements in the same style. It replaces the single `hud.png` and the matching by label.

**Files:** `rendering/hud_view.py`, the HUD theme file and the HUD pieces in `rendering/assets/`.

**Settle first** (see `open-questions.md`, phase 8b): what the renderer is handed about the HUD; where a bar sits;
what the window does when the HUD and the grid differ in height.

1. The pieces, drawn by the owner: a top cap, a row piece that tiles with matching connector edges, and a bottom cap,
   all of one width.
2. The theme file describes each piece: its picture and the anchors inside it, measured from the piece's own corner.
3. The HUD view stacks the cap, one row per element in `scenario.hud`, and the bottom cap.
4. Each element is drawn by its kind, so no element is picked by label. The energy number under the energy bar goes,
   and the score is drawn from its own element. The theme file replaces the hand-tuned numbers in the draw helpers.

**Done when:** a scenario that lists one more element gets one more row with no change to the art, the code or the
theme file, and the observation and the reward are unchanged.

---

## Phase 9: Logging

1. Check the CSVs of a real training run and a real evaluation run.
2. Write a small plot helper from scratch that reads a run's CSV (the episode columns `r`, `l`, `t` plus the
   scenario's metrics) and draws what a scenario needs.

**Done when:** a plot is made from the CSV of a real run.

---

## Phase 10: Close out

- Write `docs/architecture/` fresh, for people.
- A naming pass once everything is wired and legacy is gone. The scenario and family names are done, and the
  mechanic's names (`digesters/tier_chain/`, `TierChainPopulation`) stay.

**Done when:** the test suite still passes.

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world. See "Effect orbs get their magic from the engine" in
  `codebase-ideas.md`.
- Sampled fields and the weighted orb blocks.
