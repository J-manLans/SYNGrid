# Scenario Rebuild Roadmap

What is left of the rebuild, in order, and the state of each part: written, a stub or legacy, and whether it has run.

Nothing done stays here: when a step is done it is removed, and what was done and run goes in its commit message.
When every step of a phase is done, its file and its line under "Phases" go.
Reasons go in `design-decisions.md`, undecided points in `open-questions.md`. No dates, except where a fact needs
one to be read correctly.

This README is read every session: the **Next** line, the table, and the list of phases. Each phase is a file of its
own. A session reads the file of the phase **Next** names, and another phase's file only when the work reaches it or
the owner asks.

**Next:** fine-tuning the class diagrams (`class-diagrams.md`), then the observation pass (phase 7), starting with what it settles
first.

**Tests from now on:** every piece of new code ships with its tests, and code and tests are reviewed together.

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
| Negative orb: a `FixedRewardOrb` of the negative kind, `NegDigester`, population, bundle | written | learned; played in human mode in spatial, sparse and dense; `negatives_eaten` read by the owner at the end of a human run, and correct; after it became a fixed reward orb, a random walk of each locked scenario with it on matched the old code |
| Positive orb: `DirectKind.POSITIVE`, `PosDigester`, the renderer's branch | written; no population, config block or bundle, so no scenario holds one | not run outside the unit tests |
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
| Class diagrams, `scripts/diagrams/add_protocol_edges.py` | written | run: by hand, 92 classes and 7 protocol arrows into `output/diagrams/`; without pylint it stops with the install message |
| Runner probe, `scripts/runner_probe.py` | written | run: the same digests on repeated runs, and the same as the scratch version it came from; on Python 3.12 the same digests as on 3.10 |
| Tests | `tests/unit/` and `tests/integration/`, kept to as few as cover the behavior: the world's step and free cells, the watchers, the environment's step, the observation handler's copy, entries and the catalogue, delay's config, the SB3 runner and `BaseAgentRunner`; single tier chain termination, the droid, the tier and direct digesters, the engine, the orb meta, the base orb, the tier orb, the fixed reward orb and the timer; the tier chain and negative populations, the world recipe and the tier digester recipe; the catalogue; the config manager; the scenario config blocks' limits; the CSV episode logger, the artifact manager, the run controls callback, the policy resolver and the runner registry; a sweep of every registered scenario through Gymnasium's checker and a whole episode. Left without a test file on purpose: the observation and render code, rebuilt in phases 7 and 8; interfaces and data holders with no logic of their own; the entry points and glue the probes run; `extractors.py`; `gymnasium/environment.py`, whose step `test_env_step.py` covers | run: passes; every unit and integration test file worked through with coverage and mutmut, and mutmut rerun under `forkserver` over all of them, its silent gaps closed |
| Regression test, `tests/regression/`, and the four baselines | written | run: 4 pass; red on a change to a tier orb's worth, to a base config's value and on a missing baseline |

---

## Phases

In order.

- [Before phase 7: The class diagrams](class-diagrams.md)
- [Phase 7: The observation pass](phase-7-observation.md): an observation the scenario describes, and the new orb meta.
- [Phase 8a: The renderer rework](phase-8a-renderer.md): a renderer split by reason to change, and a fog overlay that shows what the agent sees.
- [Phase 8b: The modular HUD](phase-8b-hud.md): a HUD stacked from drawn pieces and drawn by element kind, so a scenario can list any number of elements in the same style. It replaces the single `hud.png` and the matching by label.
- [Phase 9: Logging](phase-9-logging.md)
- [Phase 10: Close out](phase-10-close-out.md)

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world. See "Effect orbs get their magic from the engine",
  `codebase-ideas/effect-orbs-magic-from-engine.md`.
- Sampled fields and the weighted orb blocks.
