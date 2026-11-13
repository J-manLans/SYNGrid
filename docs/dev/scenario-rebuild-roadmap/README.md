# Scenario Rebuild Roadmap

What is left of the rebuild, in order, and the state of each part: written, a stub or legacy, and whether it has run.

Nothing done stays here: when a step is done it is removed, and what was done and run goes in its commit message.
When every step of a phase is done, its file and its line under "Phases" go.
Reasons go in `design-decisions.md`, undecided points in `open-questions.md`. No dates, except where a fact needs
one to be read correctly.

This README is read every session: the **Next** line, the table, and the list of phases. Each phase is a file of its
own. A session reads the file of the phase **Next** names, and another phase's file only when the work reaches it or
the owner asks.

**Next:** fine-tuning the class diagrams (`class-diagrams.md`), then the observation pass (phase 7), starting with what
it settles first.

**Tests from now on:** every piece of new code ships with its tests, and code and tests are reviewed together.

---

## Where things stand

*Has it run* is one of: **written** (never run), **run** (executed, by a script, a human or the environment) or
**learned** (an agent trained on it and learned).

| Part | State | Has it run |
|---|---|---|
| Spatial: entry, base config, run YAML | written | learned |
| Scaling sparse: entry, base config, run YAML | written | learned |
| Scaling dense: entry, base config, run YAML | written | learned |
| The lock: entries, tunables, merge, perception check | written | learned, through every locked scenario |
| Delay: entry, base config, run YAML, `DelayWatcher`, `max_steps` from the delay | written; locked, with the delay, the negative orb's switch and the perception as its tunables | learned; a break penalty of −0.5 not yet run at delays of 30 and 70 |
| `StepResult`, `Watcher`, `NoWatcher` | written | run |
| Max-tier digester | written | learned |
| Step-wise digester | written | written |
| Threshold digester | written | run; never trained |
| Negative orb: a `FixedRewardOrb` of the negative kind, `NegDigester`, population, bundle | written | learned |
| Positive orb: `DirectKind.POSITIVE`, `PosDigester`, the renderer's branch | written; no population, config block or bundle, so no scenario holds one | not run outside the unit tests |
| `SingleTierChainTermination` | written; every scenario in the family uses it | learned |
| Droid, engine, `GridWorld`, orbs, timer | written | learned |
| Environment | written | learned; `check_env` passes |
| `vector_markovian`, `vector_fog_of_war` | written | learned |
| The other five perceptions | legacy, in `legacy/` | — |
| Renderer | written | run, in human mode |
| `HumanRunner` | written | run |
| SB3 runner: `SB3Runner`, and an `SB3Spec` each for PPO, FSPPO and RPPO in `RUNNER` | written | learned with PPO and RPPO; FSPPO only run |
| `BaseAgentRunner`, the extension point for users' runners | written | run; video recording not run since `_wrap_episode_stats` came in |
| CSV logger | written | run; never on a full training run |
| `runners/agent_runners/utils/extractors.py` | legacy: imported by nothing, kept as phase 7b's starting point | — |
| `scenario/families/continuous/` | empty package | — |
| Replay scripts, `scripts/replay/` | written | run |
| Class diagrams, `scripts/diagrams/add_protocol_edges.py` | written | run |
| Runner probe, `scripts/runner_probe.py` | written | run |
| Tests | `tests/unit/` and `tests/integration/`, kept to as few as cover the behaviour. No test file on purpose: the observation and render code, rebuilt in phases 7 and 8; interfaces and data holders with no logic of their own; the entry points and glue the probes run; `extractors.py` | run: passes; every file swept with coverage and mutmut |
| Regression test, `tests/regression/`, and the four baselines | written | run: 4 pass |

---

## Phases

In order.

- [Before phase 7: The class diagrams](class-diagrams.md): Mermaid class diagrams of the rebuilt strain, grouped as the
  owner thinks of the code, to redraw in draw.io.
- [Phase 7: The observation pass](phase-7-observation.md): an observation the scenario describes, and the new orb meta.
- [Phase 8a: The renderer rework](phase-8a-renderer.md): a renderer split by reason to change, and a fog overlay that
  shows what the agent sees.
- [Phase 8b: The modular HUD](phase-8b-hud.md): a HUD stacked from drawn pieces and drawn by element kind, so a scenario
  can list any number of elements in the same style. It replaces the single `hud.png` and the matching by label.
- [Phase 9: Logging](phase-9-logging.md): the CSV logging checked on real runs, and a plot made from a run's CSV.
- [Phase 10: Close out](phase-10-close-out.md): fresh docs for people, and a naming pass once legacy is gone.

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world. See "Effect orbs get their magic from the engine",
  `codebase-ideas/effect-orbs-magic-from-engine.md`.
- Sampled fields and the weighted orb blocks.
