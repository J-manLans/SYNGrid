# Code Map

Where the rebuilt strain's code lives, and the path a run and a step take through it, so an agent can find its way
without reading every file first. Written for agents.

What stays out: why things are shaped as they are (`design-decisions.md`), what is written, stub or run (the roadmap's
table), traps (`AGENTS.md`), and `legacy/`. A line here names a file and what to look for in it, not what
the code says, so it goes stale only when a file moves or the path changes. Update it in the same change as the move.
When this map and the code disagree, the code is right; fix the map.

---

## The tree

Not kept here: a listing goes stale and costs words every session. Print the rebuilt strain's tree when it is needed:

```bash
find src/syn_grid -path '*/legacy' -prune -o -path '*/assets' -prune -o -type f ! -name '__init__.py' ! -name '*.pyc' -print | sort
```

Outside `src/syn_grid/`: `scripts/replay/` (probe, diff, sweep), `scripts/runner_probe.py` (fingerprints each
algorithm's runner), `scripts/diagrams/add_protocol_edges.py` (draws the class diagrams into `output/diagrams/` with
pyreverse, the whole strain and one per group in its `GROUPS`, with protocol and handed-in arrows; needs the
`diagrams` extra), `scripts/diagrams/layers.py` (draws the strain by layer into `output/diagrams/layers/` from `ast`
alone: an overview of the layers and one diagram per layer, its classes grouped as `SUBGROUPS` lists them),
`scripts/tidy.sh`, `tests/regression/baselines/` (each locked scenario's baseline, which
`tests/regression/test_replay.py` checks through the replay probe), and `tests/conftest.py` (Hypothesis's profiles).

## The layers

- **`core/`** is the simulation and knows no scenario. `GridWorld` runs the orb field and the step clock around a
  droid. `SynergyDroid` moves, eats and keeps energy and score; every reward goes through it. Its private
  `DigestionEngine` routes an eaten orb to the one digester of its kind. An orb is a `BaseOrb` with a meta (kind and
  tier) and a timer.
- **`scenario/`** composes a world from parts and knows no Gymnasium. A `Scenario` is a frozen recipe: observation
  rules, termination rules, HUD, metrics, `build_world` and `make_watcher`. `blocks/` holds the shared parts,
  `families/` each family's configs, entry type and catalogue of entries, `entry.py` the entry every family
  subclasses, and `registry.py` collects the catalogues and maps a `ScenarioName` to its entry.
- **`gymnasium/`** adapts a `Scenario` to the Gymnasium API: `SYNGridEnv`, the perceptions, the CSV logger.
- **`runners/`** drive an environment: SB3 agents for training and eval, and human play.
- **`rendering/`** draws plain values it is handed. **`config/`** holds the Pydantic models and the run YAMLs.

`core/` imports nothing outside `core/`. The scenario reaches the core only by building it.

## Start-up: `python3 -m syn_grid`

1. `__main__.py` picks the strain with `USE_LEGACY`, then calls `app.main`.
2. `app.main` registers `syn_grid-v0` (`gymnasium/utils/env_factory.py`) and loads three configs through
   `ConfigManager`: `global_config.yaml` names the scenario; the scenario's run YAML goes through its entry's
   `full_conf`, which for a locked scenario checks it against the tunables model and merges it onto
   `families/goal/single_tier_chain/base_configs/<name>.yaml`; `runner_config.yaml` picks the algorithm.
3. `build_scenario` (`scenario/registry.py`) looks the name up in `SCENARIOS` and calls its entry's `build`
   (`scenario/entry.py`), which checks the config class and the allowed perception, then calls `_compose`. Every
   entry of the family is a `SingleTierChainEntry` (`families/goal/single_tier_chain/builders.py`), listed in the
   family's catalogue, `scenarios.py`. Its `_compose` assembles the orb bundles, the observation rules, the
   termination, the HUD, the metrics, the watcher and a `WorldRecipe`. The entries differ in their scoring mode, tag
   function and watcher.
4. `human_control` sends it to `HumanRunner`. Otherwise `runner_registry.build_runner` calls the algorithm's entry
   in `RUNNER`, an `SB3Spec` that makes `SB3Runner`, and `app.dispatch` trains or evaluates.
5. The runner makes environments with `env_factory.make(scenario, render_mode)`. `sb3_runner._make_env` wraps
   each in `RecordEpisodeStatistics`, the CSV logger and video as configured, then `DummyVecEnv` and `VecNormalize`,
   and `VecFrameStack` if the spec stacks frames. `policy_resolver` maps the perception name's prefix to the SB3 policy.

## One environment

`SYNGridEnv.__init__` (`gymnasium/environment.py`) calls `scenario.build_world()`, which is the `WorldRecipe`
(`scenario/blocks/recipes/world_recipe.py`). Each call makes new orbs from every bundle's population, one digester
per bundle (`make_digester`, for tier orbs a `TierDigesterRecipe`), a `SynergyDroid` holding those digesters, and a
`GridWorld`. The world remaps orb identities to dense ids on construction. The env then builds the
`ObservationHandler`, which picks the perception class from `PERCEPTIONS` by the scenario's `Perception`.

**`reset`:** `GridWorld.reset(np_random)` restarts the clock, resets the droid (centre cell, full energy, score 0,
every digester's episode state) and every orb, then spawns up to `max_active_orbs`. The watcher is reset. The
perception is reset and writes the first observation.

**`step(action)`:**

1. `GridWorld.perform_droid_action`: one off the clock; `droid.take_step` moves and returns the step and boundary
   cost; every orb's timer ticks; an orb under the droid is eaten through `droid.consume_orb` → `engine.digest` → the
   digester of its kind, else an expired orb despawns; one ready orb spawns if the field has room. It returns the
   cost plus what digestion paid, already applied to energy and score.
   It returns a `StepResult`: that reward, and the eaten orb's meta or `None`.
2. The scenario's watcher, made once per environment by `scenario.make_watcher`, is handed the world and the result
   in `react`. `NoWatcher` does nothing; delay's `DelayWatcher` (the family's `watcher.py`) freezes and releases
   orbs.
3. `scenario.termination.evaluate(world)` returns an `EpisodeOutcome`. For the family it is
   `families/goal/single_tier_chain/termination.py`: the droid ending the task (out of energy, a chain broken or
   completed) or the clock, with the timeout penalty only when the clock alone ended it. Every scenario in the
   family uses it.
4. If terminated, `droid.on_termination(terminal_reward)` adds what the digesters gathered and did not pay, plus the
   terminal reward, to the score and to this step's reward.
5. At the end of an episode the scenario's metrics go into `info` under `SYN_STATS_KEY`.
6. Render if a mode is set, then the perception writes the observation.

**Who reads what:** termination, the HUD readers and the metrics all read the world. What a digester counts is
reached with `world.droid.get_digester(TierDigester)`, which finds the registered subclass. The readers every scenario
can use are in `scenario/blocks/readers.py`; the family's own are in `builders.py`.

## Where to look

- **What a tier orb pays and when:** `core/droid/digestion/digesters/tier_chain/`. `tier_digester.py` tracks the
  chain; each scoring mode is a subclass. Spatial and sparse use `MaxTierDigester`, dense `ThresholdDigester`.
  `StepWiseDigester` is for the continuous scenarios, which have no entry yet.
- **The negative orb:** a `FixedRewardOrb` from `core/orbs/direct/fixed_reward_orb.py`, its digester `NegDigester` in
  `direct_digester.py`, its population `scenario/blocks/populations/neg_population.py`, and `_negative_bundle` and
  `neg_orb_conf.enabled` in `builders.py`.
- **A new orb kind:** its kind in `core/orbs/orb_meta.py`, its class under `core/orbs/`, a digester under
  `core/droid/digestion/digesters/`, a population under `scenario/blocks/populations/`, and a bundle in the family's
  `builders.py`.
- **What the agent sees:** `scenario/blocks/observation.py` (what the scenario states), the perceptions under
  `gymnasium/observation_space/perceptions/` (how it is encoded), `core/orbs/orb_meta.py` (the identity). The
  perceptions not yet rebuilt are listed in a TODO in `observation_handler.py`; their copies are in `legacy/`.
- **Extractors for SB3:** `runners/agent_runners/utils/extractors.py`.
- **Step and boundary cost, energy, `DroidAction`:** `core/droid/synergy_droid.py`. **Spawning, lifespans, the clock:**
  `core/grid_world.py` and `core/utils/timer.py`.
- **When an episode ends:** `scenario/blocks/termination.py` (the interface), the family's `termination.py`.
- **Delay's freeze:** the family's `watcher.py` (`DelayWatcher` and its recipe), on `Watcher` in
  `scenario/blocks/watcher.py`, reading `StepResult` and `is_cell_free` from `core/grid_world.py`. Its delay is
  `delay_conf` on `TierDelayScenarioConf`, which also sets `max_steps` from it.
- **A scenario's values:** the run YAML in `config/yaml/scenarios/` holds the tunables; the locked rest is in
  `base_configs/`. The models are `config/models/scenario/` (shared vocabulary and orb kinds),
  `scenario/families/goal/config.py` (every goal scenario) and the family's `config.py` (full models and tunables).
- **A new scenario:** an entry in the family's catalogue (`scenarios.py`), a member in `ScenarioName`
  (`config/models/global_models.py`), and a run YAML. A new family adds its catalogue to `SCENARIOS` in `registry.py`.
- **The HUD and the window:** `scenario/blocks/hud.py` (what is shown), `rendering/pygame_renderer.py` (drawing),
  the runners (window input).
- **Training, eval, saving:** `runners/agent_runners/sb3/sb3_runner.py` (`SB3Spec` and the one SB3 runner) and
  `artifact_manager.py`. Each SB3 algorithm is an `SB3Spec` in `RUNNER` (`runner_registry.py`).
- **A runner of your own:** a `BaseAgentRunner` subclass, a member in `Algorithm` (`config/models/runner_models.py`)
  and an entry in `RUNNER`.
- **CSV output:** `gymnasium/utils/episode_logging/`.
