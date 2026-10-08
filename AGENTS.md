# AGENTS.md

Guidance for AI agents working in this repository. Verified against the repo at
`refactor-of-scenario-refactor`; trust executable config over prose.

**This branch is mid-refactor.** Read "Branch status" and "What is broken on
this branch" before trusting a run, a test or a script.

## Branch status

| Branch | Status |
|---|---|
| `cleanup-after-versioning` | last stable state of the codebase |
| `main` | older (v0.4.0) |
| `scenario-example` | copy of the abandoned first scenario refactor; `origin` only |
| `refactor-of-scenario-refactor` | active; a top-down rebuild |

Everything branched from `cleanup-after-versioning` is work in progress and may
be broken. That includes `scenario-example`, which tried to make scenarios
first-class objects across the whole codebase. Don't treat these branches as a
reference for how things should work, and don't judge the code by whether it
currently runs.

On the active branch one scenario (`goal_tier_chain_spatial`) runs end to end.
The test suite and the replay/lock scripts still target the previous config
schema and do not run.

The `../syngrid-known-good` worktree named below is a detached commit
(`e0e9259`) that is on no branch. It is not `cleanup-after-versioning`.

## Working rules on this branch

The rebuild has one rule: **there is only one version of anything.**

- Edit or replace the existing file in place, and don't leave the old version
  next to it.
- Do not create parallel versions of files, and don't use `new_` prefixes or
  similar. An earlier phase did this and it was abandoned because juggling two
  versions was too hard to work with. `new_*` files on the pushed branch are
  leftovers from that phase, not a pattern to follow.
- Work goes top-down: the YAML and model files first, then `app.py` and the
  runners package, then the scenario package. Code further down may still have
  the old shape, so imports and call signatures may not match between layers.
- Mention anything surprising, but don't fix it unless asked.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # required; see the editable-install trap below
```

```bash
python3 -m syn_grid                             # run (there is no [project.scripts] entry point)
python3 -m syn_grid.check_env                   # gymnasium contract self-check
ruff check src/syn_grid tests                   # lint  (CI scope — not the whole repo)
ruff format --check src/syn_grid tests          # format
./scripts/tidy.sh                               # check --fix + format; needs venv on PATH
pytest                                          # full suite — currently fails collection
```

There is no Makefile, no pre-commit config, and no single test entry point.

`syn_grid.check_env` is a standalone dev tool, not a config flag: it builds the
scenario `global_config.yaml` selects, makes one env, and runs Gymnasium's
`check_env` on it. Run it after touching the environment.

## The editable-install trap (check this first)

`import syn_grid` resolves through the editable install, which can point at a
**different checkout**. A sibling worktree at `../syngrid-known-good` has been
the target in the past, which makes local runs silently test the wrong code —
or fail collection entirely, because the two checkouts have different module
layouts.

Always confirm before trusting any test or run:

```bash
python -c "import syn_grid; print(syn_grid.__file__)"
```

To probe a revision without touching the venv, `PYTHONPATH=src` takes precedence
over the `.pth` file. `PYTHONPATH=../syngrid-known-good/src` does the same for
that worktree. Library versions are shared, so this isolates code only.

## Config system

Config is **inside the package**, in `src/syn_grid/config/yaml/`, and is three
files rather than one:

| File | Model | Holds |
|---|---|---|
| `global_config.yaml` | `GlobalConf` | `scenario` (a name), `human_control`, `snapshot` |
| `runner_config.yaml` | `RunnerConf` | algorithm, seed, train/eval settings |
| `<scenario name>.yaml` | looked up in `SCENARIO_MODELS` | that scenario's world and observation tunables |

`app.load_experiment_configs` loads the global file first, then uses
`global_conf.scenario` both as the scenario file's name and as the key into
`SCENARIO_MODELS` (`config/models/scenario_registry.py`) to pick the schema.
Editing a config requires an editable install, and there is no `--config` flag
yet.

Models live in `config/models/`: `common_models.py` is the shared vocabulary
(grid, droid, orb kinds, observation blocks), `tier_chain_models.py` is the tier
chain family, `global_models.py` and `runner_models.py` are the other two files.
The split is by *family*, not by scenario name: four tier-chain names share one
hierarchy, and a scenario gets its own subclass only when it needs a field the
family lacks (`TierDelay*` adds `delay`, `TierDense*` adds the reward ladder
and `perception_conf`).

- **Every model is `frozen=True, extra="forbid", strict=True`**, spelled out as
  class kwargs on each class (there is deliberately no shared base). An unknown
  or misspelled key fails at load. This closed the old trap where a typo'd key
  was silently dropped.
- **Single-field bounds are `Field` constraints** (`gt=0`, `ge=0`, `le=0`);
  `@model_validator` is only for rules that compare fields. Validators have
  descriptive names (`validate_penalties`, `validate_chain_fits_grid`) because a
  subclass validator with the same name silently replaces its parent's.
- **A config names its scenario, then configures it.** No field identifies a
  scenario. `ScenarioName` has four members but only `goal_tier_chain_spatial`
  has a YAML and a builder; selecting another fails with `FileNotFoundError`.
- **`max_active_orbs` is not in any model.** A tier chain's field size is its
  `max_tier`. The field returns with the continuous family's pool model.
- **`Perception` lists only verified perceptions** (`vector_fog_of_war`,
  `vector_markovian`). `PERCEPTIONS` in `observation_handler.py` has seven
  entries; the other five are deliberately unselectable until each is verified.
  Do not add enum members to make them reachable.
- **A `negative` block in a tier-chain config has no visible effect.** It
  registers a `NegativeDigester`, but `TierChainPopulation` creates only tier
  orbs.
- `human_control: true` routes `app.main()` to `HumanRunner` and skips training
  entirely. `snapshot.enabled` short-circuits: it saves a config copy and exits.
- `config/scenarios/*.yaml` are **old-schema files that no longer load**. They
  are kept as the source for porting the remaining scenarios; do not treat them
  as runnable.

## The scenario boundary

`Scenario` (in `src/syn_grid/scenario/scenario.py`) is a **recipe**: a frozen
dataclass of identity, the rules the Gymnasium adapter needs, and a way to build
the world.

```python
scenario_name, scenario_tag, observation, termination, build_world
```

`app.py` builds one `Scenario` from config and hands it to the runner. Each env
calls `scenario.build_world()` and gets a `GridWorld` with its own droid, orbs
and digestion engine.

> A `Scenario` must hold nothing that changes during an episode.

Training builds 16 envs from the same `Scenario` object in one process. Anything
stateful stored on it — a world, a droid, a digester — is shared by all of them.
State belongs in what `build_world()` returns. The rule objects on the scenario
(`ObservationRules`, `TierChainTermination`, `SpawningRules`) are frozen and
stateless for the same reason; termination reaches episode state through the
`world` it is handed, never through a reference of its own.

The second working rule:

> If a conditional's purpose is "because scenario X works differently from
> scenario Y", it does not belong in the component that holds it.

Runtime components receive built pieces, not config. `GridWorld` takes
`grid_dimensions`, `droid`, `orb_population`, `spawning`; `SynergyDroid` takes its
`DigestionEngine`; the env, `ObservationHandler` and the perceptions read
`scenario.observation`. The world is the only holder of the grid size: the env
hands `world.grid_dimensions` to the renderer and the perceptions, and the
scenario does not carry it. If you find
yourself passing a `*Conf` below `scenario/registry.py`, the builder is the
place that should have consumed it.

Builders live in `scenario/registry.py`. `_tier_chain_scenario(...,
delay_on_consume=)` is the shared helper for the tier-chain family and
`_build_tier_chain_world` is what `build_world` calls. Three builders are still
`...` placeholders.

- **`delay_on_consume` is `int | None`, and `None` means off.** `SpawningRules`
  tests `is not None`, so passing `False` turns the delay mechanic *on*.
- **Each reward has one owner.** Digestion owns every reward caused by
  consuming an orb; termination owns when the episode ends and the one reward
  the clock causes (`timeout_penalty`). Termination does not overwrite an orb
  reward.
- **Scoring mode is not config.** Each builder passes its `ScoringMode` (now in
  `core/digestion/tier_digester.py`) to the digester: max-tier for spatial and
  scaling sparse, threshold for scaling dense and delay.
- **`TierChainTermination` is cut to what spatial reaches.** The removed
  branches (the 10.0
  completion ceiling and its `curriculum` flag, the exhausted delay field, delay
  suppressing the chain-break ending, pending-reward settlement on timeout) are
  in `scenario/rules/termination.py` at `c105f7a`, for when their scenarios are
  ported.
- **The builders still `cast` their config**, because `SCENARIO_MODELS` and
  `SCENARIO_BUILDERS` are two tables on the same key. Merging them is open.
- **Run ids do not include the scenario name.** `_set_models_base_id` derives
  identity from perception, seed, `scenario.scenario_tag`, the optional `tag` from
  `runner_config.yaml` joined onto it, and the algorithm. The scenario tag is
  the scenario's own axis (grid for spatial). Grid size is no longer
  added separately, so a change outside the axis is only in the id if the user
  tags it. Changing it orphans every existing checkpoint path.

`Cancellation` and Effect Orbs appear in the intended hierarchy but have no
implementation. Do not invent them. `docs/dev/scenario-refactor.md` describes
the *previous* design (rule-bag `Scenario`, `OrbFactory`, `FullConf`) and is
stale.

## Digestion

`core/digestion/` turns "an orb was consumed" into a reward. It routes by orb
*kind*, not by scenario.

- `digestion.py` — the contract: `OrbDigester`, `DigestionResult`, `Event`.
- `engine.py` — `DigestionEngine`: hands the orb to the digester that owns its
  kind, lets every other digester `notice()` it, and tallies events. It holds no
  scoring rules.
- `tier_digester.py` — the two tier-chain scoring modes (max-tier, threshold)
  and the chain state. The scoring
  mode is a constructor argument; orbs no longer carry it.
- `negative_digester.py` — a negative orb is worth its own reward.

In max-tier scoring a completed chain pays `GoalDroidConf.completion_reward`,
not the last orb's reward. Tier-chain orbs are built worth 0.0 with no
cool-down; `TierOrb` still takes the reward-ladder and cool-down arguments, but
`TierOrbConf` no longer has them; the spatial builder states them when it makes
its `TierChainPopulation`. The ladder is config only for the scenario that pays
per tier: `TierDenseOrbConf` carries it, and `TierDenseScenarioConf` also brings
back a `perception_conf` block (`max_score`). Those models are registered in
`SCENARIO_MODELS` but have no YAML and no builder yet.

Readers get counts with `engine.count(ChainBroken)` and digester state with
`engine.get(TierOrbDigester).pending_reward` / `.chained_tiers`. `get` raises
`KeyError` when that digester is not registered, so a scenario without tier orbs
will need those readers (termination, the HUD, `base_perception`) to stop
assuming one.

`TierOrbDigester` was verified equal to the old `DigestionEngine` on ~196k random
digests before the old engine was deleted. Step-wise scoring, with its known
defect (`_max_reward_bonus` not cleared on a chain break), has since been
removed along with `tier_consumption_penalty` and `reward_multiplier`. It
belongs to the continuous family; the last copy is
`core/digestion/new_tier_digester.py` at `c105f7a`.

`TierChainTermination` does not end an episode on `score <= 0`. The check is
commented out pending one training run; see the note in `termination.py`.

## Verify before asserting

All four mistakes below happened in this repo, and each cost a wrong conclusion
that survived until someone checked. Each is a specific check, not a general
instruction to be careful.

**A config value's effective setting is not where it appears in YAML.** Read the
code that consumes it. `chain_break_penalty` was once absent from the config but
hardcoded as `-0.1` inside the digestion engine; a grep of the config alone said
the penalty did not exist.

**A field duplicated across config blocks is not changed by changing one.** The
duplicates are gone from the current models, but the general form still bites:
when probing or comparing, confirm the variable you care about was free to vary.
Pinning it holds the experiment constant and the "no change" result is then
vacuous.

**A passing diff means the code under test agreed, not that the question was
asked.** Check the resolved config, not the source of the value. Two probes using
different scoring modes produce identical-looking output for entirely different
reasons.

**Absence of a visible prompt is not absence of a permission decision.** Approval
prompts are client-side and invisible to the agent. Do not conclude a
configuration is inert because a command returned without asking.

More generally: for any claim about *where* a behaviour came from or *when* it
changed, prefer `git log -S`, a replay probe, or loading the artifact over
inference from a message, a filename, or a config comment.

## Global mutable state

`GridWorld.__init__` sets one **class attribute** every time a world is built:

```python
BaseOrb.set_life_span(grid_rows, grid_cols)   # sets cls._life_span
```

Training uses `n_envs: 16` in a single process, so all envs share this. Two
`GridWorld`s with different geometry will silently rewrite each other's lifespan,
and `base_perception.py` snapshots the value at construction, so a declared
`Box` high can stop matching the timers it emits. It is genuinely world-wide —
the grid's Manhattan diameter — and it is the last global of its kind.

A consequence worth internalising: **`GridWorld.reset()` with no argument uses an
unseeded `default_rng()`.** Any test that resets twice, or resets after building
the world, silently re-rolls the orb layout and becomes a coin flip. Pass
`np.random.default_rng(seed)`.

## Observations alias one buffer

Every perception returns `self._obs_data` — the same object each step, and
`environment.py` hands that straight back. So `obs_t is obs_{t+1}`. SB3 copies
into rollout buffers so training is unaffected, but anything retaining two
consecutive observations (video recording, logging, `np.array_equal` on history)
gets silently corrupted data. `get_orb_positions()` and friends likewise return
internal lists by reference.

## What is broken on this branch

Known and deliberate; the "don't fix it unless asked" working rule applies.

- **Tests: 17 collection errors.** They import `FullConf`,
  `syn_grid.config.models` re-exports, the deleted `core/droid/digestion_engine`
  and `OrbFactory`. `tests/utils/config_helpers.py` is on the old schema too.
- **`scripts/replay_probe.py`, `replay_sweep.sh`, `scenario_lock.py`** import
  `FullConf` and read `test_configs.yaml`, neither of which exists. Behaviour on
  this branch is therefore **not locked** against the pre-refactor digests in
  `scripts/scenario_lock_baseline.json`.
- **Five perceptions** (`vector_markovian_easy`, the three `composite_*`,
  `grid_pixel`) read `include_timer` / `enabled_orbs` from a perception config
  that no longer exists. They are unselectable, so nothing reaches them.
- **`WeightedPopulation`** still names `OrbFactoryConf`. It belongs to the
  continuous family, which has no models or builder yet.
- `[tool.setuptools.package-data]` declares only `assets/**/*`, so a
  non-editable install cannot find its YAML.

## Test suite

**Do not add, migrate or rewrite tests unless explicitly asked.** The owner
writes and curates this suite themselves. Fixing a bug, changing a config value,
or deleting a module a test imports is not a request to touch tests, and "the
change should be regression-tested" is not implied by anything in this file. A
commit may legitimately ship with no new test and with existing tests still
broken. If you believe a change needs coverage, say so in your summary and let
the owner decide.

The bar for anything that *is* asked for: a test must fail when the behaviour it
describes is broken. A test that passes against a stub, that asserts arithmetic
on its own literals, or that only exercises a mock's default value is worse than
no test, because it reads as coverage. Check that a new test actually catches its
bug — reintroduce the defect and confirm it goes red.

Until the suite is migrated, verify changes by running the real thing: load the
three YAMLs through `ConfigManager`, build the scenario, run `python3 -m syn_grid.check_env`, step a
few thousand random actions asserting observations stay inside the declared
`Box`, and do a short training run with outputs switched off. Report what was
and was not exercised.

## Architecture

```text
app.py ── load 3 YAMLs ──▶ build_scenario ──▶ Scenario ──▶ runner ──▶ env (×n)
                                                              │
                                              scenario.build_world() per env
```

`core/` is the simulation. It imports `DroidAction` from
`gymnasium.action_space` — a pure `Enum` with no SB3 dependency, but the import
direction is still `core → gymnasium`. `core/grid_world.py` imports scenario
rule types under `TYPE_CHECKING` only; keep it that way, because
`scenario/registry.py` imports `GridWorld` at runtime and a real import back
would be circular. `gymnasium/` wraps a world and owns spaces and obs encoding.
`runners/` owns training/eval, artifact IO and naming. `scenario/` is the
composition root: it imports core and builds it.

`orb_meta.py` is a value object, **not** a metaclass — there is no orb registry.
Adding an orb type means editing `orb_meta.py`, a population in
`scenario/rules/population.py`, a digester in `core/digestion/`, the config
models, `composite_grid_markovian.py`'s channel map, and `pygame_renderer.py`.
Adding a *perception* needs an entry in `observation_handler.PERCEPTIONS` and,
once verified, a `Perception` enum member.

`core/orbs/effects/` contains only a `dummy_file` and is not a package.

`plot/` is a thesis scratch module (hardcoded CWD paths evaluated at import,
`"<-- replace with your actual tag"`), and needs `matplotlib`/`pandas` from the
optional `extra` group that CI never installs. It is being rewritten.

## Repo layout

`reproduction_package/` is **committed and large** — model / vec-norm / tfevents
binaries; the repo's pack size is dominated by it. Its configs are old-schema
and no longer load. It is slated for removal before publication, so do not add
to it.

`output/` and `.venv/` are gitignored. Note that `.gitignore` lists `.vscode/`
then `!.vscode/launch.json`; the negation looks like a no-op but `launch.json` is
in fact tracked and `settings.json` correctly ignored.

`pyproject.toml` pins `requires-python = ">=3.10,<3.11"`. Nothing in the code
needs the ceiling, but don't assume a newer interpreter works.

## Replay harness and behaviour lock

`scripts/replay_probe.py`, `replay_diff.py`, `replay_sweep.sh` and
`scenario_lock.py` answer "did environment behaviour change between two
commits?" by feeding a fixed seeded action tape to the env and comparing
recordings. They are change detectors, not tests, and cannot tell you whether an
agent learns.

**None of them runs on this branch** (see above). When they are ported, the
lessons already encoded in them still apply:

- Pin `starting_score` high, or the droid's score drains on wall contact and
  every episode ends early — silently skipping the timeout branch.
- Comparisons are exact, no epsilon, so a float-reassociating refactor shows
  `1e-16` differences. Read the magnitudes before calling them behavioural.
- A comparison across a config-schema change needs two config files expressing
  the same world; one file cannot satisfy both schemas.
- Spatial's observation is now 3 orb slots where it used to be 5 (two always
  zero), so its old digest will not match even with identical dynamics.

## Pygame

Constructing a `render_mode="human"` env needs a video device; CI sets
`SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`. `pygame.mixer` is
initialised when training stops in `base_sb3_runner.py` and will raise on
machines with no audio device unless the dummy driver is set.

## In-flight work

Notes on current work live in `docs/dev/`.

`docs/dev/todo.md` is the v1.0.0 list plus, at the bottom, the notes that set
this refactor's direction ("Scenario-first runtime"). Several of its items are
now done or moot; read it as history until it is rewritten.

`docs/dev/rppo-regression.md` records the RPPO spatial investigation. **Its
original conclusion is superseded** — it blamed `chain_break_penalty` alone,
which was wrong: the timeout reward and the chain-break reward were the *same
value*, pinning their ratio at 1:1. They are independent fields now
(`GoalDroidConf.timeout_penalty`, `TierDroidConf.chain_break_penalty`). Read it
before touching the reward path: it lists which causes are ruled out, and the
seed sweep that decides whether the current pair is the *only* working one is
still outstanding.

`docs/code_style.md` documents conventions, but its §3 mandates Black — the repo
actually uses ruff, and it names a path (`src/synergygrid`) that does not exist.
Trust `scripts/tidy.sh` and CI. `docs/workflow.md` describes a two-person
branch/PR flow.
