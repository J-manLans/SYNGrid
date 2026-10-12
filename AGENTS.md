# AGENTS.md

Guidance for AI agents working in this repository. Last updated 2026-10-12.
Trust the code over this file, and say so when they disagree.

## Where things stand

**The repo is in rebuild mode, and the rebuild has started.** The owner is
rebuilding the scenario layer **by hand**, file by file, on
`rebuilding-scenario`. `refactor-of-scenario-refactor` is the product of two
large multi-model refactors. Parts of it are good, but the whole is
inconsistent, so it is a reference to read and not a base to build on.

| Branch | Role |
|---|---|
| `rebuilding-scenario` | the rebuild; branched from `cleanup-after-versioning` at `2f0dcc4` |
| `cleanup-after-versioning` | last stable state |
| `refactor-of-scenario-refactor` | reference for the rebuild |
| `main` | older (v0.4.0) |
| `scenario-example` | abandoned first scenario refactor; `origin` only |

The pushed branch can be behind the local working tree. Read local files.

Where to look, in this order:

- `docs/dev/scenario-rebuild-roadmap.md` is the order of work and has a table
  of what is written, empty or still legacy.
- `docs/dev/design-decisions.md` is what was chosen, what was turned down and
  why. Read it before giving a design opinion.
- `docs/dev/open-questions.md` is what has not been decided yet.
- `docs/dev/codebase-ideas.md` is ideas that are not problems to solve now.
- `docs/dev/glossary.md` is what the words mean. Use its terms, and add a term
  there when a new one is coined.
- `docs/dev/regressions.md` is for behaviour that worked before and does not
  now. It has a template and no entries.

State on 2026-10-11: the config models, the registry and the spatial YAML are
done. `scenario.py` and the blocks are written, and the spatial builder returns
a `Scenario`. Its pieces are stubs: the tier-chain population makes no orbs,
its termination never ends an episode, the spawning hooks have no bodies, and
`build_world` raises `NotImplementedError`. The other three builders raise too.
`core/droid/digestion/digestion.py` holds the routing key and the digester
protocol. The tier digester's base and its three scoring modes are written in
`digesters/tier_chain/` and are not wired into a builder. The negative digester
is an empty file, and `engine.py` is still the old engine. `core/`,
`gymnasium/` and `rendering/` still import the old config, so
`import syn_grid.scenario.registry` works while `import syn_grid.app` and
`python3 -m syn_grid.check_env` fail. The roadmap expects that until its
phase 5.

**No file in the working strain is final.** A file can be a stub, half-moved or
about to change shape, and its docstrings describe what it is meant to be, not
how far it has come. What is a stub, what is empty and what is still legacy is
recorded here and in the roadmap's table, and never in a docstring.

## How to work with the owner

- **The owner writes the code.** Act as a collaborator: explain, review, give a
  design opinion, answer questions. Do not edit source, write plans or offer
  "which should I implement" choices unless asked for an edit explicitly.
- **Audit and discuss before writing anything.** Say what exists, what is
  missing and what you would create, then wait.
- **Plain language.** Report what you verified and what you did not.
- **Mention anything surprising, but don't fix it unless asked.**
- **No temporary state in docstrings.** Do not write `STUB:`, "not written
  yet", "until phase N" or "for now" in a docstring. A docstring says what the
  thing is and does, and must still read correctly when the rebuild is
  finished. An old solution that slips into a docstring is hard to find again.
  Progress goes in this file and the roadmap; reasons and rejected alternatives
  go in `design-decisions.md`.
- **Work that is still to do goes in a `# TODO` or `# NOTE` comment,** which the
  owner can search for and tick off. Put a tag in brackets when several belong
  together, as in `# TODO(phase 3):` and `# TODO(lock):`.
- **Do not add, migrate or rewrite tests unless explicitly asked.** The owner
  curates the suite. If you think a change needs coverage, say so and leave it.
  A test that is asked for must fail when its behaviour is broken; reintroduce
  the defect and confirm it goes red.

## The two strains

The working strain is `src/syn_grid/`. A frozen copy of the stable code is in
`src/syn_grid/legacy/`, so it can be trained exactly as it was and its curves
compared with the rebuild. Each legacy package and loose file carries a
`_legacy` suffix (`core_legacy/`, `config_legacy/`, `app_legacy.py`) and
imports the other legacy copies. `USE_LEGACY` in `src/syn_grid/__main__.py`
picks which app runs. Nothing under `legacy/` is edited, and it is deleted when
the rebuild is done.

Legacy is a reference and not a target. Several decisions change rewards on
purpose, so the rebuild is held against legacy's learning curves and not its
rewards step for step. The list of differences is in `design-decisions.md`
under "Legacy is a reference, not a target".

Things the switch depends on:

- Only one strain is imported per process. Both register the Gymnasium id
  `syn_grid-v0`, each with its own entry point.
- A folder rename rewrites imports, not strings. The legacy config still names
  its images as `assets/sprites/...` while the folder is `assets_legacy/`, so on
  2026-10-11 `python -m syn_grid.legacy.check_env_legacy` still stopped in the
  renderer with a `FileNotFoundError`. Whether legacy training, which does not
  render, is affected was not checked.
- The legacy run id has its own scheme (`id_tag`, seed, algorithm). Give legacy
  runs their own tag or output folder.
- The config schemas differ, so a comparison only means something if both
  config files describe the same world. Compare several seeds per side.

## The reference material

`reference/` at the repo root holds three things.

- **`reference/skeleton/scenario/`** is the intended structure of the scenario
  layer: modules, classes, protocols and signatures, with no logic. The files in
  `src/syn_grid/scenario/` started as copies of it and have moved on; where the
  two differ, `src` is the newer one. It still imports the config models from
  `syn_grid.config.models.common_scenario_models` and has a `utils/helpers.py`;
  both are gone from `src`.
- **`reference/syn_grid/`** is a copy of the package from
  `refactor-of-scenario-refactor`. Its config and runner folders are named
  `copied-config/` and `copied-runners/`, and it contains an older copy of the
  skeleton.
- **`reference/scenario-stubs.patch`** holds the stubs for the block files, the
  tier-chain population and termination, and the builders. It is committed and
  its content is in `src`, which has moved on since, so it no longer applies.

Conventions the owner asked for in skeleton files:

- Every function is a signature, a docstring saying what it does or composes,
  and a `...` body. Ruff's `PIE790` objects to that, so each file carries
  `# ruff: noqa: PIE790`.
- Imports use the real package paths (`syn_grid.scenario...`,
  `syn_grid.core...`), so they resolve to the live modules.
- Tables that are structure, such as `SCENARIOS`, are filled in.
- Do not invent Effect orbs or `Cancellation`. They have no implementation.

## Design direction

The decisions and their reasons are in `docs/dev/design-decisions.md`. The
principles behind them:

- **Everything scenario-specific is a field of `Scenario`,** settled before any
  layer below it is rebuilt. The test for a field is "does the Gymnasium side
  need it?"; what only the world needs stays inside `build_world`.
- **A `Scenario` holds nothing that changes during an episode.** Training builds
  16 envs from one `Scenario` in one process. State belongs in what
  `build_world()` returns; rules reach it through the `world` they are handed.
- **If a conditional exists "because scenario X works differently", it does not
  belong in the component that holds it.**
- **A mechanic only one scenario uses lives in that scenario's strategy** when
  it can be written against the world's public API and the orbs' own methods. It
  becomes a world method only when it must touch the world's private
  bookkeeping.
- **One supplied piece per consumer,** each a static description plus a reader
  that is handed the world: `hud` (a tuple of HUD elements), `metrics` (a tuple
  of named readers, the scenario's side of the logger) and the observation's
  global features, which belong in `ObservationRules`. The HUD and the
  observation must be free to diverge.
- **Readers are module-level functions, not lambdas or nested functions,** so
  two scenarios built from the same config compare equal.
- **Termination is an interface file plus one file per strategy,** so each
  strategy imports only what its scenario needs.
- **One orb kind is one brick:** an `OrbBundle` in the scenario layer, carrying
  the population and a way to make the digester, and a `*_orb_conf` block in the
  world config.
- **The logger itself comes late.** It is expected to be a localized change.

Settled since the rebuild started, each with an entry in `design-decisions.md`:

- Digestion lives in the droid, at `core/droid/digestion/`, with a `digesters/`
  folder. The routing key is the orb's kind alone. The droid creates the engine
  from digesters that `build_world` makes, one per `OrbBundle`.
- Digesters do not talk to each other. `digest` returns a float, and there are
  no events, no tally and no cross-kind hook. A reader takes a digester from the
  engine by its class and reads its attributes. Only a tier orb eaten out of
  order breaks a chain, so a negative orb leaves it alone.
- The tier digester is a base class with one subclass per scoring mode:
  max-tier, step-wise and threshold. The base owns the chain, the counts and the
  chain's summed reward.
- A tier orb is worth the same in every scoring mode, and the mode decides when
  it is paid. A break always costs `chain_break_penalty`. Max-tier pays the
  chain's summed worth at the last tier, so there is no fixed completion prize.
- A new scoring mode is a new registered scenario, not a config setting. The
  continuous sandbox is the exception.
- Two orb categories, direct and synergy. `OrbMeta` stores only kind and tier,
  and a tier is `None` for every kind but a tier orb. The identity is unchanged
  and is redone in the observation pass.
- Config models live beside what they configure. A scenario config is
  `world_conf` and `obs_conf`, and each layer adds a block to the world.
- An orb kind's config class is shared in the common models, a family
  subclasses it for its own use, and each scenario's world model declares the
  field, optional or mandatory. The base `WorldConf` declares no orb fields.
- Energy and score are two numbers. `max_energy` is both the start and the cap,
  and replaces `starting_score` and `max_score`.
- The step clock lives in the world. `max_steps` is in `world_conf`; the count
  itself is still on `ObservationHandler`.
- Digestion pays for orbs and termination pays for the clock. The tier digester
  pays the orbs, the chain break and the completed chain; termination pays
  `timeout_penalty`. There is no `completion_reward`.
- A tier orb is worth `base_reward * tier ** growth_factor`, set on the shared
  `TierOrbConf` and not rounded. Scaling dense has no config classes of its own.
- Legacy and `reproduction_package/` are references, not targets.
- Reward normalisation stays off in training (`norm_reward=False`).
- A tier chain holds one negative orb. `NegOrbConf` is `reward` and `cool_down`;
  a weight is only for a family whose field is sampled.
- A locked scenario loads a small model holding only what a run may choose, and
  its builder makes the full config from its fixed values. The full models stay
  as they are, and the sandbox and every scenario that is not locked load them
  directly. Settled as the direction on 2026-10-12 and not applied: no scenario
  is locked until it has reproduced its legacy results.
- A locked scenario offers the perceptions that make sense for it. Spatial
  offers only ones with a limited view.

What is still open is listed in `docs/dev/open-questions.md`, grouped by the
roadmap phase that needs the answer. Read it before giving a design opinion, and
do not treat an item on it as decided.

## Reading the reference

One scenario, `goal_tier_chain_spatial`, runs end to end on
`refactor-of-scenario-refactor`. On 2026-10-08 `python -m syn_grid.check_env`
passed there and 20,000 random actions stayed inside the declared `Box`. That
says nothing about learning.

Traps when reading it. They describe the reference, and some are superseded by
decisions on this branch:

- `delay_on_consume` is `int | None` and `None` means off; `False` turns it on.
- A `negative` block in a tier-chain config registers a digester but creates no
  orbs, so it has no visible effect.
- Scoring mode is not config; each builder passes it. Max-tier for spatial,
  scaling sparse and delay, threshold for scaling dense only. The legacy delay
  configs in `reproduction_package/delay_scenario/` all set `max_tier_scoring`.
- In max-tier scoring a completed chain pays `completion_reward`. Tier orbs are
  built worth 0.0, which threshold scoring cannot work with.
- Threshold scoring and `NegativeDigester` have never been executed there.
- `engine.get(...)` raises `KeyError` when that digester is not registered.
- `TierChainTermination` is cut to what spatial reaches, and its `score <= 0`
  ending is commented out. The removed branches are in
  `scenario/rules/termination.py` at `c105f7a`.
- Five perceptions beyond the two in the `Perception` enum exist but read a
  config that is gone; do not add enum members to reach them.
- `WeightedPopulation` names a config class that no longer exists.
- Its `docs/dev/scenario-refactor.md` is stale.

## This branch

- **Run ids** come from perception, seed, `scenario.scenario_tag`, the optional
  runner `tag` and the algorithm, not from the scenario name.
- **`docs/architecture/`** describes the code as it stood on 2026-10-11,
  including the parts that are still from before the rebuild and do not import.
  It does not describe the intended design; that is in the roadmap and
  `design-decisions.md`.
- **Scenario config models** are in `config/models/scenario/`: `common_models.py`
  and `orb_models.py`. A family's own models are in its `config.py`.
- **`core/droid/digestion/engine.py`** is still the legacy engine, and
  `core/orbs/orb_factory.py` and `gymnasium/utils/episode_termination.py` are
  still present. The roadmap removes both files. The engine picks its scoring
  mode from flags on the tier orb that `TierOrb` no longer has.
- **`env_factory.make` passes a `Scenario`,** but `SYNGridEnv` still takes
  `world_conf` and `obs_conf`.
- **Two scenarios built from the same config do not compare equal** today. The
  only differing field is `build_world`, which is a `functools.partial`.
- **The config is in line with the digesters, and the builders are not.**
  `_make_tier_digester` raises, and `TierChainPopulation` makes no orbs, so
  nothing gives a tier orb its reward from the config yet.
- **`core/orbs/orb_factory.py` calls `TierOrb` with a signature it no longer
  has.** The file does not import and the roadmap removes it.
- **The tier digesters were only tried in throwaway scripts.** No test covers
  them.
- **`blocks/spawning.py` is mid-change.** `SpawningRules` has three fields, its
  docstring still lists `delay_on_consume`, and `OrbFieldUpdate` is declared and
  unused.
- **Tests import the legacy strain.** On 2026-10-11 collection found 230 tests
  and no errors. They were not run.
- **`scripts/replay_probe.py` imports the legacy strain** on this branch. It was
  not run.

## The learning problem

The owner reports that spatial runs on `refactor-of-scenario-refactor` but the
agent does not learn as it should. No agent has verified this, and the cause is
unknown. Do not assume the refactor caused it; it may predate it.

- `docs/dev/rppo-regression.md` is not on this branch. Read it with
  `git show cleanup-after-versioning:docs/dev/rppo-regression.md` before
  touching the reward path. Its original conclusion is superseded: the timeout
  and chain-break rewards used to be the same value, and are independent fields
  now.
- Phase 0 of the roadmap comes first: confirm that the legacy strain learns
  spatial over several seeds, and record a step-by-step trace with the replay
  scripts. A learning regression is silent: the env still runs and passes
  `check_env`. Whether phase 0 has been done is not recorded anywhere in the
  repo.
- When using the replay harness: pin `starting_score` high or episodes end early
  and skip the timeout branch; comparisons are exact, so read the magnitude of a
  difference before calling it behavioural.

## Before trusting a run

`import syn_grid` resolves through the editable install, which can point at a
different checkout. Check first:

```bash
.venv/bin/python -c "import syn_grid; print(syn_grid.__file__)"
```

`PYTHONPATH=src` overrides it for one command.

```bash
pip install -e ".[dev]"                    # required
python3 -m syn_grid                        # run; USE_LEGACY in __main__.py picks the strain
python3 -m syn_grid.check_env              # Gymnasium contract self-check
ruff check src/syn_grid tests              # lint, CI scope
./scripts/tidy.sh                          # check --fix + format
```

`python` is not on PATH outside the venv. `requires-python` is `>=3.10,<3.11`.
Rendering needs `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` on a machine
without devices.

## Verify before asserting

Each of these cost a wrong conclusion in this repo.

- **A config value's effective setting is in the code that consumes it,** not
  where it appears in YAML. A penalty was once absent from config and hardcoded
  in the engine.
- **When comparing, confirm the variable was free to vary.** Pinning it makes
  "no change" vacuous.
- **A passing diff means the code agreed, not that the question was asked.**
  Check the resolved config.
- **No visible prompt does not mean no permission decision.** Prompts are
  client-side.
- For where a behaviour came from or when it changed, prefer `git log -S`, a
  replay probe or loading the artifact over inference from a name or a comment.

## Things that bite regardless of the refactor

Carried over from earlier versions of this file. The roadmap's phase 3 is where
the first two are dealt with.

- **`BaseOrb._life_span` is a class attribute** set whenever a world is built,
  and shared by every env in the process.
- **`GridWorld.reset()` with no argument uses an unseeded rng,** so a second
  reset re-rolls the orb layout. Pass `np.random.default_rng(seed)`.
- **Every perception returns the same buffer each step,** so `obs_t is
  obs_{t+1}`. Anything that keeps two observations must copy.
- **`reproduction_package/` is committed and large.** Do not add to it. It is a
  record of what the thesis runs did, not something the rebuild must reproduce.
- **`plot/` is a thesis scratch module** with hardcoded paths evaluated at
  import, and needs the optional `extra` dependencies.
- **`docs/code_style.md` mandates Black;** the repo uses ruff.
