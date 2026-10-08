# AGENTS.md

Guidance for AI agents working in this repository. Last updated 2026-10-08.
Trust the code over this file, and say so when they disagree.

## Where things stand

**The repo is in rebuild mode.** `refactor-of-scenario-refactor` is the product
of two large multi-model refactors. Parts of it are good, but the whole is
inconsistent, and the owner no longer has a clear picture of how it fits
together. It is not being patched further. The owner is rebuilding the scenario
layer **by hand**, file by file, from a skeleton, using this branch as a
reference.

| Branch | Role |
|---|---|
| `cleanup-after-versioning` | last stable state; the rebuild branched from here |
| `rebuilding-scenario` | where the rebuild happens; holds the legacy strain |
| `refactor-of-scenario-refactor` | reference for the rebuild, and where the skeleton is written; not a base to build on |
| `main` | older (v0.4.0) |
| `scenario-example` | abandoned first scenario refactor; `origin` only |

The pushed branches can be behind the local working tree. Read local files.
On `refactor-of-scenario-refactor` much of the reference is uncommitted local
work (the digestion modules, `hud.py`, `metrics.py`, the skeleton), so do not
discard or stash it carelessly. This file exists on both branches; the copy on
the branch you were not working on may be older.

## How to work with the owner

- **The owner writes the code.** Act as a collaborator: explain, review, give a
  design opinion, answer questions. Do not edit source, write plans or offer
  "which should I implement" choices unless asked for an edit explicitly.
- **Audit and discuss before writing anything.** Say what exists, what is
  missing and what you would create, then wait.
- **Plain language.** Report what you verified and what you did not.
- **Mention anything surprising, but don't fix it unless asked.**
- **Do not add, migrate or rewrite tests unless explicitly asked.** The owner
  curates the suite. If you think a change needs coverage, say so and leave it.
  A test that is asked for must fail when its behaviour is broken; reintroduce
  the defect and confirm it goes red.

## The rebuild branch

`rebuilding-scenario` was created from `cleanup-after-versioning` and holds two
strains of the code side by side.

**Legacy strain: `src/syn_grid/legacy/`.** A frozen copy of the stable code.
Folders and loose files carry a `_legacy` suffix (`core_legacy`,
`config_legacy`, `assets_legacy`, `app_legacy.py`) and import only each other
through `syn_grid.legacy.`. It is never edited, and it is deleted when the
rebuild is done. It exists so the stable code can be trained exactly as it was
and its curves compared with the rebuild.

**Working strain: the normal paths.** This is what gets rebuilt, file by file,
from the skeleton and from the refactor branch's code. Setting up the
surrounding wiring by hand is part of the exercise: it is how the weak joints
are meant to be found.

- Already brought over from the refactor branch: `app.py`, `check_env.py`,
  `config/` (without the old-schema `config/scenarios/`), `runners/`,
  `core/orbs/direct/negative_orb.py` and `docs/architecture/`.
- Still stable-shaped: `core/`, `gymnasium/`, `rendering/`, `utils/`. There is
  no `scenario/` package yet. So the working strain does not import yet, which
  is expected.
- Small glue files suggested but not copied when this was written:
  `gymnasium/utils/env_factory.py` (the copied files call
  `make(scenario, render_mode)`) and `plot/plot_utils.py` (on the rebuild branch
  it imports from `syn_grid.legacy`, which breaks when legacy is deleted).
- `tier_orb.py` was deliberately not copied. Its new shape carries three design
  decisions (no config object, no scoring flags on the orb, `max_tier` as a
  constructor argument) that the owner wants to meet during the rebuild.

**The switch.** `src/syn_grid/__main__.py` has `USE_LEGACY` and imports the
chosen app inside the branch, so only one strain is loaded per process.

How to tell a leak: a legacy run whose traceback shows a `syn_grid` path
outside `syn_grid/legacy/` has loaded a working-strain file. Imports are
rewritten by a folder rename; strings are not. The three that had to be fixed by
hand were the Gymnasium `entry_point` in
`legacy/gymnasium_legacy/utils/env_factory.py` and the hardcoded `"assets"`
lookups for the font and the training-complete sound. The legacy path helper
resolves its root to `syn_grid/legacy/`, which is correct because config and
assets live there.

Verified on 2026-10-08: `python -m syn_grid.legacy.check_env_legacy` passes, a
rendering env steps, the sound loads, and no module outside `syn_grid.legacy`
gets imported.

Still to sort out before comparing curves:

- Both strains write to the same `output/` folders with the same run-id scheme,
  so legacy runs need their own tag or folder.
- Once the config schemas differ, a comparison only means something if both
  config files describe the same world. Compare several seeds per side.

Package inits stay empty, in the rebuild too, and imports use full module
paths. The four inits that re-export (`perceptions/vector`, `composite`,
`spatial`, and `runners/agent_runners/sb3`) are kept on purpose.

## The reference skeleton

`src/syn_grid/reference/skeleton/` holds the intended structure of the scenario
layer: modules, classes, protocols and signatures, with no logic. It mirrors the
real paths (`skeleton/scenario/...`, `skeleton/core/digestion/...`). It lives
inside the package while it is being written and moves to a top-level
`reference/` on the new branch. Once agreed, **the skeleton is the working
truth** and is edited until it fits.

Conventions the owner asked for:

- Every function is a signature, a docstring saying what it does or composes,
  and a `...` body. Ruff's `PIE790` objects to `...` after a docstring. The
  owner removed the `# ruff: noqa: PIE790` line from `registry.py`, so that file
  fails lint on purpose; do not re-add it or strip the `...` bodies.
- Imports use the real package paths (`syn_grid.scenario...`,
  `syn_grid.core.digestion...`), never `syn_grid.reference...`. They therefore
  resolve to the live modules, not to other skeleton files.
- Tables that are structure, such as `SCENARIO_BUILDERS`, are filled in.
- Do not invent Effect orbs or `Cancellation`. They have no implementation.

Written so far, all under `skeleton/scenario/`:

| File | State |
|---|---|
| `registry.py` | signatures and docstrings; keeps the current shape where nothing was agreed |
| `scenario.py` | `hud` and `metrics` are fields |
| `rules/observation.py` | `GlobalFeature` added, `global_features` is a field, `max_score` removed |
| `rules/hud.py`, `rules/metrics.py` | as the owner wrote them, with docstrings |
| `rules/termination/termination.py` | the interface only |

Not written yet: `rules/population.py`, `rules/spawning.py`,
`rules/termination/tier_chain_termination.py`,
`rules/termination/continuous_termination.py`, and all of `core/digestion/`.

Five decisions shape several of those files and should be settled first:

1. Orbs and digesters: separate `orb_population` and `scoring_mode`, or one
   per-kind recipe that supplies both.
2. `Event` and `engine.count()`: keep, rename, or drop in favour of counts on
   the digester.
3. The cross-kind hook: `notice` on every digester, or a second protocol for
   synergy digesters only.
4. The step clock: where it lives decides what a reader is handed.
5. Scope: whether the skeleton includes the continuous family's pieces.

Open on `registry.py` itself:

1. `max_score` is still a parameter of the family helper, but the skeleton's
   `ObservationRules` no longer has that field.
2. No negative-orb count anywhere in the helper or the world builder.
3. Builders take the base `ScenarioConf` and `cast` it, because the models table
   and the builders table are separate.
4. Whether the world readers belong in the registry or in a tier-chain file.

## Design direction

Agreed with the owner:

- **Everything scenario-specific is a field of `Scenario`,** settled before any
  layer below it is rebuilt. The test for a field is "does the Gymnasium side
  need it?"; what only the world needs stays inside `build_world`.
- **A `Scenario` holds nothing that changes during an episode.** Training builds
  16 envs from one `Scenario` in one process. State belongs in what
  `build_world()` returns; rules reach it through the `world` they are handed.
- **If a conditional exists "because scenario X works differently", it does not
  belong in the component that holds it.**
- **One supplied piece per consumer,** each a static description plus a reader
  that is handed the world: `hud` (a tuple of HUD elements), `metrics` (a tuple
  of named readers, the scenario's side of the logger) and the observation's
  global features, which belong in `ObservationRules`. The HUD and the
  observation must be free to diverge.
- **Readers are module-level functions, not lambdas or nested functions,** so
  two scenarios built from the same config compare equal.
- **Termination is an interface file plus one file per strategy,** so each
  strategy imports only what its scenario needs.
- **The logger itself comes late.** It is expected to be a localized change.
- **In digestion, the scoring and routing are fine; the scaffolding is weak:**
  how state is read out, and how digesters are paired with orbs.

Still open:

- **`Event` and `engine.count()`.** The class is only ever counted, only the
  tier digester emits any, and a step yields at most one. Options: rename it for
  what it is, or move the counts onto the digester and drop it.
- **The cross-kind hook (`notice`).** It is synergy behaviour. Rename it, or move
  it to a second protocol that only synergy digesters implement.
- **A per-kind recipe that supplies both the orbs and the digester,** so a kind
  is added in one place. It must be a recipe, not instances.
- **Negative orbs in a tier chain.** Intended: a few sit in the pool, spawn,
  despawn at the end of their lifespan and reappear elsewhere after their
  cool-down; under delay they freeze and return with the field. That needs a
  kind filter on the refill action, a tuple of after-actions, the negative count
  added to field size and observation slots, and a tier-chain negative model
  without `weight`.
- **Where the step clock lives.** It is on `ObservationHandler`, so a reader
  that only gets the world cannot supply steps or moves.
- **HUD form.** A small vocabulary of elements the renderer can draw, or a
  drawer per scenario. The current HUD is one sprite with fixed positions.
- **Generic readers that assume a tier digester:** the env's HUD and log info,
  and `base_perception`, which every perception inherits.

## The refactor branch, as a reference

One scenario, `goal_tier_chain_spatial`, runs end to end. On 2026-10-08
`python -m syn_grid.check_env` passed and 20,000 random actions stayed inside
the declared `Box`. That says nothing about learning.

Local, unfinished state: `hud` and `metrics` are built in
`scenario/registry.py` but not passed to `Scenario`, whose two fields are
commented out. `rules/observation.py` carries the owner's notes for the global
features, with `GlobalFeature` currently nested inside `ObservationRules`. The
environment still reads the tier digester directly. The two scenario inits have
been emptied.

How it is put together:

- **Config** is three YAMLs in `src/syn_grid/config/yaml/` (global, runner, and
  one named after the scenario), validated by frozen, strict pydantic models
  that forbid unknown keys. `SCENARIO_MODELS` picks the scenario's schema.
- **`scenario/registry.py`** is the composition root. `_tier_chain_scenario` is
  the family helper and `_build_tier_chain_world` builds one world per env.
  Three of the four builders are placeholders.
- **`core/digestion/`** routes a consumed orb to the digester that owns its
  kind. `TierOrbDigester` holds the chain state and two scoring modes.
- **`docs/architecture/`** describes this branch's design. Treat it and the
  docstrings as intent. `docs/dev/scenario-refactor.md` is stale.

Traps when reading it:

- `delay_on_consume` is `int | None` and `None` means off; `False` turns it on.
- A `negative` block in a tier-chain config registers a digester but creates no
  orbs, so it has no visible effect.
- Scoring mode is not config; each builder passes it. Max-tier for spatial and
  scaling sparse, threshold for scaling dense and delay.
- In max-tier scoring a completed chain pays `completion_reward`. Tier orbs are
  built worth 0.0, which threshold scoring cannot work with.
- Threshold scoring and `NegativeDigester` have never been executed here.
- `engine.get(...)` raises `KeyError` when that digester is not registered.
- `TierChainTermination` is cut to what spatial reaches, and its `score <= 0`
  ending is commented out. The removed branches are in
  `scenario/rules/termination.py` at `c105f7a`.
- Run ids come from perception, seed, `scenario.scenario_tag`, the optional
  runner `tag` and the algorithm, not from the scenario name.
- `Perception` lists only the two verified perceptions. Five more exist but read
  a config that is gone; do not add enum members to reach them.

Broken here, deliberately left:

- Tests fail collection (old schema, deleted modules).
- `scripts/replay_probe.py`, `replay_sweep.sh` and `scenario_lock.py` target the
  old schema and do not run, so behaviour here is not locked to the baseline.
- `WeightedPopulation` names a config class that no longer exists.
- `config/scenarios/*.yaml` and `reproduction_package/` configs are old-schema.

## The learning problem

The owner reports that spatial runs on the refactor branch but the agent does
not learn as it should. No agent has verified this, and the cause is unknown.

The owner ran the legacy strain on spatial with RPPO for about 3M steps and
reported that it checked out. That is one seed, and no agent has
seen the curve. It suggests the stable code learns and the problem came in
later, but treat that as likely, not proven.

- Read `docs/dev/rppo-regression.md` before touching the reward path. Its
  original conclusion is superseded: the timeout and chain-break rewards used to
  be the same value, and are independent fields now.
- A learning regression is silent: the env still runs and passes `check_env`.
  The check for each rebuilt step is the legacy strain: same config, same seed,
  compare the curves.
- Legacy behaviour has not been locked with a recording. On the rebuild branch
  `scripts/replay_probe.py` already points at the legacy strain.
  `scripts/scenario_lock.py` and its baseline exist only on the refactor branch
  and would need their imports pointed at legacy first.
- When using the replay harness: pin `starting_score` high or episodes end early
  and skip the timeout branch; comparisons are exact, so read the magnitude of a
  difference before calling it behavioural; spatial's observation here has 3 orb
  slots where it had 5, so old digests will not match.

## Before trusting a run

`import syn_grid` resolves through the editable install, which can point at a
different checkout. Check first:

```bash
.venv/bin/python -c "import syn_grid; print(syn_grid.__file__)"
```

`PYTHONPATH=src` overrides it for one command. `../syngrid-known-good` is a
detached commit (`e0e9259`), not `cleanup-after-versioning`.

```bash
pip install -e ".[dev]"                    # required
python3 -m syn_grid                        # run
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

Observed on this branch, most carried over from the previous version of this
file. Check them again on the stable one.

- **`BaseOrb._life_span` is a class attribute** set whenever a world is built,
  and shared by every env in the process.
- **`GridWorld.reset()` with no argument uses an unseeded rng,** so a second
  reset re-rolls the orb layout. Pass `np.random.default_rng(seed)`.
- **Every perception returns the same buffer each step,** so `obs_t is
  obs_{t+1}`. Anything that keeps two observations must copy.
- **`reproduction_package/` is committed and large.** Do not add to it.
- **`plot/` is a thesis scratch module** with hardcoded paths evaluated at
  import, and needs the optional `extra` dependencies.
- **`docs/code_style.md` mandates Black;** the repo uses ruff.
