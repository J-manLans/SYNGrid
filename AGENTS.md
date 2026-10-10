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
| `cleanup-after-versioning` | last stable state; the rebuild starts from here |
| `refactor-of-scenario-refactor` | reference for the rebuild; not a base to build on |
| `main` | older (v0.4.0) |
| `scenario-example` | abandoned first scenario refactor; `origin` only |

The pushed branch can be behind the local working tree. Read local files.

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

## The rebuild plan

Planned by the owner; none of it had been carried out when this was written.

1. **New branch from `cleanup-after-versioning`.**
2. **A frozen legacy strain beside the working one.** Every package and loose
   file gets a `legacy_*` copy with its imports pointing at the other legacy
   copies. A switch in the entry point runs either the working app or the legacy
   app, so legacy can be trained exactly as it was and its curves compared with
   the rebuild. Nothing under `legacy_*` is edited, and it is deleted when the
   rebuild is done.
3. **Rebuild the working strain file by file** from the skeleton and from this
   branch's code. Setting up the surrounding wiring by hand is part of the
   exercise: it is how the weak joints are meant to be found.

Things the legacy switch depends on:

- Only one strain is imported per process. Both register a Gymnasium env.
- A folder rename rewrites imports, not strings. The Gymnasium entry point and
  the package paths used to find config and assets must name the legacy folders.
- Both strains start with the same run-id scheme, so legacy runs need their own
  tag or output folder.
- Once the config schemas differ, a comparison only means something if both
  config files describe the same world. Compare several seeds per side.

## The reference skeleton

`src/syn_grid/reference/skeleton/` holds the intended structure of the scenario
layer: modules, classes, protocols and signatures, with no logic. It mirrors the
real paths (`skeleton/scenario/...`, `skeleton/core/digestion/...`). It lives
inside the package while it is being written and moves to a top-level
`reference/` on the new branch. Once agreed, **the skeleton is the working
truth** and is edited until it fits.

Conventions the owner asked for:

- Every function is a signature, a docstring saying what it does or composes,
  and a `...` body. Ruff's `PIE790` objects to that, so each file carries
  `# ruff: noqa: PIE790`.
- Imports use the real package paths (`syn_grid.scenario...`,
  `syn_grid.core.digestion...`), never `syn_grid.reference...`. They therefore
  resolve to the live modules, not to other skeleton files.
- Tables that are structure, such as `SCENARIOS`, are filled in.
- Do not invent Effect orbs or `Cancellation`. They have no implementation.

Written so far: `scenario/registry.py` only. It keeps the current signatures
wherever no change was agreed, which leaves these open on that file:

1. `orb_population` and `scoring_mode` as two parameters, or one per-kind recipe.
2. `max_score` on the family helper.
3. No negative-orb count anywhere in the helper or the world builder.
4. Builders take the base `ScenarioConf` and `cast` it, because the models table
   and the builders table are separate.
5. Whether the world readers belong in the registry or in a tier-chain file.

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

## This branch, as a reference

One scenario, `goal_tier_chain_spatial`, runs end to end. On 2026-10-08
`python -m syn_grid.check_env` passed and 20,000 random actions stayed inside
the declared `Box`. That says nothing about learning.

Local, unfinished state: `hud` and `metrics` are built in
`scenario/registry.py` but not passed to `Scenario`, whose two fields are
commented out. The environment still reads the tier digester directly.

How it is put together:

- **Config** is three YAMLs in `src/syn_grid/config/yaml/` (global, runner, and
  one named after the scenario), validated by frozen, strict pydantic models
  that forbid unknown keys. `SCENARIOS` in `scenario/registry.py` picks the
  scenario's schema and its builder. Shared models are in `config/models/`;
  scenario-specific ones are in a `config.py` beside what they configure, such
  as `scenario/goal/tier_chain/config.py`.
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

The owner reports that spatial runs on this branch but the agent does not learn
as it should. No agent has verified this, and the cause is unknown. Do not
assume the refactor caused it; it may predate it.

- Read `docs/dev/rppo-regression.md` before touching the reward path. Its
  original conclusion is superseded: the timeout and chain-break rewards used to
  be the same value, and are independent fields now.
- Before rebuilding, confirm that `cleanup-after-versioning` learns, and lock
  its behaviour with the replay scripts, which run there. A learning regression
  is silent: the env still runs and passes `check_env`.
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
