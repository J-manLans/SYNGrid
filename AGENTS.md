# AGENTS.md

Guidance for AI agents working in this repository. Last updated 2026-10-25.
Trust the code over this file, and say so when they disagree.

## Where things stand

**The repo is in rebuild mode, and the rebuild has started.** The scenario
layer is being rebuilt file by file on `rebuilding-scenario`: the owner and the
agent design each piece together, the agent writes it, and the owner reviews
it. `refactor-of-scenario-refactor` is the product of two
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
- `docs/dev/landscape-changes.md` is the changes that could move what the agent sees or is paid. Read it when a
  comparison with legacy shows a difference, and add to it when a change touches the reward, the observation or the
  layout of orbs.
- `docs/dev/glossary.md` is what the words mean. Use its terms, and add a term
  there when a new one is coined.
- `docs/dev/regressions.md` is for behaviour that worked before and does not
  now. It has a template and no entries.

State on 2026-10-13: the config models, the registry and the spatial YAML are
done. `scenario.py` and the blocks are written, and the spatial builder returns
a `Scenario`. The populations are in `scenario/blocks/populations/`: the tier-chain
one makes the spatial scenario's tier orbs, and the negative one is in the
bundle a config with a `neg_orb_conf` block gets. Its termination and its
world readers are written. `build_world` has its body, and on 2026-10-16
calling it on the spatial scenario returned a `GridWorld`; the world's `reset`
does not run yet. The other three builders raise.
`core/droid/digestion/digester.py` holds the digester protocol, and the
routing key `OrbKind` is in `core/orbs/orb_meta.py`. `engine.py` is the new
router, built from digesters. The tier digester's base and its three scoring
modes are written in `digesters/tier_chain/`, and the spatial builder names the
max-tier one through a `TierDigesterRecipe` in `scenario/blocks/recipes/`.
The negative digester is written in `digesters/neg_digester.py`, and the
negative bundle names it. Since 2026-10-23 `import syn_grid.app` works and
`python3 -m syn_grid.check_env` passes on the spatial scenario.

Phases 1 and 2 of the roadmap are done on the scenario side, and phase 3 is
done since 2026-10-19: the droid, the world and the orbs take pieces instead of
config.
The droid, step 5, is done since 2026-10-16, and so is the removal of
`orb_factory.py`, step 7. The recipe's body, step 8, has been written since
2026-10-14. The timer and the orbs, steps 1 to 4, are done since 2026-10-18;
the world, step 6, since 2026-10-19, which finishes phase 3. Phase 2
left four things written against a core that does not exist yet; the roadmap
lists them at the end of its phase 2.

Phase 5, the Gymnasium side, started on 2026-10-21, and the owner writes it by
hand. The roadmap's "State on 2026-10-23" under phase 5 says what is wired;
what is left of it is a human run to each ending. The
order of the later phases changed the same day: 5, 6, 8, 9, 7, 7b, 7c, 10.

State on 2026-10-25: an agent learned spatial through the rebuild. The owner
trained RPPO on 5x5, seed 3, to a mean reward of about 8.5 against a ceiling
of about 9.02, with episodes of about 15 steps; the roadmap's phase 6 has the
details and what is not done. The owner played spatial by hand, with and
without a negative orb, and a PPO run with a negative orb and the full view
(`vector_markovian`) learned too. The family was renamed `single_tier_chain`, and the
window work of the same day moved input out of the renderer, which is step 3
of phase 7b done early.

**No file in the working strain is final.** A file can be a stub, half-moved or
about to change shape, and its docstrings describe what it is meant to be, not
how far it has come. What is a stub, what is empty and what is still legacy is
recorded here and in the roadmap's table, and never in a docstring.

## How to work with the owner

- **Design together, the agent writes, the owner reviews.** Since 2026-10-13.
  Before that the owner wrote the code by hand and the agent only advised. A
  design is talked through first, and the owner decides. The agent then writes
  the code and the docs that go with it, and says what it changed so the owner
  can review it. Do not write plans or offer "which should I implement"
  choices.
- **When the direction is clear enough, go ahead and say what was done.** The
  owner welcomes that, also when their last message was not a plain "do it".
  Ask "shall I?" in one line only when the choice is genuinely open, or when
  the work creates or moves files and folders.
- **Audit and discuss before writing anything.** Say what exists, what is
  missing and what you would create, then wait.
- **Make the work reviewable.** After writing, list each file touched and what
  changed in it, say what was run and what was only read, and name anything
  written against code that does not exist yet. Code the owner has not read is
  not settled.
- **Agree on where a new file or folder goes before creating it.** Show the
  tree it would give and say what sort of thing lives at that level. The owner
  reads the structure to picture the design, and on 2026-10-12 and 13 the
  populations and the recipes were each moved several times because they were
  placed first and discussed after.
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

## The reference

The reference is the branch `refactor-of-scenario-refactor`, on `origin` and
locally. Read a file from it with
`git show refactor-of-scenario-refactor:src/syn_grid/<path>` or in a temporary
worktree. "Reading the reference" below lists its traps.

A `reference/` folder at the repo root held a copy of that package, the
skeleton of the scenario layer and the stubs patch. It was removed on
2026-10-23: the copy matched the branch apart from renames spilled from `src`,
and `src` had moved past the skeleton and the patch. It is in the history up
to `2b181e0`.

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
  engine by its class and reads its attributes. The engine raises when a
  digester is missing or a kind has two. Only a tier orb eaten out of
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
- Energy and score are two numbers, and the droid keeps both. `max_energy` is
  both the start and the cap, and replaces `starting_score` and `max_score`.
  The score is the episode's return. Every reward goes through the droid: its
  actions, digestion, and `on_termination`, which takes in what digestion is
  owed and what termination says the ending costs. Changed on
  2026-10-15, reversing the score in the environment from 2026-10-14.
  Since 2026-10-16 the ending's payout goes to the score alone and leaves
  energy as the episode ended it.
- The step clock lives in the world. `max_steps` is in `world_conf`, and
  `GridWorld` takes it and counts down `steps_left`. Termination, the HUD and
  the metrics read it from the world; the observation handler keeps no count
  since 2026-10-23.
- Digestion pays for orbs and termination pays for the clock. The tier digester
  pays the orbs, the chain break and the completed chain; termination pays
  `timeout_penalty`. There is no `completion_reward`.
- The world runs the orb field: it fills it at reset, takes orbs off it and
  brings back one that is ready. A scenario states only the field's size,
  `max_active_orbs` on its `WorldRecipe`. `SpawningRules` and its file are
  gone, and there is no hook for a scenario to react to an orb being eaten. An
  effect that belongs to an orb kind goes in that kind's digester.
- Delay tests long-term credit assignment: after a consume the other orbs
  vanish from the observation for `delay` steps and return in the same cells.
  Where that mechanic lives is open and left for phase 8.
- A goal scenario never truncates: running out of steps sets `terminated`.
  `EpisodeOutcome.truncated` defaults to false and is for a continuous
  scenario, where the clock is not a failure.
- Termination decides and changes nothing. It returns
  `EpisodeOutcome.ending_reward`, which the environment puts through
  `droid.on_termination`; `evaluate` takes no `reward`. What a digester is owed at the
  end (`undigested`, on the protocol, the gathered sum in threshold
  and 0 elsewhere) is settled by the droid. One termination class serves all
  three modes. The clock is charged only when it alone ended the episode.
- A truncated episode settles nothing: `on_termination` runs only on
  `terminated`, because a learner bootstraps the cut-off future from the last
  observation and a payout there would count what the droid holds twice.
  Since 2026-10-25.
- A tier orb is worth `base_reward * tier ** growth_factor`, set on the shared
  `TierOrbConf` and not rounded. Scaling dense has no config classes of its own.
- Legacy and `reproduction_package/` are references, not targets.
- Reward normalisation stays off in training (`norm_reward=False`).
- A tier chain holds one negative orb. `NegOrbConf` is `reward` and `cool_down`;
  a weight is only for a family whose field is sampled.
- An orb carries its own cool-down and lifespan, each a number or `None`. A
  cool-down of `None` is an orb that does not return in the episode, which is
  what a tier-chain tier orb has. The world asks every orb the same questions
  and no kind. The populations hand each orb both, `BaseOrb` acts on `None` in
  either (its timer never completes) and answers `has_expired()` and
  `is_ready()`. Since 2026-10-18.
- Every scenario in the single tier chain family is a single chain: the whole
  chain is laid out at once and left until it is eaten, whatever `max_tier` is.
  The family states once, in `_tier_bundle`, that its tier orbs have no
  cool-down and no lifespan, and no scenario in it can change that. The negative
  orb's lifespan is `_reachable_life_span(rows, cols)`, the longest walk between
  two cells.
- The family is named `single_tier_chain`: its folder, the
  `build_single_tier_chain_*` builders, `_single_tier_chain_scenario` and
  `SingleTierChainTermination`. The mechanic keeps `tier_chain`
  (`digesters/tier_chain/`, `TierChainPopulation`). The scenario names,
  `GOAL_TIER_CHAIN_*` and `goal_tier_chain_*.yaml`, are left to the naming pass
  in phase 10, since the scenario name is also the output folder. Since
  2026-10-25.
- A tier bundle's `make_digester` is a `TierDigesterRecipe`: a frozen dataclass
  of the scoring mode's class, `max_tier` and `chain_break_penalty`, in
  `scenario/blocks/recipes/`. It compares by value, which a `partial` does not. The
  folder is for every recipe: a frozen object a scenario holds that makes
  something with episode state when a world is built.
- A scenario's `build_world` is a `WorldRecipe`, shared by every family, in
  `scenario/blocks/recipes/`. It holds the world config, the bundles and the field's size
  rules, and reads only what the base `WorldConf` has. The full config is
  reachable as `scenario.build_world.world_conf`.
- A digester sets `self.KIND: Final[OrbKind]` in `__init__`. In the class body
  a `Final` does not satisfy the protocol under Pyright.
- A population says how many orbs it makes, through `count`. A fixed-field
  family sums the counts of its bundles for the field size, the observation
  slots and the sort limit. A sampled field takes its field size from the
  config, and the counts are then only the size of the pool.
- `scenario/` is `blocks/`, `families/`, `registry.py` and `scenario.py`.
  `blocks/` holds what no family owns, with the shared populations and recipes
  in folders of their own. `families/` holds the groups, `goal/` today, and
  under them the family folders.
- Populations are shared, in `scenario/blocks/populations/`, and take values, not a
  config. A family's builder states its own choices, as `_tier_bundle`
  does with the cool-down. They are not in `core/orbs/`: the world is handed
  its orbs, and what it owns is spawning.
- A locked scenario loads a small model holding only what a run may choose, and
  its builder makes the full config from its fixed values. The full models stay
  as they are, and the sandbox and every scenario that is not locked load them
  directly. Settled as the direction on 2026-10-12 and not applied: no scenario
  is locked until it has reproduced its legacy results.
- A locked scenario offers the perceptions that make sense for it. Spatial
  offers only ones with a limited view.
- The algorithm in the runner config is an `Algorithm` enum, and the `RUNNER`
  table is keyed by it, as `ScenarioName` and `SCENARIOS` are.
- The policy's name comes from the algorithm: the resolver picks it from the
  algorithm's `policy_aliases`, and no runner states a suffix or a variant. The
  table from a perception's leading word to a base policy stays.
- Human play is not an agent runner. `app.main` runs `HumanRunner` and returns
  before an agent runner is built, and human control overrides the snapshot and
  the runner config with no check.
- The renderer reads no input. `HumanRunner` reads its own keys and shows a
  summary when an episode ends. Agent runs with a human renderer share run
  controls in `BaseAgentRunner` (Space pauses, Escape or the close button
  quits), eval directly and training through `RunControlsCallback`. Quitting
  returns instead of raising: the run controls return whether the player quit,
  eval returns from its loop, and the callback returns False from `_on_step`,
  SB3's way to end `learn()`, after which training saves. `SYNGridEnv.close()`
  closes the window in each runner's `finally`. Since 2026-10-25.
- Episode state is first set in `reset`, not in `__init__`, so reading it
  before the first reset raises instead of running an episode that did not
  start where it should. The droid follows it; the other layers are aligned
  as the rebuild reaches them.

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
- **The runners run.** On 2026-10-13 `build_runner` made all three agent
  runners from the real configs. `HumanRunner` and `app.py` import since
  2026-10-23. On 2026-10-25 the owner trained PPO and RPPO on spatial 5x5,
  evaluated the PPO model, and played spatial by hand. FrameStackPPO was not
  run. Training with `csv_output` on was not run, so no CSV of a real run has
  been read.
- **`alg` in the runner config is an `Algorithm` enum,** in
  `config/models/runner_models.py`, and the `RUNNER` table in
  `runner_registry.py` is keyed by it. A wrong name fails when the config is
  loaded. The run id uses its `.value`.
- **The base SB3 runner names the policy.** `resolve_policy` takes the
  perception and the algorithm class and returns the one name in that
  algorithm's `policy_aliases` that starts with the perception's base policy.
  The concrete runners say nothing about the policy. The table from `vector`,
  `composite` and `grid` to `Mlp`, `MultiInput` and `Cnn` is kept on purpose,
  so perception names stay free of SB3's wording. Its `composite` and `grid`
  rows are not reachable while the `Perception` enum has only vector members.
- **`HumanRunner` is a plain class, not an agent runner.** With `human_control`
  set, `app.main` runs `HumanRunner(scenario).play()` and returns before an
  agent runner is built. Human control overrides the snapshot and the runner
  config without a warning, which the owner chose on 2026-10-13. `play()`
  plays one episode. It blocks on pygame's event queue for each move, takes
  the arrow keys through `_KEY_ACTIONS`, and at the end shows each HUD
  element, the metrics and the last reward until a key is pressed.
- **Run controls for agent runs with a human renderer** are in
  `BaseAgentRunner._handle_run_controls(renderer)`. Eval calls it before each
  step and puts the episode number in the window's title. Training calls it
  through `runners/agent_runners/sb3/run_controls_callback.py`, a debugging
  aid that is added only when `train_conf.render_mode` is `human`. A paused
  run is stopped, not training in the background.
- **The training sound** is played by `BaseAgentRunner` when training stops.
  A machine without an audio device prints a line and skips it.
- **`runners/agent_runners/utils/extractors.py` is not imported by anything.**
  It hardcodes two policy names and reads observation keys from before the
  rebuild, and Pyright reports 7 errors in it.
- **`docs/architecture/`** describes the code as it stood on 2026-10-11,
  including the parts that are still from before the rebuild and do not import.
  The runner parts were brought up to 2026-10-13: `01_runners.md`, the runner
  tree in `00_file_structure.md` and the startup flow in `07_runtime_flow.md`.
  The family's folder name and the runner and renderer changes of 2026-10-25
  were brought in the same day.
  It does not describe the intended design; that is in the roadmap and
  `design-decisions.md`.
- **Scenario config models** are in `config/models/scenario/`: `common_models.py`
  and `orb_models.py`. A family's own models are in its `config.py`.
- **`core/droid/digestion/engine.py` is the new router.** The droid builds it
  from the digesters it is handed. Nothing outside the droid reads it; a
  reader takes a digester through `droid.get_digester(...)`.
- **`gymnasium/utils/episode_termination.py` is gone,** deleted on 2026-10-16
  in `403e076` together with `core/orbs/orb_factory.py`. The environment asks
  `scenario.termination` since 2026-10-23.
- **The renderer takes `grid_dimensions`** and loads its images from a
  `_GRAPHIC_RESOURCES` table, brought back on 2026-10-21 from `87684a8` on
  `refactor-of-scenario-refactor`. Its `render` keeps the old signature, and
  the environment builds the old lists from `orbs_on_field()`. On 2026-10-21 it
  drew off-screen frames from a real spatial world on 5x5, 5x7 and 7x5 grids.
  Since 2026-10-25 it reads no input. Besides `render` it has `draw_summary`,
  `draw_pause`, `set_title` and `close`, and the owner has run it in human
  mode.
- **Only the two perceptions in the `Perception` enum are in `src`** since
  2026-10-23: `vector_markovian` and `vector_fog_of_war`. The other five are
  in `legacy/` and are rewritten from there in phase 7. `BasePerception` keeps
  only what the two use: the position bounds (`rows - 1`, `cols - 1`), the slot
  count, the sort limit and `max_identity`. Neither writes an orb's timer. The
  observation handler imports, and on 2026-10-23 both perceptions stayed inside
  their `Box` on a real spatial world on 5x5, 5x7 and 7x5 grids.
- **`SYNGridEnv` takes a `Scenario`** since 2026-10-21. Its `__init__` builds
  the world, the renderer and the observation handler from it, and keeps its
  `hud`, `termination` and `metrics`. Since 2026-10-23 `step` takes the
  droid's reward, asks `termination.evaluate(world)`, adds
  `droid.on_termination(outcome.ending_reward)` to the reward when the episode
  is terminated, and at the end of an episode puts the metrics, by name, in
  `info` under `SYN_STATS_KEY`. Gymnasium's `check_env` and the repo's
  `check_env` pass. Over 663 random episodes the summed rewards equalled the
  droid's score in every one. Since 2026-10-25 its `close()` closes the
  renderer's window. The owner has run it in human mode.
- **Two scenarios built from the same config compare equal,** since
  `build_world` became a `WorldRecipe` on 2026-10-12.
- **`WorldRecipe.__call__` has its body since 2026-10-14,** written before the
  droid it calls. It joins the bundles' orbs, makes one digester per bundle,
  builds a `SynergyDroid` from the digesters, the grid and `max_energy`,
  `step_penalty` and `boundary_penalty` as plain values, and returns
  `GridWorld(grid, max_active_orbs, max_steps, orbs, droid)`. The droid's
  signature is the one the droid took in step 5 of phase 3. The droid and
  the world are imported at module level since 2026-10-16, and
  `import syn_grid.scenario.registry` still works. On 2026-10-16 and again on
  2026-10-19 calling it on the spatial scenario returned a fresh `GridWorld`
  with a droid that resets.
- **The spatial builder's bundles are complete, and `build_world` joins
  them.** The tier bundle makes the orbs with their rewards and a max-tier
  digester from the config.
- **`SingleTierChainTermination.evaluate` reads `world.droid.energy` and calls
  `world.droid.get_digester(TierDigester)`,** both on the droid since
  2026-10-16. It was run with a stand-in world around real digesters, before
  the droid had `get_digester`. Since 2026-10-23 `evaluate` takes only the
  world and reads `world.steps_left`, in the protocol and in the class.
- **The droid's `on_termination` is called by the environment's `step`**
  since 2026-10-23, and returns what it adds to the last step's reward. Since
  2026-10-16 it adds to the score and not to energy. Earlier a script ran a
  threshold episode ending on the clock through the real droid, digesters and
  termination: the score equalled the summed rewards. Only max-tier has run
  through the environment.
- **A config with a `neg_orb_conf` block builds a `Scenario`.** It gets the
  negative bundle, the `_Neg` tag, a `negatives_eaten` metric and one more orb
  in the field size, the observation slots and the sort limit. On 2026-10-25
  the owner played it by hand: the negative orb spawned, despawned and
  returned as planned. The same day a PPO run on spatial 5x5 with
  `vector_markovian` and a negative orb learned, somewhat slower than without
  one. CSV was off, so `negatives_eaten` over the run was not read.
- **The world readers in the single tier chain builders** each return a value from
  `world.droid.get_digester(...)`. Since 2026-10-23 they run against the real
  droid, through the HUD and the metrics the environment reads. Pyright was not
  run after `get_digester` was added.
- **The HUD is read through `scenario.hud`** since 2026-10-21. The shared
  readers `energy`, `score` and `steps_left` are in `scenario/blocks/readers.py`
  since 2026-10-23, and the single tier chain
  helper lists energy, score, steps left and the chain. On 2026-10-21 every
  element was read from a real spatial world after three steps. The
  environment's `get_hud_data()` returns the values by label, for the renderer
  and the human runner's summary. The renderer is
  handed `scenario.hud` when it is built, fills each bar against its
  `HudBar.maximum`, and picks energy, steps left and the chain by label, so the
  score element is read and not drawn. On 2026-10-22 it drew off-screen frames
  from a real spatial world over two episodes.
- **The CSV logger writes the scenario's metrics** since 2026-10-23.
  `CSVEpisodeLogger` takes the metric names, and its header is `LogKey`
  (`r`, `l`, `t`, from `RecordEpisodeStatistics`) followed by them. `LogKey`
  holds only those three. On 2026-10-23 it wrote a CSV through the real
  environment for spatial, and with a negative orb, which adds
  `negatives_eaten`. `plot/plot_utils.py` still names the chain members that
  were removed from `LogKey`.
- **The engine and the digesters were only tried in throwaway scripts.** No
  test covers them. The negative digester was fed a stand-in object, not a
  real negative orb.
- **`GridWorld` is rebuilt since 2026-10-19.** The constructor takes the grid,
  `max_active_orbs`, `max_steps`, the orbs and the droid. `reset(rng)` restarts
  the step clock, resets the droid and the orbs, and fills the field.
  `perform_droid_action` is the droid's `take_step` and what follows from the
  field: every orb ticks, the droid eats the orb it stands on (before an expiry
  is checked, so it wins an orb on its last tick), expired orbs leave, and one
  ready orb comes back in a random free cell if there is room. The world keeps
  no list of orbs and asks each one `is_active`, `has_expired()` and
  `is_ready()`. `orbs_on_field()` replaces the three `get_orb_*` getters. It was run in
  throwaway scripts on the spatial scenario, with and without a negative orb,
  and no test covers it. `scenario.build_world()` gives a world that resets and
  steps.
- **Two delay methods are parked in the world.** `_deactivate_all_orbs` and
  `_reactivate_all_orbs` are used by no one and still read `self._conf.delay`,
  which is gone, so Pyright reports one error there. A `# TODO(delay)` says
  they move into the delay scenario (phase 8).
- **`scenario/families/continuous/` is an empty package.** It holds only an
  `__init__.py`. No continuous family is written.
- **Tests import the legacy strain.** On 2026-10-11 collection found 230 tests
  and no errors. They were not run.
- **`scripts/replay_probe.py` imports the legacy strain** on this branch. It was
  not run.

## The learning problem

The owner reports that spatial runs on `refactor-of-scenario-refactor` but the
agent does not learn as it should. No agent has verified this, and the cause is
unknown. Do not assume the refactor caused it; it may predate it. On this
branch an RPPO run learned spatial on 2026-10-25 (see "State on 2026-10-25"
above), so the problem is not in the rebuild as it stands, on that one seed.

- `docs/dev/rppo-regression.md` is not on this branch. Read it with
  `git show cleanup-after-versioning:docs/dev/rppo-regression.md` before
  touching the reward path. Its original conclusion is superseded: the timeout
  and chain-break rewards used to be the same value, and are independent fields
  now.
- Phase 0 of the roadmap comes first: confirm that the legacy strain learns
  spatial over several seeds, and record a step-by-step trace with the replay
  scripts. A learning regression is silent: the env still runs and passes
  `check_env`. On 2026-10-13 the owner reported that a legacy spatial run
  learned, and that most stateless PPO scenarios were verified before it. The
  git-ignored `output/` folder holds one run from 2026-10-08: spatial 5x5,
  RPPO, `vector_fog_of_war`, seed 3. No agent looked at its curves. More seeds
  can be run with `USE_LEGACY`; no step-by-step trace is recorded.
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
the first one is dealt with.

- **`GridWorld.reset` needs an rng.** Since 2026-10-19 it has no unseeded
  fallback. Pass `np.random.default_rng(seed)`, or let the environment pass its
  seeded `np_random`.
- **Every perception returns the same buffer each step,** so `obs_t is
  obs_{t+1}`. Anything that keeps two observations must copy.
- **`reproduction_package/` is committed and large.** Do not add to it. It is a
  record of what the thesis runs did, not something the rebuild must reproduce.
- **`plot/` is a thesis scratch module** with hardcoded paths evaluated at
  import, and needs the optional `extra` dependencies.
- **`docs/code_style.md` mandates Black;** the repo uses ruff.
- **A pygame window and a model load crash together on this machine.** The
  display runs on the AMD iGPU, so opening a pygame window loads Mesa's
  `libLLVM-15`, and Triton's own LLVM then segfaults when SB3 loads a model
  (Adam imports `torch._dynamo`, which imports Triton). It hits any run that
  opens a window before it creates or loads a model: eval with a human or
  `rgb_array` renderer, and training with one. Start through `scripts/run_app.sh` or the
  `launch.json` entry, which run the window on the NVIDIA card, or set
  `SDL_FRAMEBUFFER_ACCELERATION=0`. The investigation is in
  `docs/scratch_pad.md`.
