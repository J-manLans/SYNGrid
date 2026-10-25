# Scenario Rebuild Roadmap

The order to rebuild the scenario layer in, written 2026-10-10. Each phase says what it produces, which files it
touches, what has to be decided first and how to tell it is done. The reasons behind the decisions are in
`design-decisions.md`; this file is only the order of work.

The route is a vertical slice: get `goal_tier_chain_spatial` running end to end on the new structure, check it against
legacy as a reference, and only then widen to the other scenarios. Until phase 5 nothing runs, so each phase before it has its own
small check.

**Order of work since 2026-10-21:** phase 5, 6, 8, 9, 7, 7b, 7c and then 10. The phase numbers are kept so that the
references to them stay valid, and they are renumbered in the close-out pass. The observation pass (7) and the
renderer rework (7b) come late on purpose. The legacy check (6) then runs while the observation is still the legacy
one, so a difference there comes from the environment or the world, and the observation pass is the only change in
its step. The renderer rework comes after the observation pass because its fog overlay needs the view radius that
pass adds. The modular HUD (7c) comes after the rework, since it changes the look that 7b is checked against.

---

## Where things stand

| Part | State |
|---|---|
| Config models, registry, spatial YAML | done; the YAML validates. Orb models are in `config/models/scenario/orb_models.py` |
| `scenario/scenario.py`, `blocks/hud.py`, `blocks/metrics.py`, `blocks/orb_bundle.py` | written, from the skeleton |
| `blocks/observation.py`, `termination.py`, `blocks/populations/population.py` | written: the rules dataclass and the two protocols. A population has `create` and `count` |
| `blocks/populations/tier_chain_population.py` | written: one tier orb per tier, with its reward and the cool-down it is handed. Its count is `max_tier` |
| `blocks/populations/neg_population.py` | written: one negative orb, used by the negative bundle. Its count is 1. It hands the orb a lifespan, which the builder states |
| `families/goal/single_tier_chain/termination.py` | written: the four endings, what the chain is owed, and the timeout penalty. It reads `energy` and `get_digester(...)` from the droid |
| `families/goal/single_tier_chain/builders.py` | the spatial builder returns a `Scenario` whose tier bundle makes max-tier digesters, with a negative bundle and one more orb in its counts when the config has a `neg_orb_conf` block; the other three builders raise `NotImplementedError` |
| `blocks/recipes/world_recipe.py` | written: `WorldRecipe`, what a scenario holds as `build_world`, shared by every family. Its body builds the droid and the world, and returns a `GridWorld` for the spatial scenario since 2026-10-16 |
| `blocks/recipes/tier_digester_recipe.py` | written: `TierDigesterRecipe`, what a tier bundle holds to make its digester |
| Tier-chain config | in line with the digesters since 2026-10-11: no `completion_reward`, the reward ladder on the shared `TierOrbConf`, no dense classes |
| `core/droid/digestion/digester.py` | written: the `OrbDigester` protocol (`KIND`, `reset`, `digest` returning a float, `undigested`). The routing key, `OrbKind`, is in `orb_meta.py` |
| `core/droid/digestion/digesters/tier_chain/` | written: the `TierDigester` base and the max-tier, step-wise and threshold modes, each with `undigested`. The spatial builder names the max-tier one |
| `core/droid/digestion/digesters/neg_digester.py` | written: `NegDigester` pays the orb's own reward, counts `negatives_eaten` and is owed nothing at the end. The negative bundle names it |
| `core/orbs/orb_meta.py` | stores kind and tier; the radix identity is unchanged |
| `core/droid/digestion/engine.py` | written: a router with `reset`, `digest`, `get` and `undigested`, built from digesters. The droid builds it from the digesters it is handed |
| `core/droid/synergy_droid.py` | written (step 5 of phase 3, done 2026-10-16): takes its digesters, the grid and `max_energy`, `step_penalty` and `boundary_penalty` as values, builds its engine and keeps it private, and holds energy and score, which are first set in `reset`. Every reward goes through it: `on_termination(ending_reward)` takes in what digestion has left undigested and what the ending costs or pays, adds them to the score and leaves energy alone. `get_digester(...)` gives a reader a digester by class |
| `core/grid_world.py` | written (step 6 of phase 3, done 2026-10-19): the constructor takes the grid, `max_active_orbs`, `max_steps`, the orbs and the droid, and the world keeps the step clock as `steps_left`. `reset` takes an rng and fills the field. `perform_droid_action` is the droid's `take_step` plus what follows from the field: orbs tick, the droid eats one it stands on, expired orbs go, and one ready orb comes back if there is room. The world keeps no list of orbs and asks each one. `orbs_on_field()` replaces the three getters, so `environment.py`'s `render` still calls getters that are gone. Two delay methods are parked in it with a `# TODO(delay)` and still read `self._conf`. It was run in throwaway scripts on the spatial scenario, with and without a negative orb |
| `core/utils/timer.py`, `core/orbs/base_orb.py`, `tier_orb.py`, `negative_orb.py` | written (steps 1 to 3 of phase 3, done 2026-10-18): a timer set to `None` never completes. An orb takes its own cool-down and lifespan, each a number or `None`, and answers `has_expired()` and `is_ready()`. The class-level `_life_span` is gone. The world asks them since 2026-10-19 |
| `scenario/blocks/hud.py`, `scenario/blocks/readers.py` | the element types, since 2026-10-21, and the shared readers `energy`, `score` and `steps_left`, in their own file since 2026-10-23. The single tier chain helper lists energy, score, steps left and the chain |
| `gymnasium/`, `rendering/` | phase 5 started on 2026-10-21; see its state below. The environment imports and `check_env` passes on the spatial scenario since 2026-10-23. The CSV logger writes the scenario's metrics. Since 2026-10-25 the renderer reads no input: it draws a frame, an episode summary and the pause screen, sets the window's title, and is closed through the environment's `close()` |
| Runners, `env_factory.py` | take a `Scenario`. Training and eval ran on the spatial scenario on 2026-10-25; see phase 6. `HumanRunner` reads its own keys and shows a summary when an episode ends. Eval and training with a human renderer share run controls in `BaseAgentRunner`, training through `RunControlsCallback` |
| Legacy strain | in `src/syn_grid/legacy/`, switched by `USE_LEGACY` in `__main__.py` |

`import syn_grid.scenario.registry` works and `build_scenario` returns the spatial `Scenario`. Two scenarios built
from the same config compare equal. `scenario.build_world()` returns a `GridWorld` since 2026-10-16, which resets and steps since 2026-10-19. `import syn_grid.app` and
`python3 -m syn_grid.check_env` work since 2026-10-23, and an agent learned spatial through the rebuild on 2026-10-25.

---

## Phase 0: A baseline to compare against

Do this first, or a later mismatch can't be told apart from a problem that was already there. Legacy is a reference
and not a target: the rebuild's rewards differ from it on purpose, so the curves are what gets compared.

- Confirm the legacy strain learns spatial, with several seeds.
- Record its behaviour with the replay scripts in `scripts/`: one seed, a fixed action sequence, the reward and the
  observation per step. Use a `starting_score` high enough that episodes reach the timeout.

**Done when:** there are legacy training curves to hold the rebuild's against in phase 6, and a recorded
step-by-step trace whose observations can be diffed.

**State on 2026-10-13:** partly done. The owner reports that a legacy spatial run learned. The git-ignored `output/`
folder holds one run from 2026-10-08: spatial 5x5, RPPO, `vector_fog_of_war`, seed 3. More seeds are a matter of
switching `USE_LEGACY` and running. No step-by-step trace is recorded.

---

## Phase 1: Digestion

**Produces:** something that turns a consumed orb into a reward, with no knowledge of scenarios.

**Files:** `core/droid/digestion/digester.py`, `engine.py`, and `digesters/tier_chain/`, which holds the tier base
and one file per scoring mode.

**Settled on 2026-10-11:**

- The droid creates the engine. `build_world` makes the digesters, one per `OrbBundle`, and hands them to the droid.
- No events, no tally and no cross-kind hook. `digest` returns a float, and a reader takes what it needs from the
  digester itself.
- A tier orb is worth the same in every scoring mode, and the mode decides when it is paid.

**Settled on 2026-10-12:** the engine raises when two digesters own one kind, when an eaten orb's kind has no
digester, and when `get` is asked for a class that is not registered.

**Write:**

1. `digester.py`: the `OrbDigester` protocol with `KIND`, `reset` and `digest`. The routing key is the orb's kind,
   `OrbKind` in `orb_meta.py`. Written.
2. `digesters/tier_chain/`: the tier base holding the chain, the counts and the chain's summed reward, and the
   max-tier, step-wise and threshold modes. Written.
3. `engine.py`: a router with `reset`, `digest` and `get`. Written.

**Done when:** `import syn_grid.scenario.blocks.orb_bundle` works, the engine routes an orb to the digester of its
kind, and a max-tier digester fed tiers 1, 2, 3 reports progress, progress, completion, and fed a wrong tier pays
`chain_break_penalty`. All of it was seen through the engine in a throwaway script on 2026-10-12, together with the
three cases that raise; nothing in the test suite covers it.

---

## Phase 2: The blocks the spatial builder names

**Produces:** the pieces a `Scenario` is made of, for the single tier chain family.

**Files:** `scenario/blocks/termination.py`, `observation.py`; `scenario/blocks/populations/`;
 `scenario/families/goal/single_tier_chain/termination.py`.

**Write:**

1. Population: the protocol and the tier-chain one, which makes one tier orb per tier, both in `blocks/populations/`.
   Written.
2. The orb field: the whole chain on the field from the first step, no tier orb expires, and an eaten tier orb does
   not return in that episode. The last one is a requirement and not a detail: with one orb per tier it means a
   chain can never be broken with a tier 1, which the scoring rules count on in a goal scenario. The scenario's side
   is one value, `max_active_orbs` on its `WorldRecipe`, and it is written. The rest is the orbs' and the world's,
   in phase 3. `blocks/spawning.py` is deleted.
3. Termination: the outcome type and the protocol in `blocks/`; the tier-chain strategy in its own file. It ends on
   energy 0, a broken chain, a completed chain or the clock. It returns only what the ending adds,
   `timeout_penalty` when the clock alone ended the episode, and changes nothing. What a chain is owed is settled by
   the droid since 2026-10-15. Nothing for a completed chain. Written, and checked against a stand-in world.
4. Observation rules: only what the current perception needs for now. The full shape waits for phase 7.

**Done when:** each file imports on its own, and the termination strategy gives the right outcome for a hand-built
world state in each of its four endings.

**Done on 2026-10-13.** Each file was imported on its own, and the termination was run against a stand-in world
around real digesters in all three scoring modes. No test covers any of it. What the phase wrote against things that
do not exist yet, and so hands to phase 3:

- `SingleTierChainTermination` reads `world.droid.energy` and calls `world.droid.get_digester(...)`. The droid has
  neither, so a type checker reports both.
- `TierChainPopulation` gives its tier orbs a cool-down of `None`, and `BaseOrb` does not act on it. No orb is
  handed a lifespan.
- `WorldRecipe` holds `max_active_orbs`, and its body raises.
- The five world readers in the single tier chain builders call `world.droid.get_digester(...)`. They were written on
  2026-10-13, after the phase was closed, as the part of phase 4 that could be done early.

---

## Phase 3: Core takes pieces, not config

**Produces:** a world, a droid and orbs that are handed what they need.

**Files:** `core/droid/synergy_droid.py`, `core/grid_world.py`, `core/orbs/`, `core/utils/timer.py`; on the scenario
side `blocks/populations/`, `blocks/recipes/world_recipe.py` and the single tier chain builders.

**Write:**

1. Droid: energy and score as two numbers; takes its digesters and the grid size. It keeps its engine to itself and
   gives a reader a digester by class through `get_digester(...)`. `SingleTierChainTermination` already reads `energy` and
   calls `get_digester`.
2. World: takes the grid, `max_active_orbs`, `max_steps`, the orbs and the droid; counts the steps of an episode. It fills the
   field at reset and brings back an orb that is ready when there is room. When it counts,
   `steps_left` leaves the signature of `TerminationRules.evaluate`.
   Each step it asks its orbs whether they are ready or expired. An orb does not notify the world.
3. Orbs: the tier orb loses its scoring flags. `orb_factory.py` goes, replaced by the population. An orb carries
   its own cool-down and lifespan, each a number or `None`: `BaseOrb` says whether it is ready to return and
   whether it expires, and the world asks every orb the same two questions.

**Order of work**, set on 2026-10-13. Each step needs only the ones before it.

| # | File | What changes |
|---|---|---|
| 1 | `core/utils/timer.py` | only if needed, so a timer can stand for "never" when a cool-down or lifespan is `None`. Done 2026-10-18 |
| 2 | `core/orbs/base_orb.py` | takes a lifespan per orb, acts on a cool-down of `None`, and says whether it is ready to return and whether it has expired. The class-level `_life_span` goes. Done 2026-10-18 |
| 3 | `core/orbs/synergy/chain/tier_orb.py`, `core/orbs/direct/negative_orb.py` | pass the lifespan through. Done 2026-10-18 |
| 4 | `scenario/blocks/populations/` | hand each orb its lifespan: `None` for tier orbs, a number for the negative orb. Done 2026-10-18 |
| 5 | `core/droid/synergy_droid.py` | takes digesters and the grid size as values, makes its own engine, splits energy from score, gets `on_termination`, and gets `get_digester`. Done 2026-10-16 |
| 6 | `core/grid_world.py` | takes the grid, `max_active_orbs`, `max_steps`, the orbs and the droid, counts the steps, fills the field at reset, and asks its orbs each step |
| 7 | `core/orbs/orb_factory.py` | deleted, once the world no longer imports it. Done 2026-10-16, in `403e076` |
| 8 | `scenario/blocks/recipes/world_recipe.py` | the body of `__call__`, the last piece of phase 4. Done 2026-10-14, written before the droid; run on the spatial scenario on 2026-10-16 and 2026-10-19, where it returns a fresh `GridWorld` per call |
| 9 | `scenario/families/goal/single_tier_chain/builders.py` | pass the negative orb's lifespan to its population, since it follows from the grid size. Done 2026-10-18, with `_reachable_life_span` |

- The orbs come first because they import nothing that is broken, so they can be tried alone.
- The droid comes before the world because the world is handed a droid. With `energy` and `get_digester` on the
  droid, the type errors in the single tier chain termination and builders go away.
- The recipe's body comes last because it calls the constructors that steps 5 and 6 define.
- At step 6, `steps_left` can leave `TerminationRules.evaluate`, and the open question of orbs or populations is
  settled. With a lifespan per orb from step 2, a list of orbs is enough.
- `DroidAction` is in `core/droid/droid_action.py` since 2026-10-16, so core imports nothing from `gymnasium/`.

**Check while here** (from the "things that bite" list in `AGENTS.md`):

- `BaseOrb._life_span` was a class attribute shared by every env in the process. Step 2 removed it on 2026-10-18.
  No perception in `src` reads it since 2026-10-23. A perception that writes an orb's timer in phase 7 has to find
  the maximum lifespan elsewhere.
- `GridWorld.reset()` without an rng re-rolled the layout. It takes a required rng since 2026-10-19.

**Done when:** a world can be built by hand in a script, reset with a seed, and stepped with actions.

**State on 2026-10-16.** Step 5, the droid, is done. It has `get_digester(...)`, keeps its
engine private as `_digestion_engine`, has its docstrings, and is formatted. Its episode state is first set in
`reset`; see "Episode state is first set in `reset`" in `design-decisions.md`. `on_termination(ending_reward)` adds
what digestion has left undigested and what the ending costs or pays to the score, and leaves energy alone. The
digesters' names follow the digestion metaphor: `undigested()`, `chain_worth` and `chain_tier`.

`WorldRecipe.__call__` returns a `GridWorld` for the spatial scenario, with a droid that resets.

**State on 2026-10-19: phase 3 is done.** Steps 5, 7 and 8 were checked and are done, as the table says. Step 6 was
written in five small commits: the world asks its orbs instead of keeping two lists, `orbs_on_field()` replaces the
three getters, the droid's `perform_action` is `take_step`, `reset` requires an rng, and the delay methods and the
identity remap carry TODOs. Left over from it: the environment's `render` calls the old getters (phase 9), the
delay methods move into the delay scenario (phase 8), and the identity remap stays until phase 7.

**State on 2026-10-18.** Steps 1 to 4 are done, together with the part of step 9 that hands the negative orb its
lifespan. `Timer` takes `None` as never completing. `BaseOrb` takes a lifespan beside the cool-down, and answers
`has_expired()` and `is_ready()`. `TierOrb` and `NegativeOrb` pass the lifespan through, and both populations take it
as a field. The single tier chain builders state it: `_tier_bundle` hands `None` to the tier orbs beside their `None` cool-down, since
every scenario in the family lays the whole chain out at once, and `_negative_bundle` gets
`_reachable_life_span(rows, cols)`. Next is step 6, the world: it still ticks
the timers and tests `is_completed()` itself, and should ask `has_expired()` and `is_ready()` instead.

Done on 2026-10-15: the score is back on the droid, beside energy, and every reward goes through it. Termination
returns only `EpisodeOutcome.ending_reward` and its `evaluate` takes no `reward`.

For phase 5: the step is the droid acting, `termination.evaluate(world, steps_left)`, and if the episode ended
`world.droid.on_termination(outcome.ending_reward)` when `outcome.terminated`. The step's reward is the sum of the two.
The observation, the HUD and the episode info are read after that, and read the score from the droid.

---

## Phase 4: The builders

**Produces:** a `Scenario` for spatial.

**Files:** `scenario/families/goal/single_tier_chain/builders.py`.

**Done on 2026-10-11:** the config is in line with the digesters. The reward ladder is one formula on the shared
`TierOrbConf`, `completion_reward` is gone from the models, the termination, the builder and the spatial YAML, and
dense has no config classes of its own.

**Write:**

1. `build_single_tier_chain_spatial`: the tag and the tier bundle (tier orbs with their rewards, and a
   `TierDigesterRecipe` that makes a max-tier digester from `max_tier` and `chain_break_penalty`). Written.
2. `_single_tier_chain_scenario`: what the family shares.
3. The body of `WorldRecipe.__call__` in `scenario/blocks/recipes/world_recipe.py`. The recipe itself is written,
   and the family helper holds one as `build_world`. The body calls the droid's and the world's constructors, so it
   is written last in phase 3.
4. The world readers, which take their numbers from the digesters through `world.droid.get_digester(...)`. Written
   on 2026-10-13 and read from a stand-in world.

**Done when:** `import syn_grid.scenario.registry` works, `build_scenario` returns a `Scenario`, two scenarios built
from the same config compare equal, and `scenario.build_world()` returns a new world on each call.

---

## Phase 5: The Gymnasium side

**Produces:** an environment that runs a `Scenario`, headless and with a human at the keys, on the existing renderer
and the existing perceptions with the smallest changes that make them run. The rework of both comes in phases 7 and 7b.

**Files:** `gymnasium/environment.py`, `gymnasium/observation_space/`, `gymnasium/utils/episode_termination.py`
(removed), `rendering/pygame_renderer.py`, `core/grid_world.py`.

**Settled:** where the renderer's settings come from. `RendererConf` goes. The renderer's images become constants in
the renderer, the grid size comes from the world, and the frame rate from the environment's metadata. See "The
renderer takes plain values, and the HUD is the scenario's list" in `design-decisions.md`.

**Write:**

1. The world gets a public, read-only `grid_dimensions`.
2. The environment takes the scenario, builds its world from it, and asks `scenario.termination` when an episode
   ends. The steps left come from the world. The TODO in `step` has the order.
3. Remove `episode_termination.py`.
4. The HUD and the episode info read from `scenario.hud` and `scenario.metrics`. The scenario's `hud` lists energy,
   score and steps left beside the chain, through the shared readers in `scenario/blocks/readers.py`. The renderer keeps
   its hand-placed look and takes a dictionary of values, which the environment fills from the HUD elements by label.
5. The renderer takes the grid size as plain values and the environment hands it the orbs on the field, from
   `orbs_on_field()`.
6. The perceptions get the smallest change that makes them run: a tier orb's timer is `None`, and
   `base_perception.py` reads `BaseOrb._life_span`. Before patching, look at what legacy wrote in that slot for a tier
   orb that never ticks, and write the same, so the patch is not a difference at the legacy check.

**Done when:** `python3 -m syn_grid.check_env` passes, random actions stay inside the observation space, and a human
run plays an episode to each of its endings.

**State on 2026-10-23.** The owner writes this phase by hand. What is left is the human run to each ending.

- Step 1 is done: the world has a read-only `grid_dimensions`, `(rows, cols)`, and each caller unpacks it.
- Step 2 is done since 2026-10-23. `__init__` takes the scenario, builds the world, the renderer and the observation
  handler, and keeps its HUD, termination and metrics. `step` asks `termination.evaluate(world)`, which reads the
  steps left from the world, and adds what `droid.on_termination` settles to the last step's reward.
  `python3 -m syn_grid.check_env` passes, and over 663 random episodes the summed rewards equalled the score.
- Step 3 was done on 2026-10-16: `episode_termination.py` went in `403e076`.
- Step 4 is done for the HUD. `scenario/blocks/readers.py` has the shared readers `energy`, `score` and `steps_left`, and
  the single tier chain helper lists energy, score, steps left and the chain. `get_hud_data()` returns every element's value
  under its label. The renderer is handed `scenario.hud` when it is built and keeps each bar's maximum by label, so the
  energy bar fills against `max_energy` and the steps bar against `max_steps`. `_draw_hud` picks `energy`, `steps
  left` and `current tier chain` by label, under a TODO for 7c. The score element is read and not drawn until 7c, and
  neither is a new element. Since 2026-10-23 the episode info is the scenario's metrics by name, read after the droid
  is settled, and the CSV logger takes its columns from the metric names.
- Step 5 is done. The renderer takes `grid_dimensions` and loads its images from a `_GRAPHIC_RESOURCES` table, taken
  from `87684a8` on `refactor-of-scenario-refactor`. `render` keeps its old signature, and the environment builds
  the old lists from `orbs_on_field()`, all of them active.
- Step 6 is done for the two perceptions in the enum. The handler takes `scenario.observation`, `grid_dimensions`
  and `max_identity`, and `PERCEPTIONS` is keyed by `Perception` and holds only those two. `BasePerception` takes the
  same three and keeps the position bounds, `observation_slot_count`, `sort_limit` and `max_identity`. Neither
  perception writes the timer, so the legacy question about the lifespan slot is not on their path. On 2026-10-23
  the other five perceptions and the helpers only they used were deleted; their copies are in `legacy/`. The
  position bounds became exact (`rows - 1`, `cols - 1`), the handler stopped counting steps and `get_observation`
  takes only the world. `ObservationRules` lost `max_steps` and `max_tier`, which nothing read. On 2026-10-23 both
  perceptions stayed inside their `Box` over 3,000 random steps on 5x5, 5x7 and 7x5 spatial worlds.

**State on 2026-10-25.** The owner played the spatial scenario by hand, with and without a negative orb. When an
episode ends, `HumanRunner` shows a summary over the last frame: each HUD element, the scenario's metrics and the last
reward, so an ending's payout can be read off the screen. Which endings were reached by hand is not recorded here.
The input work done the same day is step 3 of phase 7b; see "Runners own the window's input, the renderer draws, and
the env closes the window" in `design-decisions.md`.

---

## Phase 6: Check against legacy

**Produces:** evidence that the rebuilt spatial scenario learns at least as well as the baseline. It runs after
phase 5 and before the observation pass.

1. Replay the phase 0 action sequence through the rebuild and diff the observation per step. Read the size of a
   difference before calling it behavioural.
2. Train several seeds on each side and compare the curves.

**Known differences to account for:** anything that changed on purpose is in `design-decisions.md`, and the changes
that could move the observation or the reward are in `landscape-changes.md`. The rewards are
expected to differ: a completed max-tier chain pays what its orbs are worth, where legacy paid a fixed 10. Set the
base reward so the two are of similar size before comparing curves.

**Done when:** the observation traces match, or every difference is explained, and the rebuild's curves are no worse
than legacy's within seed noise. If legacy did not learn in phase 0, this phase shows the same problem and not a new
one.

**State on 2026-10-25: a first result.** The owner trained RPPO on spatial 5x5 with `vector_fog_of_war`, seed 3. The
mean reward reached about 8.5, against a ceiling of about 9.02 (tiers 1 to 3 worth `1 + 2^1.5 + 3^1.5`, less a step
penalty of 0.001 a step), with episodes of about 15 steps. The owner reads it as the rebuild reaching what legacy
reached. Legacy's ceiling is not the same number, so a comparison is of each side's share of its own ceiling and of
how fast the curves climb, not of the reward level. Not done: the replay diff, more seeds, an eval with CSV for the
completion rate, and the legacy curves beside it.

---

## Phase 7: The observation pass

**Produces:** an observation the scenario describes, and the new orb meta.

**Files:** `core/orbs/orb_meta.py`, `scenario/blocks/observation.py`, `gymnasium/observation_space/`, the
observation part of the config.

**Settle first:**

- Which global features an observation carries. They share their readers with the HUD and keep their own list.
- Whether non-tier synergy orbs despawn like negative orbs.
- The shape of `obs_conf`.
- What the observation shows for an orb that never expires, for a perception that writes an orb's timer. A tier
  orb's `orb.TIMER.remaining` is `None` while it is on the field. Neither perception in `src` writes the timer.

**Write:**

1. The orb identity: where the dense id lives, and whether the radix formula becomes a plain sort on kind and tier.
   `OrbMeta` already stores only kind and tier.
2. Global features supplied by the scenario, as `GlobalFeature`s in `ObservationRules.global_features`, each with
   its bound and a reader from `scenario/blocks/readers.py` or the family's builder. The two kept perceptions carry
   none, after the owner's ablation; see "The kept perceptions observe only what earned its place" in
   `design-decisions.md`. Chain progress is needed where tier orbs return, since a first and a second tier 1 then
   pay differently and only the chain position tells them apart. In the single tier chain family no tier orb returns.
3. The other perceptions, rewritten from their copies in `legacy/` against the new rules.
4. The view radius, a field on `ObservationRules` that the limited-view perception reads in place of its hardcoded
   distance, and that the renderer's fog overlay reads in phase 7b. One number, so the picture cannot show a window
   different from the one the agent has.

**Do this as the only change in its step**, and diff the observation against legacy again. Check whether anything
scales the observation by its upper bound.

**Done when:** the observation trace matches legacy for spatial, or every difference is explained.

---

## Phase 7b: The renderer rework

**Produces:** a renderer split by reason to change, and a fog overlay that shows what the agent sees. The HUD keeps
its look and its label matching; the modular HUD in 7c replaces both.

**Files:** `rendering/pygame_renderer.py`, `rendering/world_view.py`, `rendering/hud_view.py`, `rendering/assets/`,
`runners/human_runner/`, `pyproject.toml`.

**Write:**

1. Move the renderer's images and font into `rendering/assets/` with `git mv`: `sprites/`, `tiles/` and `fonts/`.
   `sounds/` stays in `assets/`, since the runner uses it. Add the new folder to the package data. A pure move, in
   its own commit.
2. Split `pygame_renderer.py` into a coordinator, `world_view.py` and `hud_view.py`, as pure moves. Three modules,
   with no base classes or interfaces.
3. Move `get_user_action` and its quit handling to the human runner. Done on 2026-10-25, together with the summary,
   the pause screen, the window's title and `close()`, which the split in step 2 moves as they are.
4. The world view takes a view radius and dims every cell farther from the droid than that, with the same distance
   the perception uses. The radius is the `ObservationRules` field from phase 7.

**Done when:** a human run looks as before apart from the fog, and the observation and the reward are unchanged.

---

## Phase 7c: The modular HUD

**Produces:** a HUD stacked from drawn pieces and drawn by element kind, so a scenario can list any number of
elements and they come out in the same style. It replaces the single `hud.png` and the label matching from phase 5.

**Files:** `rendering/hud_view.py`, the HUD theme file and the HUD pieces in `rendering/assets/`.

**Settle first:**

- Where a bar sits. A stack grows in one direction, so a bar across it either lives in a cap at a fixed size or
  stretches with the stack. Another option is a row piece per element kind, a stat row and a bar row, of one width
  and with the same connector edges, so a bar is a row like any other.
- What the window does when the HUD and the grid differ in height. The taller of the two sets it.

**Write:**

1. The pieces, drawn by the owner: a top cap, a row piece that tiles with matching connector edges, and a bottom cap,
   all of one width.
2. The theme file describes each piece: its picture and the anchors inside it, measured from the piece's own corner.
   Every row reuses the row piece's anchors.
3. The HUD view stacks the cap, one row per element the scenario lists and the bottom cap, in the order of
   `scenario.hud`.
4. Each element is drawn by its kind, so the HUD draws the elements a scenario lists and picks none by label. The
   label matching from phase 5 goes, and with it the energy number drawn under the energy bar, a leftover of
   legacy's score; the score is drawn from its own element. Bars already fill against `HudBar.maximum` since phase 5.
   The theme file replaces the hand-tuned numbers in the draw helpers.

**Done when:** a scenario that lists one more element gets one more row with no change to the art, the code or the
theme file, and the observation and the reward are unchanged.

---

## Phase 8: The rest of the single tier chain family

One scenario at a time, each held against its thesis runs in `reproduction_package/` as a reference.

1. **Scaling sparse.** Max-tier, like spatial. Mostly a builder and a YAML.
2. **Negative orb.** One orb in a tier chain. Its digester, population and bundle are written, and the family helper
   counts it in the field size, the observation slots and the sort limit and names a `negatives_eaten` metric, whose
   reader is written. Its lifespan is `_reachable_life_span` since 2026-10-18. On 2026-10-25 the owner played it by
   hand: it spawned, despawned and returned as planned. The same day the owner trained PPO on spatial 5x5 with
   `vector_markovian` and a negative orb (`reward: -3`, `cool_down: 10`), and it learned, somewhat slower than
   without one. Before that run, 20,000 random steps with it stayed inside the observation `Box`. CSV was off, so
   `negatives_eaten` over the run was not read. It does not affect the chain.
3. **Delay.** Its own termination strategy, and the field going empty for `delay` steps after a consume, the
   negative orb with it. Where that mechanic lives and how the scenario triggers it is open; nothing in the scenario
   layer is reserved for it.
4. **Dense.** The threshold digester is written, and the droid settles what its chain is owed when the episode
   ends. What is left is its builder.

---

## Phase 9: Logging

- The CSV logger reads `scenario.metrics` since 2026-10-23: the environment puts them in `info` at the end of an
  episode, and the logger's columns are `RecordEpisodeStatistics`'s `r`, `l` and `t` followed by the metric names.
  What is left is to check the CSVs of a real training and evaluation run.

---

## Phase 10: Close out

- Tests: the owner rebuilds the suite mostly from scratch, against the new structure. The old suite targets the old
  schema and imports the legacy strain.
- Renumber the phases in the order they were done, and update every reference to a phase number.
- Delete `legacy/` and the `USE_LEGACY` switch once the comparison is no longer needed.
- Bring `AGENTS.md` and `docs/architecture/` in line with what was built.
- Lock the scenarios that have reproduced their legacy results: move their fixed values from the YAML into the
  builder (see `codebase-ideas.md`).
- A naming pass, once everything is wired and legacy is gone, so every part is named in the same pass. The single
  tier chain family's own names were done early, on 2026-10-25, before phase 8 adds three builders to it. What is
  left of it is the scenario names, the `GOAL_TIER_CHAIN_*` enum members and `goal_tier_chain_*.yaml`, which also tie
  the rebuild to saved configs, the output folders and the thesis. The mechanic's names (`digesters/tier_chain/`,
  `TierChainPopulation`) describe the orbs and stay.
  See: "The family is named single tier chain" in `design-decisions.md`.

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world.
- Sampled fields and the weighted orb blocks.
