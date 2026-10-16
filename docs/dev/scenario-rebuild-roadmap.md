# Scenario Rebuild Roadmap

The order to rebuild the scenario layer in, written 2026-10-10. Each phase says what it produces, which files it
touches, what has to be decided first and how to tell it is done. The reasons behind the decisions are in
`design-decisions.md`; this file is only the order of work.

The route is a vertical slice: get `goal_tier_chain_spatial` running end to end on the new structure, check it against
legacy as a reference, and only then widen to the other scenarios. Until phase 5 nothing runs, so each phase before it has its own
small check.

---

## Where things stand

| Part | State |
|---|---|
| Config models, registry, spatial YAML | done; the YAML validates. Orb models are in `config/models/scenario/orb_models.py` |
| `scenario/scenario.py`, `blocks/hud.py`, `blocks/metrics.py`, `blocks/orb_bundle.py` | written, from the skeleton |
| `blocks/observation.py`, `termination.py`, `blocks/populations/population.py` | written: the rules dataclass and the two protocols. A population has `create` and `count` |
| `blocks/populations/tier_chain_population.py` | written: one tier orb per tier, with its reward and the cool-down it is handed. Its count is `max_tier` |
| `blocks/populations/neg_population.py` | written: one negative orb, used by the negative bundle. Its count is 1. It gives the orb no lifespan |
| `families/goal/tier_chain/termination.py` | written: the four endings, what the chain is owed, and the timeout penalty. It reads `energy` and `get_digester(...)` from the droid |
| `families/goal/tier_chain/builders.py` | the spatial builder returns a `Scenario` whose tier bundle makes max-tier digesters, with a negative bundle and one more orb in its counts when the config has a `neg_orb_conf` block; the other three builders raise `NotImplementedError` |
| `blocks/recipes/world_recipe.py` | written: `WorldRecipe`, what a scenario holds as `build_world`, shared by every family. Its body builds the droid and the world, and returns a `GridWorld` for the spatial scenario since 2026-10-16 |
| `blocks/recipes/tier_digester_recipe.py` | written: `TierDigesterRecipe`, what a tier bundle holds to make its digester |
| Tier-chain config | in line with the digesters since 2026-10-11: no `completion_reward`, the reward ladder on the shared `TierOrbConf`, no dense classes |
| `core/droid/digestion/digester.py` | written: the `OrbDigester` protocol (`KIND`, `reset`, `digest` returning a float, `undigested`). The routing key, `OrbKind`, is in `orb_meta.py` |
| `core/droid/digestion/digesters/tier_chain/` | written: the `TierDigester` base and the max-tier, step-wise and threshold modes, each with `undigested`. The spatial builder names the max-tier one |
| `core/droid/digestion/digesters/neg_digester.py` | written: `NegDigester` pays the orb's own reward, counts `negatives_eaten` and is owed nothing at the end. The negative bundle names it |
| `core/orbs/orb_meta.py` | stores kind and tier; the radix identity is unchanged |
| `core/droid/digestion/engine.py` | written: a router with `reset`, `digest`, `get` and `undigested`, built from digesters. The droid builds it from the digesters it is handed |
| `core/droid/synergy_droid.py` | written (step 5 of phase 3, done 2026-10-16): takes its digesters, the grid and `max_energy`, `step_penalty` and `boundary_penalty` as values, builds its engine and keeps it private, and holds energy and score, which are first set in `reset`. Every reward goes through it: `on_termination(ending_reward)` takes in what digestion has left undigested and what the ending costs or pays, adds them to the score and leaves energy alone. `get_digester(...)` gives a reader a digester by class |
| `core/grid_world.py` | being rebuilt (step 6 of phase 3): the constructor takes the grid, `max_active_orbs`, `max_steps`, the orbs and the droid, and the world keeps the step clock as `steps_left`. `reset` and `perform_droid_action` still read `self._conf`, which is gone, so it does not run |
| `core/` (orbs), `gymnasium/`, `rendering/` | the droid and the world import since 2026-10-16. The rest still import the old config and do not import. The environment, `episode_termination.py` and the base perception still read the old engine's `chained_tiers`, `stats` and `_pending_reward` |
| Runners, `env_factory.py` | already take a `Scenario` |
| Legacy strain | in `src/syn_grid/legacy/`, switched by `USE_LEGACY` in `__main__.py` |

`import syn_grid.scenario.registry` works and `build_scenario` returns the spatial `Scenario`. Two scenarios built
from the same config compare equal. `scenario.build_world()` returns a `GridWorld` since 2026-10-16, whose `reset` does not run yet. `import syn_grid.app` and
`python3 -m syn_grid.check_env` fail until phase 5.

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

**Produces:** the pieces a `Scenario` is made of, for the tier-chain family.

**Files:** `scenario/blocks/termination.py`, `observation.py`; `scenario/blocks/populations/`;
 `scenario/families/goal/tier_chain/termination.py`.

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

- `TierChainTermination` reads `world.droid.energy` and calls `world.droid.get_digester(...)`. The droid has
  neither, so a type checker reports both.
- `TierChainPopulation` gives its tier orbs a cool-down of `None`, and `BaseOrb` does not act on it. No orb is
  handed a lifespan.
- `WorldRecipe` holds `max_active_orbs`, and its body raises.
- The five world readers in the tier-chain builders call `world.droid.get_digester(...)`. They were written on
  2026-10-13, after the phase was closed, as the part of phase 4 that could be done early.

---

## Phase 3: Core takes pieces, not config

**Produces:** a world, a droid and orbs that are handed what they need.

**Files:** `core/droid/synergy_droid.py`, `core/grid_world.py`, `core/orbs/`, `core/utils/timer.py`; on the scenario
side `blocks/populations/`, `blocks/recipes/world_recipe.py` and the tier-chain builders.

**Write:**

1. Droid: energy and score as two numbers; takes its digesters and the grid size. It keeps its engine to itself and
   gives a reader a digester by class through `get_digester(...)`. `TierChainTermination` already reads `energy` and
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
| 1 | `core/utils/timer.py` | only if needed, so a timer can stand for "never" when a cool-down or lifespan is `None` |
| 2 | `core/orbs/base_orb.py` | takes a lifespan per orb, acts on a cool-down of `None`, and says whether it is ready to return and whether it has expired. The class-level `_life_span` goes |
| 3 | `core/orbs/synergy/chain/tier_orb.py`, `core/orbs/direct/negative_orb.py` | pass the lifespan through |
| 4 | `scenario/blocks/populations/` | hand each orb its lifespan: `None` for tier orbs, a number for the negative orb |
| 5 | `core/droid/synergy_droid.py` | takes digesters and the grid size as values, makes its own engine, splits energy from score, gets `on_termination`, and gets `get_digester` |
| 6 | `core/grid_world.py` | takes the grid, `max_active_orbs`, `max_steps`, the orbs and the droid, counts the steps, fills the field at reset, and asks its orbs each step |
| 7 | `core/orbs/orb_factory.py` | deleted, once the world no longer imports it |
| 8 | `scenario/blocks/recipes/world_recipe.py` | the body of `__call__`, the last piece of phase 4 |
| 9 | `scenario/families/goal/tier_chain/builders.py` | pass the negative orb's lifespan to its population, since it follows from the grid size |

- The orbs come first because they import nothing that is broken, so they can be tried alone.
- The droid comes before the world because the world is handed a droid. With `energy` and `get_digester` on the
  droid, the type errors in the tier-chain termination and builders go away.
- The recipe's body comes last because it calls the constructors that steps 5 and 6 define.
- At step 6, `steps_left` can leave `TerminationRules.evaluate`, and the open question of orbs or populations is
  settled. With a lifespan per orb from step 2, a list of orbs is enough.
- `DroidAction` is in `core/droid/droid_action.py` since 2026-10-16, so core imports nothing from `gymnasium/`.

**Check while here** (from the "things that bite" list in `AGENTS.md`):

- `BaseOrb._life_span` is a class attribute shared by every env in the process. Step 2 removes it.
- `GridWorld.reset()` without an rng re-rolls the layout.

**Done when:** a world can be built by hand in a script, reset with a seed, and stepped with actions.

**State on 2026-10-16: start here next session.** Step 5, the droid, is done. It has `get_digester(...)`, keeps its
engine private as `_digestion_engine`, has its docstrings, and is formatted. Its episode state is first set in
`reset`; see "Episode state is first set in `reset`" in `design-decisions.md`. `on_termination(ending_reward)` adds
what digestion has left undigested and what the ending costs or pays to the score, and leaves energy alone. The
digesters' names follow the digestion metaphor: `undigested()`, `chain_worth` and `chain_tier`.

`WorldRecipe.__call__` returns a `GridWorld` for the spatial scenario, with a droid that resets. The world's `reset`
and `perform_droid_action` still read `self._conf`. Steps 1 to 4, the orbs' lifespans and cool-downs, have not been
checked against the code; the world's step 6 needs them, since it asks its orbs each step.

Done on 2026-10-15: the score is back on the droid, beside energy, and every reward goes through it. Termination
returns only `EpisodeOutcome.ending_reward` and its `evaluate` takes no `reward`.

For phase 5: the step is the droid acting, `termination.evaluate(world, steps_left)`, and if the episode ended
`world.droid.on_termination(outcome.ending_reward)` when `outcome.terminated`. The step's reward is the sum of the two.
The observation, the HUD and the episode info are read after that, and read the score from the droid.

---

## Phase 4: The builders

**Produces:** a `Scenario` for spatial.

**Files:** `scenario/families/goal/tier_chain/builders.py`.

**Done on 2026-10-11:** the config is in line with the digesters. The reward ladder is one formula on the shared
`TierOrbConf`, `completion_reward` is gone from the models, the termination, the builder and the spatial YAML, and
dense has no config classes of its own.

**Write:**

1. `build_tier_chain_spatial`: the tag and the tier bundle (tier orbs with their rewards, and a
   `TierDigesterRecipe` that makes a max-tier digester from `max_tier` and `chain_break_penalty`). Written.
2. `_tier_chain_scenario`: what the family shares.
3. The body of `WorldRecipe.__call__` in `scenario/blocks/recipes/world_recipe.py`. The recipe itself is written,
   and the family helper holds one as `build_world`. The body calls the droid's and the world's constructors, so it
   is written last in phase 3.
4. The world readers, which take their numbers from the digesters through `world.droid.get_digester(...)`. Written
   on 2026-10-13 and read from a stand-in world.

**Done when:** `import syn_grid.scenario.registry` works, `build_scenario` returns a `Scenario`, two scenarios built
from the same config compare equal, and `scenario.build_world()` returns a new world on each call.

---

## Phase 5: The Gymnasium side

**Produces:** an environment that runs a `Scenario`.

**Files:** `gymnasium/environment.py`, `gymnasium/observation_space/`, `gymnasium/utils/episode_termination.py`
(removed), `rendering/pygame_renderer.py`.

**Settle first:** where the renderer's settings come from. The old `RendererConf` has no counterpart in the new models.

**Write:**

1. The environment takes the scenario, builds its world from it, and asks `scenario.termination` when an episode ends.
2. The observation handler reads the steps left from the world.
3. The HUD and the episode info read from `scenario.hud` and `scenario.metrics`, or a minimal stand-in.

**Done when:** `python3 -m syn_grid.check_env` passes, random actions stay inside the observation space, and a human
run plays an episode to each of its endings.

---

## Phase 6: Check against legacy

**Produces:** evidence that the rebuilt spatial scenario learns at least as well as the baseline.

1. Replay the phase 0 action sequence through the rebuild and diff the observation per step. Read the size of a
   difference before calling it behavioural.
2. Train several seeds on each side and compare the curves.

**Known differences to account for:** anything that changed on purpose is in `design-decisions.md`. The rewards are
expected to differ: a completed max-tier chain pays what its orbs are worth, where legacy paid a fixed 10. Set the
base reward so the two are of similar size before comparing curves.

**Done when:** the observation traces match, or every difference is explained, and the rebuild's curves are no worse
than legacy's within seed noise. If legacy did not learn in phase 0, this phase shows the same problem and not a new
one.

---

## Phase 7: The observation pass

**Produces:** an observation the scenario describes, and the new orb meta.

**Files:** `core/orbs/orb_meta.py`, `scenario/blocks/observation.py`, `gymnasium/observation_space/`,
`rendering/pygame_renderer.py`, the observation part of the config.

**Settle first:**

- Which global features an observation carries, and whether the HUD and the observation share readers.
- Whether non-tier synergy orbs despawn like negative orbs.
- The shape of `obs_conf`.

**Write:**

1. The orb identity: where the dense id lives, and whether the radix formula becomes a plain sort on kind and tier.
   `OrbMeta` already stores only kind and tier.
2. Global features supplied by the scenario: energy, steps left, chain progress. Chain progress has to stay: the
   break rule pays a first and a second tier 1 differently, and only the chain position tells them apart.
3. The two verified perceptions against the new rules.

**Do this as the only change in its step**, and diff the observation against legacy again. Check whether anything
scales the observation by its upper bound.

**Done when:** the observation trace matches legacy for spatial, or every difference is explained.

---

## Phase 8: The rest of the tier-chain family

One scenario at a time, each held against its thesis runs in `reproduction_package/` as a reference.

1. **Scaling sparse.** Max-tier, like spatial. Mostly a builder and a YAML.
2. **Negative orb.** One orb in a tier chain. Its digester, population and bundle are written, and the family helper
   counts it in the field size, the observation slots and the sort limit and names a `negatives_eaten` metric, whose
   reader is written. What is left is its lifespan and running it. It does not affect the chain.
3. **Delay.** Its own termination strategy, and the field going empty for `delay` steps after a consume, the
   negative orb with it. Where that mechanic lives and how the scenario triggers it is open; nothing in the scenario
   layer is reserved for it.
4. **Dense.** The threshold digester is written, and the droid settles what its chain is owed when the episode
   ends. What is left is its builder.

---

## Phase 9: Logging and HUD

- The logger reads `scenario.metrics`. Expected to be a localized change.
- The form of the HUD: a small vocabulary of elements, or a drawer per scenario.

---

## Phase 10: Close out

- Tests: yours to curate; the old suite targets the old schema.
- Delete `legacy/` and the `USE_LEGACY` switch once the comparison is no longer needed.
- Bring `AGENTS.md` and `docs/architecture/` in line with what was built.
- Lock the scenarios that have reproduced their legacy results: move their fixed values from the YAML into the
  builder (see `codebase-ideas.md`).

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world.
- Sampled fields and the weighted orb blocks.
