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
| `blocks/observation.py`, `population.py`, `termination.py` | written: the rules dataclass and the two protocols |
| `blocks/spawning.py` | stub: three fields, hooks without bodies, and an unused `OrbFieldUpdate` protocol |
| `goal/tier_chain/population.py`, `termination.py` | stubs: no orbs are made, and no episode ends |
| `goal/tier_chain/builders.py` | the spatial builder returns a `Scenario`; `build_world`, `_make_tier_digester`, the negative bundle and the other three builders raise `NotImplementedError` |
| Tier-chain config | in line with the digesters since 2026-10-11: no `completion_reward`, the reward ladder on the shared `TierOrbConf`, no dense classes |
| `core/droid/digestion/digestion.py` | written: the `OrbDigester` protocol (`KIND`, `reset`, `digest` returning a float). The routing key, `OrbKind`, is in `orb_meta.py` |
| `core/droid/digestion/digesters/tier_chain/` | written: the `TierDigester` base and the max-tier, step-wise and threshold modes, each with `owed_at_episode_end`. Not wired into a builder |
| `core/droid/digestion/digesters/neg_digester.py` | written: `NegDigester` pays the orb's own reward and counts `negatives_eaten`. Not wired into a builder |
| `core/orbs/orb_meta.py` | stores kind and tier; the radix identity is unchanged |
| `core/droid/digestion/engine.py` | written: a router with `reset`, `digest` and `get`, built from digesters. The droid still constructs it the old way |
| `core/` (world, droid, orbs), `gymnasium/`, `rendering/` | still import the old config; do not import. The droid, the environment, `episode_termination.py` and the base perception still read the old engine's `chained_tiers`, `stats` and `_pending_reward` |
| Runners, `env_factory.py` | already take a `Scenario` |
| Legacy strain | in `src/syn_grid/legacy/`, switched by `USE_LEGACY` in `__main__.py` |

`import syn_grid.scenario.registry` works and `build_scenario` returns the spatial `Scenario`. Two scenarios built
from the same config do not compare equal yet, because `build_world` is a `functools.partial`. `import syn_grid.app`
and `python3 -m syn_grid.check_env` fail until phase 5.

---

## Phase 0: A baseline to compare against

Do this first, or a later mismatch can't be told apart from a problem that was already there. Legacy is a reference
and not a target: the rebuild's rewards differ from it on purpose, so the curves are what gets compared.

- Confirm the legacy strain learns spatial, with several seeds.
- Record its behaviour with the replay scripts in `scripts/`: one seed, a fixed action sequence, the reward and the
  observation per step. Use a `starting_score` high enough that episodes reach the timeout.

**Done when:** there are legacy training curves to hold the rebuild's against in phase 6, and a recorded
step-by-step trace whose observations can be diffed.

---

## Phase 1: Digestion

**Produces:** something that turns a consumed orb into a reward, with no knowledge of scenarios.

**Files:** `core/droid/digestion/digestion.py`, `engine.py`, and `digesters/tier_chain/`, which holds the tier base
and one file per scoring mode.

**Settled on 2026-10-11:**

- The droid creates the engine. `build_world` makes the digesters, one per `OrbBundle`, and hands them to the droid.
- No events, no tally and no cross-kind hook. `digest` returns a float, and a reader takes what it needs from the
  digester itself.
- A tier orb is worth the same in every scoring mode, and the mode decides when it is paid.

**Settled on 2026-10-12:** the engine raises when two digesters own one kind, when an eaten orb's kind has no
digester, and when `get` is asked for a class that is not registered.

**Write:**

1. `digestion.py`: the `OrbDigester` protocol with `KIND`, `reset` and `digest`. The routing key is the orb's kind,
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

**Files:** `scenario/blocks/population.py`, `spawning.py`, `termination.py`, `observation.py`;
`scenario/goal/tier_chain/population.py`, `termination.py`.

**Write:**

1. Population: the protocol in `blocks/`, and the tier-chain one that makes one tier orb per tier.
2. Spawning: the whole chain on the field from the first step, no tier orb expires, and an eaten tier orb does not
   return in that episode. The last one is a requirement and not a detail: with one orb per tier it means a chain
   can never be broken with a tier 1, which the scoring rules count on in a goal scenario.
3. Termination: the outcome type and the protocol in `blocks/`; the tier-chain strategy in its own file. It ends on
   energy 0, a broken chain, a completed chain or the clock. It only adds to the step's reward: what the tier
   digester's `owed_at_episode_end` returns at every ending, and `timeout_penalty` when the clock runs out. Nothing
   for a completed chain.
4. Observation rules: only what the current perception needs for now. The full shape waits for phase 7.

**Done when:** each file imports on its own, and the termination strategy gives the right outcome for a hand-built
world state in each of its four endings.

---

## Phase 3: Core takes pieces, not config

**Produces:** a world, a droid and orbs that are handed what they need.

**Files:** `core/droid/synergy_droid.py`, `core/grid_world.py`, `core/orbs/`.

**Write:**

1. Droid: energy and score as two numbers; takes its digesters and the grid size.
2. World: takes the grid, the droid, the orbs and the spawning rules; counts the steps of an episode.
3. Orbs: the tier orb loses its scoring flags. `orb_factory.py` goes, replaced by the population.

**Check while here** (from the "things that bite" list in `AGENTS.md`):

- `BaseOrb._life_span` is a class attribute shared by every env in the process.
- `GridWorld.reset()` without an rng re-rolls the layout.

**Done when:** a world can be built by hand in a script, reset with a seed, and stepped with actions.

---

## Phase 4: The builders

**Produces:** a `Scenario` for spatial.

**Files:** `scenario/goal/tier_chain/builders.py`.

**Done on 2026-10-11:** the config is in line with the digesters. The reward ladder is one formula on the shared
`TierOrbConf`, `completion_reward` is gone from the models, the termination, the builder and the spatial YAML, and
dense has no config classes of its own.

**Write:**

1. `build_tier_chain_spatial`: the tag and the tier bundle (tier orbs with their rewards, a max-tier digester made
   by `_make_tier_digester` from `max_tier` and `chain_break_penalty`).
2. `_tier_chain_scenario`: what the family shares.
3. `_build_tier_chain_world` and the world readers, which take the chain's state from the tier digester through the
   engine.

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
2. **Negative orb.** One orb in a tier chain: its bundle, one more field cell and observation slot, and a metric
   that reads `negatives_eaten`. The digester is written. It does not affect the chain.
3. **Delay.** Its own termination strategy, the cooldown after a consume, and the negative orb freezing with the
   field.
4. **Dense.** The threshold digester is written, and the family's termination pays what its chain is owed. What is
   left is its builder.

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
