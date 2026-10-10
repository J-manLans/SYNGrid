# Scenario Rebuild Roadmap

The order to rebuild the scenario layer in, written 2026-10-10. Each phase says what it produces, which files it
touches, what has to be decided first and how to tell it is done. The reasons behind the decisions are in
`design-decisions.md`; this file is only the order of work.

The route is a vertical slice: get `goal_tier_chain_spatial` running end to end on the new structure, compare it with
legacy, and only then widen to the other scenarios. Until phase 5 nothing runs, so each phase before it has its own
small check.

---

## Where things stand

| Part | State |
|---|---|
| Config models, registry, spatial YAML | done; the YAML validates |
| `scenario/scenario.py`, `blocks/hud.py`, `blocks/metrics.py`, `blocks/orb_bundle.py` | written, from the skeleton |
| `blocks/observation.py`, `population.py`, `spawning.py`, `termination.py` | empty |
| `goal/tier_chain/builders.py` | signatures and docstrings; the spatial builder stops after the tag |
| `core/droid/digestion/digestion.py`, `digesters/*.py` | empty |
| `core/droid/digestion/engine.py` | still the legacy engine |
| `core/` (world, droid, orbs), `gymnasium/`, `rendering/` | still import the old config; do not import |
| Runners, `env_factory.py` | already take a `Scenario` |
| Legacy strain | in `src/syn_grid/legacy/`, switched by `USE_LEGACY` in `__main__.py` |

`import syn_grid.scenario.registry` fails today. It starts working at the end of phase 4.

---

## Phase 0: A baseline to compare against

Do this first, or a later mismatch can't be told apart from a problem that was already there.

- Confirm the legacy strain learns spatial, with several seeds.
- Record its behaviour with the replay scripts in `scripts/`: one seed, a fixed action sequence, the reward and the
  observation per step. Use a `starting_score` high enough that episodes reach the timeout.

**Done when:** there are legacy training curves and a recorded step-by-step trace to diff against in phase 6.

---

## Phase 1: Digestion

**Produces:** something that turns a consumed orb into a reward, with no knowledge of scenarios.

**Files:** `core/droid/digestion/digestion.py`, `engine.py`, `digesters/tier_digester.py`.

**Settle first:**

- Who builds the engine. Suggested: `build_world` makes the digesters, the droid builds its engine from them.
- Events and the engine's tally: kept, since effect orbs will emit them too. Update the digestion entry in
  `design-decisions.md` to say so.

**Write:**

1. `digestion.py`: the routing key (the orb's type), `kind_of`, the result of a digestion, and the `OrbDigester`
   protocol with `kind`, `reset` and `digest`.
2. `tier_digester.py`: a slim tier base holding the chain state, the counts and the held reward, and max-tier as the
   one concrete class. Under the additive rule it pays the chain-break penalty on a break and nothing on completion.
3. `engine.py`: a router with `reset`, `digest` and `get`.

**Done when:** `import syn_grid.scenario.blocks.orb_bundle` works, and a max-tier digester fed tiers 1, 2, 3 reports
progress, progress, completion, and fed a wrong tier pays `chain_break_penalty`.

---

## Phase 2: The blocks the spatial builder names

**Produces:** the pieces a `Scenario` is made of, for the tier-chain family.

**Files:** `scenario/blocks/population.py`, `spawning.py`, `termination.py`, `observation.py`;
`scenario/goal/tier_chain/population.py`, `termination.py`.

**Write:**

1. Population: the protocol in `blocks/`, and the tier-chain one that makes one tier orb per tier.
2. Spawning: the whole chain on the field from the first step, no tier orb expires.
3. Termination: the outcome type and the protocol in `blocks/`; the tier-chain strategy in its own file. It ends on
   energy 0, a broken chain, a completed chain or the clock, and pays by the additive rule.
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

**Write:**

1. `build_tier_chain_spatial`: the tag and the tier bundle (zero-worth tier orbs, a max-tier digester).
2. `_tier_chain_scenario`: what the family shares.
3. `_build_tier_chain_world` and the world readers.
4. Drop the `max_score` parameter.

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

## Phase 6: Compare with legacy

**Produces:** evidence that the rebuild behaves like the baseline.

1. Replay the phase 0 action sequence through the rebuild and diff reward and observation per step. Read the size of
   a difference before calling it behavioural.
2. Train several seeds on each side and compare the curves.

**Known differences to account for:** anything that changed on purpose is in `design-decisions.md`. For spatial there
should be none in the rewards.

**Done when:** the traces match, or every difference is explained, and the curves agree within seed noise. If legacy
did not learn in phase 0, this phase shows the same problem and not a new one.

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

1. `OrbMeta` storing only type and tier, and everything that reads it.
2. Global features supplied by the scenario: energy, steps left, chain progress.
3. The two verified perceptions against the new rules.

**Do this as the only change in its step**, and diff the observation against legacy again. Check whether anything
scales the observation by its upper bound.

**Done when:** the observation trace matches legacy for spatial, or every difference is explained.

---

## Phase 8: The rest of the tier-chain family

One scenario at a time, each compared with its thesis runs in `reproduction_package/`.

1. **Scaling sparse.** Max-tier, like spatial. Mostly a builder and a YAML.
2. **Negative orb.** One orb in a tier chain: its bundle and digester, one more field cell and observation slot, and
   the second protocol for a tier digester to react to it.
3. **Delay.** Its own termination strategy, the cooldown after a consume, and the negative orb freezing with the
   field.
4. **Dense.** The threshold digester. Extract the shared tier base's payout hooks now that there is a second mode.

---

## Phase 9: Logging and HUD

- The logger reads `scenario.metrics`. Expected to be a localized change.
- The form of the HUD: a small vocabulary of elements, or a drawer per scenario.

---

## Phase 10: Close out

- Tests: yours to curate; the old suite targets the old schema.
- Delete `legacy/` and the `USE_LEGACY` switch once the comparison is no longer needed.
- Bring `AGENTS.md` and `docs/architecture/` in line with what was built.
- Decide which scenarios to lock (see `codebase-ideas.md`).

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to reach the world.
- Sampled fields and the weighted orb blocks.
