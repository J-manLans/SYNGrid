# Scenario Rebuild Roadmap

What is left of the rebuild, in order, and the state of each part: written, a stub or legacy, and whether it has run.

Nothing done stays here: when a step is done it is removed, and what was done and run goes in its commit message.
Reasons go in `design-decisions.md`, undecided points in `open-questions.md`. No dates, except where a fact needs
one to be read correctly.

**Next:** run spatial's remaining seeds, then lock it (phase 6).

---

## Where things stand

*Has it run* is one of: **written** (never run), **run** (executed, by a script, a human or the environment) or
**learned** (an agent trained on it and learned).

| Part | State | Has it run |
|---|---|---|
| Spatial: builder, YAML | written | learned: RPPO and PPO on 5x5, with and without a negative orb |
| Scaling sparse: builder, YAML | written | learned: tier 4, in the ablations |
| Scaling dense: builder | stub, raises `NotImplementedError` | — |
| Delay: builder, termination, its mechanic | stub, raises `NotImplementedError` | — |
| Max-tier digester | written | learned |
| Step-wise digester | written | written |
| Threshold digester | written | run, by a script through the real droid and termination; not through the environment |
| Negative orb: digester, population, bundle | written | learned; `negatives_eaten` was never read over a run |
| `SingleTierChainTermination` | written | learned |
| Droid, engine, `GridWorld`, orbs, timer | written | learned |
| `GridWorld._deactivate_all_orbs`, `_reactivate_all_orbs` | parked under `# TODO(delay)`, read a config that is gone | — |
| Environment | written | learned; `check_env` passes |
| `vector_markovian`, `vector_fog_of_war` | written | learned |
| The other five perceptions | legacy, in `legacy/` | — |
| Renderer | written | run, in human mode |
| `HumanRunner` | written | run |
| PPO and RPPO runners | written | learned |
| FrameStackPPO runner | written | written |
| CSV logger | written | run, through the environment; never on a real training run |
| `plot/` | legacy: reads `LogKey` members that are gone | — |
| `runners/agent_runners/utils/extractors.py` | legacy: imported by nothing | — |
| `scenario/families/continuous/` | empty package | — |
| Tests, `scripts/replay_probe.py` | legacy: import the legacy strain | — |

---

## Phase 6: Rebuild, verify and lock the scenarios

**Produces:** every scenario of the single tier chain family rebuilt, shown to learn the way it did in the thesis
era, and locked. Legacy and the thesis runs are references and not targets.

In this order, per scenario:

1. **Rebuild** it.
2. **Verify** it. Train several seeds and compare the shape of the curves with legacy's, or with the thesis runs in
   `reproduction_package/` where legacy has none. The curves come from TensorBoard. Only if the curves part ways:
   replay a fixed action sequence through both strains and diff the observation per step, to find where.
3. **Ablate** the values in doubt, while the YAML still sets every value. A variant that learns as well keeps the
   simpler value. Run several seeds per variant.
4. **Lock** it: its fixed values move from the YAML into its base config, and the run's YAML keeps only its axis. See
   "A locked scenario is a base config and a small model in front of its builder" in `design-decisions.md`.
5. **Record a trace** with the replay probe: one seed, a fixed action sequence, and every observation, reward and
   termination flag per step. Phase 7 is diffed against it. The probe imports the legacy strain, so the first lock
   ports it to take a `Scenario`.

**Compare with care:** anything that changed on purpose is in `design-decisions.md` under "Legacy is a reference,
not a target", and what could move the observation or reward is in `landscape-changes.md`. A completed max-tier chain
pays what its orbs are worth, where legacy paid a fixed 10, so compare each side's share of its own ceiling and how
fast the curves climb, not the reward level.

**Done when:** every scenario is rebuilt, its runs over several seeds climb towards their own ceiling while the
episodes shorten, and each is locked.

### What is left, scenario by scenario

1. **Spatial.** Run the remaining seeds. The ceiling on 5x5 is about 9.02 (tiers 1 to 3 worth `1 + 2^1.5 + 3^1.5`,
   less the step penalty); RPPO on seed 3 reached about 8.5 with episodes of about 15 steps. Then lock it, with its
   visibility fixed as the fog-of-war window, and record its trace.
2. **Scaling sparse.** Verify over several seeds at more than one `max_tier`, then lock.
3. **Negative orb.** Read `negatives_eaten` over a training run with CSV on.
4. **Delay.** Its builder, its own termination, and the field going empty for `delay` steps after a consume, the
   negative orb with it. Where that mechanic lives is open; see `open-questions.md`. The two parked delay methods in
   `GridWorld` move into it or go.
5. **Dense.** Its builder, on the threshold digester.

---

## Phase 7: The observation pass

**Produces:** an observation the scenario describes, and the new orb meta.

**Files:** `core/orbs/orb_meta.py`, `scenario/blocks/observation.py`, `gymnasium/observation_space/`, the
observation part of the config.

**Settle first** (see `open-questions.md`, phase 7): which global features an observation carries; whether non-tier
synergy orbs despawn like negative orbs; the shape of `obs_conf`; what a perception that writes an orb's timer shows
for an orb that never expires.

### 7a: Align what is there today

The two perceptions in the `Perception` enum brought in line with the new rules. Nothing new is added.

1. The orb identity: where the dense id lives, and whether the radix formula becomes a plain sort on kind and tier.
2. Global features supplied by the scenario, as `GlobalFeature`s in `ObservationRules.global_features`, each with
   its bound and a reader. The two kept perceptions carry none. Chain progress is needed where tier orbs return.
3. The view radius, a field on `ObservationRules` that the limited-view perception reads in place of its hardcoded
   distance, and that the renderer's fog overlay reads in phase 8a.

**One change at a time.** Each of the three is its own commit, and the replay probe is run after each against the
phase 6 trace. A difference is explained or reverted before the next change starts. The reward must not change.
Check whether anything scales the observation by its upper bound. When 7a is done, train spatial on a seed or two.

**Done when:** the observation trace differs from the phase 6 trace only where a change was meant to, the reward
trace is identical, and spatial still learns.

### 7b: New perceptions, one at a time

Rewritten from their copies in `legacy/` against the new rules, each added alone and checked before the next.

**Settle first:** how an extractor is chosen. The policy name comes from the perception and the algorithm in
`policy_resolver.py`, and the extractor would be a second thing the perception implies.

1. **Composite.** The observation as a dictionary of groups. SB3's `MultiInputPolicy` flattens every key by default,
   so it needs a features extractor of its own. `GroupedMetaExtractor` in `extractors.py` is a first attempt to
   rewrite with this step.
2. **Pixel.** The grid as an image, for a CNN. SB3's default CNN expects larger images, so it likely needs a small
   extractor too. `TinyGridCNN` in the same file trains but is unchecked.

**Done when:** each new perception stays inside its observation space on random steps and an agent trains on it.

---

## Phase 8a: The renderer rework

**Produces:** a renderer split by reason to change, and a fog overlay that shows what the agent sees.

**Files:** `rendering/pygame_renderer.py`, `rendering/world_view.py`, `rendering/hud_view.py`, `rendering/assets/`,
`pyproject.toml`.

1. Move the renderer's images and font into `rendering/assets/` with `git mv`: `sprites/`, `tiles/` and `fonts/`.
   `sounds/` stays in `assets/`, since the runner uses it. Add the folder to the package data. A pure move, in its
   own commit.
2. Split `pygame_renderer.py` into a coordinator, `world_view.py` and `hud_view.py`, as pure moves, with no base
   classes. The coordinator keeps the window, the summary and pause screens, the dimming and text helpers, the title
   and `close()`. The world view takes the floor, the orbs and the droid. The HUD view takes the HUD.
3. The world view takes the view radius from phase 7 and dims every cell farther from the droid, with the same
   distance the perception uses. No radius, no overlay.

**Done when:** a human run looks as before apart from the fog, and the observation and the reward are unchanged.

---

## Phase 8b: The modular HUD

**Produces:** a HUD stacked from drawn pieces and drawn by element kind, so a scenario can list any number of
elements in the same style. It replaces the single `hud.png` and the matching by label.

**Files:** `rendering/hud_view.py`, the HUD theme file and the HUD pieces in `rendering/assets/`.

**Settle first** (see `open-questions.md`, phase 8b): what the renderer is handed about the HUD; where a bar sits;
what the window does when the HUD and the grid differ in height.

1. The pieces, drawn by the owner: a top cap, a row piece that tiles with matching connector edges, and a bottom cap,
   all of one width.
2. The theme file describes each piece: its picture and the anchors inside it, measured from the piece's own corner.
3. The HUD view stacks the cap, one row per element in `scenario.hud`, and the bottom cap.
4. Each element is drawn by its kind, so no element is picked by label. The energy number under the energy bar goes,
   and the score is drawn from its own element. The theme file replaces the hand-tuned numbers in the draw helpers.

**Done when:** a scenario that lists one more element gets one more row with no change to the art, the code or the
theme file, and the observation and the reward are unchanged.

---

## Phase 9: Logging

1. Check the CSVs of a real training run and a real evaluation run.
2. The `plot/` package becomes a generic helper: it reads a run's CSV (the episode columns `r`, `l`, `t` plus the
   scenario's metrics), and a user subclasses it for the plots their scenario needs. The thesis plots ship as
   ready-made subclasses. Its hardcoded paths and the `LogKey` members that are gone are fixed with it. What the base
   class looks like is open.

**Done when:** a plot is made from the CSV of a real run, through a subclass.

---

## Phase 10: Close out

- Tests: the suite is rebuilt from scratch against the new structure, tight and complete, not large. The owner wants
  tooling that shows untested paths, such as branch coverage, and mutation testing. Which libraries is open.
- Delete `legacy/`, the `USE_LEGACY` switch and `landscape-changes.md` once the comparison is no longer needed.
- Write `docs/architecture/` fresh, for people.
- A naming pass once everything is wired and legacy is gone. The scenario and family names are done, and the
  mechanic's names (`digesters/tier_chain/`, `TierChainPopulation`) stay.

**Done when:** the new test suite passes, and `legacy/` and the `USE_LEGACY` switch are gone and nothing imports
them.

---

## Not on this roadmap

- The continuous family and the sandbox scenario.
- Effect orbs, and what they need from the engine: a tick, a stage before routing, a way to change what another
  kind's orb pays, a way to reach the world. See "Effect orbs get their magic from the engine" in
  `codebase-ideas.md`.
- Sampled fields and the weighted orb blocks.
