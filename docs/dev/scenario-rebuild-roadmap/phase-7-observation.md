# Phase 7: The observation pass

**Produces:** an observation the scenario describes, and the new orb meta.

**Files:** `core/orbs/orb_meta.py`, `scenario/blocks/observation.py`, `gymnasium/observation_space/`, the
observation part of the config.

**Settle first** (see `open-questions.md`, phase 7): which global features an observation carries; whether non-tier
synergy orbs despawn like negative orbs; the shape of `obs_conf`; what a perception that writes an orb's timer shows
for an orb that never expires; what an orb kind's bundle carries, by phase 8b at the latest.

## 7a: Align what is there today

The two perceptions in the `Perception` enum brought in line with the new rules. Nothing new is added.

1. The orb identity, as decided in "An orb's identity is the radix scalar, and tier order is meant to be seen" and
   "An orb's identity is looked up in a table the scenario builds": the meta frozen, the world's remap replaced by
   the scenario's table, and the observation unchanged. `world.max_identity`, the identity's bound, which the
   environment hands to the handler and the perception, moves with the remap. Where the table lives, and how the
   perception reads it before orb features exist, are in `open-questions.md`. The identity tests are revisited with
   it: `TestIdentities` in `tests/unit/core/test_grid_world.py`, which pins negative as 1 and the tiers counting up
   from 2, and `TestIdentity` in `tests/unit/core/orbs/test_orb_meta.py`. When spatial is trained, also train it
   with the plain radix on the same seeds.
2. Global features supplied by the scenario, as `GlobalFeature`s in `ObservationRules.global_features`, each with
   its bound and a reader. The two kept perceptions carry none. Chain progress is needed where tier orbs return.
3. The view radius, a field on `ObservationRules` that the lantern's perception reads in place of its hardcoded
   distance, and that the renderer's fog overlay reads in phase 8a.

**One change at a time.** Each of the three is its own commit, and the regression test is run after each against the
baselines. A difference is explained, and its baselines recorded again, or reverted before the next change starts. The
reward must not change. Check whether anything scales the observation by its upper bound. When 7a is done, train
spatial.

**Done when:** the observation differs from the baselines only where a change was meant to, the reward is identical,
and spatial still learns.

## 7a½: Delay's frozen orbs

Delay's next version, as decided in "Delay's frozen orbs stay on the field and block their cell" in
`design-decisions.md`. It comes after 7a, so the mark lands on the aligned observation and the rebuilt suite, and
before 7b, so the new perceptions are written with the frozen state from the start. The questions are in
`open-questions.md`, under phase 7.

1. An orb state for "on the field, but not edible", and blocking as an orb property of its own.
2. The droid's move stopped by a blocking orb, with no charge of its own.
3. `DelayWatcher` freezing orbs in their cells in place of taking them off the field, and the rule that a released
   orb waits for its cell going.
4. The frozen mark in the two kept perceptions, and the frozen skin in the renderer.
5. Relock delay, as every scenario was locked: train it and compare the curves with today's delay, ablate what is in
   doubt while its base config still changes, then lock it and record its baseline again with `probe.py --baseline`.
   Anything that moves the observation or reward goes in `landscape-changes.md`.

**Done when:** the delay spread at −0.5 is run again on it and learns. The baselines of spatial and scaling are
unchanged.

## 7b: New perceptions, one at a time

Rewritten from their copies in `legacy/` against the new rules, each added alone and checked before the next. A
perception is a format, and the scenario lists what it shows (see "A perception is a format, and the scenario says
what the agent knows"), so legacy's seven become vector, composite, grid and pixel. The
extractors start from `extractors.py`, which the owner has read, and are reworked from there rather than written from
scratch.

**Settle first:** how an extractor is chosen. The policy name comes from the perception and the algorithm in
`policy_resolver.py`, and the extractor would be a second thing the perception implies.

1. **Composite.** The observation as a dictionary of groups. SB3's `MultiInputPolicy` flattens every key by default,
   so it needs a features extractor of its own. `GroupedMetaExtractor` in `extractors.py` is a first attempt to
   rewrite with this step. The extractor is also the way out when a scenario's kinds crowd the radix identity: it
   can run the identity through an embedding, a learned vector per id, as MiniHack does, while the observation stays
   the same. See "An orb's identity is the radix scalar, and tier order is meant to be seen".
2. **Pixel.** The grid as an image, for a CNN. SB3's default CNN expects larger images, so it likely needs a small
   extractor too. `TinyGridCNN` in the same file trains but is unchecked.

3. **Delete legacy.** Once the perceptions are done. The owner deletes `legacy/`; the agent then clears what points at
   it, and a `grep -ri legacy` outside `docs/dev/archive/` and `reproduction_package/` finds nothing left:
   - `src/syn_grid/__main__.py`: the `USE_LEGACY` switch, so it runs the rebuild alone.
   - `pyproject.toml`: ruff's `extend-exclude`, coverage's `omit`, pyright's `exclude` and mutmut's `do_not_mutate`,
     each with its comment.
   - `landscape-changes.md`, whose only use is checking the rebuild against legacy, and its rows in `AGENTS.md`.
   - `AGENTS.md`: "The two strains" and the other lines on legacy; `procedures/legacy.md`; the glossary's "Strain"
     and "Legacy"; the code map and this table's legacy rows; the comment in `observation_handler.py` naming the
     perceptions still in `legacy/`.
   - Decisions that only say why legacy is kept, moved to `docs/dev/archive/` by its `README.md` or left as history.
   - `reproduction_package/`, which goes with it (see "The thesis runs live at the `v0.4.0` tag"); `README.md`'s
     pointer then sends readers to the tag alone, and the glossary loses "Reproduction package".

**Done when:** each perception stays inside its observation space, checked by the last of the world's Hypothesis
invariants, added to `TestInvariants` in `tests/integration/test_env_step.py`; an agent trains on each new one; and
nothing imports `legacy/`.
