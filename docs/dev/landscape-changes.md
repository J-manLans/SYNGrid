# Landscape changes

Changes in the rebuild that could move what the agent sees or what it is paid, and so could matter for how it learns.
This is the list to read when the legacy comparison (phase 6) shows a difference, or when a rebuilt scenario learns
worse than legacy did.

It is not a list of bugs. Most entries are decisions that were made on purpose, and an entry is here because it is easy
to forget. `design-decisions.md` says why a decision was made, and `regressions.md` holds behaviour that broke. An
entry here says what could change in the data the learner gets, and what to check.

Add an entry when a change touches the reward, the observation, the layout of orbs or how often something appears.
Leave out a change that only moves code.

---

## Entry template

**Change:**
**Touches:** reward / observation / layout / how often something appears
**Before:**
**Now:**
**Expected size:** how much it should matter, and for which scenarios
**Status:** on purpose / expected same, not tested / not checked / open
**Check:** what to look at, and in which phase

---

## The world (2026-10-19)

### The droid wins an orb on its last tick

**Change:** An orb that expires on the same step the droid lands on it is eaten, not despawned.
**Touches:** reward
**Before:** The expiry was checked first, so the droid got nothing and the orb went. It looked as if the droid had
reached it.
**Now:** The contact check comes first, so the droid is paid.
**Expected size:** Small. Only an orb with a lifespan can expire, so in the tier chain family it is the negative orb.
The droid now pays its penalty in a case where it used to escape it.
**Status:** on purpose.
**Check:** Phase 8 with the negative orb. Phase 6 if legacy spatial ran with one at all; if it did not, this never
reaches the comparison.

### The negative orb returns in a single chain

**Change:** The world refills the field in every scenario. A despawned or eaten orb that is ready comes back in a
random free cell, one per step.
**Touches:** reward, how often something appears
**Before:** The old step refilled only when `single_chain_mode` was off. In a single chain with no delay nothing was
ever put back, so a negative orb would have come once and then never again. This is read from the old code, which no
longer runs.
**Now:** It comes back after its cool-down, as the config's own comment says it should.
**Expected size:** Possibly large for a scenario with a negative orb. It decides how often the agent meets a penalty.
**Status:** on purpose, and not tested against legacy.
**Check:** Phase 6 and 8. Count the negative orbs eaten per episode on both sides, which is what the
`negatives_eaten` metric is for. Confirm first that legacy ran a negative orb in a single chain.

### The field starts full in every scenario

**Change:** `reset` fills the field up to `max_active_orbs`.
**Touches:** layout
**Before:** A scenario that was not a single chain started with one orb and grew by one per step.
**Now:** Every scenario starts full.
**Expected size:** None for the spatial scenario, which already started full. It matters for a continuous scenario
that would want a ramp.
**Status:** on purpose.
**Check:** When the first continuous scenario is written.

### Refill is one orb per step

**Change:** Not new, but now a rule of the world and not of a mode. When several orbs leave the field on one step, the
field takes several steps to fill again.
**Touches:** how often something appears
**Expected size:** None where one orb returns at a time, as with the negative orb.
**Status:** on purpose.
**Check:** A continuous scenario, if its field turns over fast.

### The layout under a seed

**Change:** The world picks orbs from a filtered list of all orbs, and not from a list that orbs were moved in and out
of.
**Touches:** layout
**Before:** A random choice from the inactive list, in the order orbs had left it.
**Now:** A random choice among the ready orbs, in the order of `ALL_ORBS`.
**Expected size:** At reset the two lists hold the same orbs in the same order, so the same seed should give the same
layout. During an episode the only orb that returns is the negative orb, so there is nothing to choose between.
**Status:** expected same, not tested. The world was checked to be deterministic, not to equal the old layout.
**Check:** Phase 6, by comparing the first layout for a seed on both sides.

### `reset` needs an rng

**Change:** The unseeded fallback is gone.
**Touches:** layout
**Before:** A reset with no generator re-rolled the layout every time.
**Now:** The environment passes its seeded generator. Nothing else calls `reset`.
**Expected size:** None, if every run goes through the environment. A script that calls the world directly must pass
a generator.
**Status:** on purpose.

### Orbs on the field are listed in `ALL_ORBS` order

**Change:** `orbs_on_field()` replaced three getters.
**Touches:** observation, if anything reads the order
**Before:** The getters listed the active orbs in the order they were spawned.
**Now:** The order is that of `ALL_ORBS`.
**Expected size:** None today. The observation reads `ALL_ORBS` and `is_active` itself, and only `render` called the
getters. It would matter if the observation pass reads this list.
**Status:** not checked in the observation.
**Check:** Phase 7.

### Tier orbs cannot expire

**Change:** The `de_spawn_tiers` option is gone.
**Touches:** reward, observation
**Before:** A config could make tier orbs tick and despawn.
**Now:** The tier bundle gives them a `None` lifespan and no scenario in the family can change it.
**Expected size:** None for runs that left the option off. A scenario that wants it gives its tier orbs a lifespan.
**Status:** on purpose.

---

## The observation

### An orb that never expires has no remaining time

**Change:** A tier orb's timer holds `None` while it is on the field.
**Touches:** observation
**Before:** Every orb had a number.
**Now:** Two perceptions put `orb.TIMER.remaining` into the observation, and `base_perception.py` reads a class
attribute that is gone.
**Expected size:** Those two perceptions do not run unchanged.
**Status:** open. It is a question in `open-questions.md`, under phase 7.
**Check:** Phase 7. Whatever number or flag replaces `None` changes the observation, so it has to be compared with
legacy.

### Where the step count comes from

**Change:** The world keeps the step clock. The environment still counts on the observation handler.
**Touches:** observation
**Expected size:** An off-by-one in the steps left would shift one feature by a step.
**Status:** not checked.
**Check:** Phase 5, when the handler is made to read the world's count.

### The dense identity

**Change:** The identity remap stays in the world until the observation pass.
**Touches:** observation
**Status:** open, with a `# TODO(phase 7)` in `grid_world.py`.
**Check:** Phase 7.

---

## The reward

The reward changes decided on 2026-10-11 and after are listed in "Legacy is a reference, not a target" in
`design-decisions.md`. They are on purpose and are not repeated here. In short: a max-tier chain pays what its orbs are
worth and not a fixed 10, a negative orb no longer breaks a chain, and an ending adds to the step's reward where
legacy's timeout replaced it.

Compare curves, not rewards, and set the base reward so the two sides are of similar size first. That is the roadmap's
advice for phase 6.
