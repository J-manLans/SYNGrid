# Landscape changes

Changes in the rebuild that could move what the agent sees or what it is paid, and so could matter for how it learns.
This is the list to read when the legacy comparison (phase 6) shows a difference, or when a rebuilt scenario learns
worse than legacy did.

It is not a list of bugs. Most entries are decisions made on purpose, here because they are easy to forget.
`design-decisions.md` says why; an entry here says what could change in the data the learner gets, and what to check.

Add an entry when a change touches the reward, the observation, the layout of orbs or how often something appears.
Leave out a change that only moves code. An entry leaves once it is checked against legacy, and the file goes when
legacy is deleted.

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
**Expected size:** Small. Only an orb with a lifespan can expire, so in the single tier chain family it is the negative orb.
The droid now pays its penalty in a case where it used to escape it.
**Status:** on purpose.
**Check:** Phase 6, with the negative orb, and with spatial if legacy ran it with one at all; if it did not, this never
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

### Delay freezes the field after the step

**Change:** Delay's watcher freezes the orbs after the world's step, not in the middle of its loop over the orbs.
**Touches:** how often something appears, observation
**Before:** Legacy froze the field when the orb was eaten, partway through the loop, so the orbs after it in
`ALL_ORBS` order ticked once more on that step and returned a step earlier than the ones before it.
**Now:** Every frozen orb waits at least `delay_conf.steps` steps, the same for each.
**Expected size:** One step on a gap of 30 or more, for some orbs. Small; read from legacy's code.
**Status:** on purpose.
**Check:** Phase 6, when delay is trained against the thesis delay runs.

### A frozen orb waits for its cell

**Change:** A frozen orb returns only when its cell is free of the droid and of orbs on the field.
**Touches:** how often something appears, observation
**Before:** Legacy brought every frozen orb back when its timer ran out, wherever the droid stood.
**Now:** An orb whose cell is taken stays frozen until the cell clears. Over 300 random-action episodes at a delay of
30, silences lasted 30 to 33 steps.
**Expected size:** A step or two on some silences, more often for a droid that waits on a frozen orb's cell.
**Status:** on purpose.
**Check:** Phase 6, with delay's training.

### Delay freezes only the eaten orb's kind

**Change:** Eating a tier orb freezes only the other tier orbs; the negative orb is never frozen.
**Touches:** observation, how often something appears
**Before:** Legacy froze every orb on the field on any eat. The thesis delay runs had the negative orb off, so this
was never trained.
**Now:** With the negative orb on, it stays on its own cycle during a silence.
**Expected size:** None for the thesis setups, which had it off.
**Status:** on purpose.

### Delay ends as the family does

**Change:** A broken chain ends a delay episode, at `chain_break_penalty`, and eating every tier orb is no ending.
**Touches:** reward, how often something appears
**Before:** Legacy's final delay ending kept a broken chain open, and ended at half the timeout penalty when every
tier orb was eaten without a completed chain. The thesis delay runs had the break open and no "every orb eaten"
ending.
**Now:** A wrong orb ends the episode at once, as in spatial and scaling, at a break penalty of -0.5 where the
thesis delay runs had -1.0.
**Expected size:** Large. A droid that breaks no longer gets to try again in the same episode, and the break penalty
becomes the main early lesson in the chain's order (see "Delay's values" in `design-decisions.md`).
**Status:** on purpose.
**Check:** Phase 6, against the thesis delay curves, knowing the ending and the spare steps both changed.

### Delay's steps follow from the delay

**Change:** Delay's `max_steps` is `2 * delay + 50` on tier 3 and 5x5, set by its config model.
**Touches:** how often something appears, reward
**Before:** The thesis delay runs had `max_steps` of 150, 200, 300, 550 and 850 for delays of 30, 70, 130, 260 and
390: 90, 60, 40, 30 and 70 spare steps.
**Now:** 50 spare steps at every delay: 110, 190, 310, 570 and 830.
**Expected size:** Large where the spare steps moved most. Delay 30 loses 40 steps and 390 loses 20, so early
timeouts come sooner; 130 gains 10 and 260 gains 20, so they should time out less early on. The ordering of the
thesis curves across delays may change, since part of it was the clock.
**Status:** on purpose.
**Check:** Phase 6. Compare each delay with its thesis run knowing its spare steps changed, and compare delays with
one another as the measurement.

---

## The observation

### An orb that never expires has no remaining time

**Change:** A tier orb's timer holds `None` while it is on the field.
**Touches:** observation
**Before:** Every orb had a number.
**Now:** Neither perception in `src` writes an orb's timer. The legacy perceptions that did are rewritten in phase 7.
**Expected size:** None for the two kept perceptions. A rewritten perception that writes the timer needs a number for
`None`.
**Status:** open. It is a question in `open-questions.md`, under phase 7.
**Check:** Phase 7. Whatever number or flag replaces `None` changes the observation, so it has to be compared with
legacy.

### Where the step count comes from

**Change:** The world keeps the step clock. The observation handler keeps no count since 2026-10-23, and no
perception in `src` observes the steps left.
**Touches:** observation
**Expected size:** None for the two kept perceptions. Where a global feature reads `world.steps_left`, an off-by-one
against legacy would shift that feature by a step.
**Status:** not checked.
**Check:** Phase 7, when a global feature reads the world's count.

### The dense identity

**Change:** The identity remap stays in the world until the observation pass.
**Touches:** observation
**Status:** open, with a `# TODO(phase 7)` in `grid_world.py`.
**Check:** Phase 7.

### The observation carries no chain progress and no orb timers

**Change:** None against legacy's `vector_fog_of_war` and `vector_markovian`, which never carried either. Recorded
so that a rewritten perception or a continuous scenario does not assume the agent sees them.
**Touches:** observation
**Before:** In legacy the chain's tier was a global value in `vector_markovian_easy` and the composite perceptions,
and the timer was an option in `vector_markovian_easy` and `composite_markovian` and always in
`composite_grid_markovian`. The owner's ablation before the rebuild left both out of the vector observation.
**Now:** The agent sees the droid's position and each orb slot's active flag, row, column and identity, as in legacy's
kept perceptions. A phase 6 comparison on them shows no difference here.
**Expected size:** None where one chain lies on the board, since the orbs still there show how far the agent got. It
matters where tier orbs return, as in a continuous scenario: a stateless agent has no way to know where it is in the
chain. For the delay scenario the missing timer is the measurement and is on purpose.
**Status:** on purpose. Chain progress for a continuous scenario is an open question.
**Check:** When a continuous family is written, or when a perception is rewritten in phase 7.

### Position bounds are exact

**Change:** The upper bound of a row and a column in the `Box` is `rows - 1` and `cols - 1`, for the droid and for
every orb slot.
**Touches:** observation
**Before:** The bound was `rows` and `cols`, one more than any position can be.
**Now:** The bound is the largest position. The values observed are unchanged.
**Expected size:** None expected. The runners wrap the env in `VecNormalize(norm_obs=True)`, which scales by running
statistics and not by the bounds, and SB3 rescales a `Box` by its bounds only for an image space.
**Status:** expected same, not tested.
**Check:** Phase 5, `check_env` on the spatial scenario. Revisit if a perception or feature extractor ever divides by
`observation_space.high`.

---

## The reward

The reward changes decided on 2026-10-11 and after are listed in "Legacy is a reference, not a target" in
`design-decisions.md`. They are on purpose and are not repeated here. In short: a max-tier chain pays what its orbs are
worth and not a fixed 10, a negative orb no longer breaks a chain, and an ending adds to the step's reward where
legacy's timeout replaced it.

Compare curves, not rewards, and set the base reward so the two sides are of similar size first. That is the roadmap's
advice for phase 6.
