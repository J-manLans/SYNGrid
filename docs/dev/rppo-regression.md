# RPPO spatial-scenario regression

Superseded in part — see "Correction" below. The ruled-out list and the
provenance gap still stand. Bug hunt, separate from the v1.0.0 list in `todo.md`.

## Correction (2026-09-27)

The original conclusion was that `chain_break_penalty` was the single variable
and that `0.0` reproduced the thesis-era curves. **That framing was wrong**, and
the error came from a code defect rather than from the sweep itself.

`check_episode_end` took a parameter named `timeout_penalty`, but its caller
passed `chain_break_penalty` into it. There has never been a configurable timeout
penalty in the config schema. The consequence:

> Until `c5805b1`, the timeout reward and the chain-break reward were **the same
> value**, so their ratio was pinned at exactly 1:1 and no config edit could
> change it.

So every row of the old table below described a *different* reward landscape
than intended, and not in a uniform way:

| `chain_break_penalty` | timeout reward, thesis era | ratio |
|---|---|---|
| `0.0` | `0.0` — no penalty for running out the clock at all | n/a |
| `-0.01` | `-0.01` | 1:1 |
| `-0.1` | `-0.1` | 1:1 |

The old table's "flatlines" and "matches thesis-era" outcomes were therefore
measurements of three different landscapes, not of one knob. Reading them as a
penalty-strength sweep cannot be supported.

What now works is `chain_break_penalty: -0.01` with `timeout_penalty: -1.0` — a
**100:1** ratio that was simply unreachable while the two shared a field. Both
landscapes appear to train; the point is that the ratio is the free variable, and
it was pinned.

The timeout reward is now a real config field (`DroidConf.timeout_penalty`).

## The original result, for the record

**`chain_break_penalty: 0.0` reproduced the thesis-era training curves on seed 3
RPPO.** Seed 7 also appeared to learn, where it previously flatlined for the
full 8.5M steps. Read this table as history, not as guidance.

| `chain_break_penalty` | outcome |
|---|---|
| `-0.01` | flatlines |
| `-0.1` (the value in the May config) | learns, onset ~7M |
| `0.0` | matches thesis-era curves |

## Mechanism (hypothesis, not established)

At a -1 `timeout_penalty` and a -0.01 `chain_break_penalty` the agent learns that
being inactive is costly, but so is picking the wrong orb as well, but not so costly
it's not worth the effort. Let's say the agent collects an orb out of order at step 50.
It has then received 0.001 * 50 in step penalty, and a 0.01 penalty for the wrong orb.
This equals to a penalty of 0.06, still less than avoiding orbs all together, but a pointer
to get the order right. And the longer the episode gets, the more costly it is to collect orbs
out of order.

## Ruled out

Worth keeping so nobody re-investigates them.

- **Environment / observation drift** — byte-identical given `-0.1`.
- **`_max_reward_bonus` leak** — on the `_step_wise_scoring` path; the spatial
  config uses `max_tier_scoring`. The `+9` the probe measured *was* this leak
  firing, because the probe was pinned to `grid_world_conf.max_tier_scoring` while
  the engine reads the `tier_orb_conf` copy. Not on the spatial path.
- **`truncated` never `True`** — also dead at May 16, so not this. Now decided
  rather than open: timeouts are terminations by design, so no value function is
  bootstrapped at the horizon. Pinned in
  `tests/gymnasium/utils/test_episode_termination.py`.

No longer live as suspects: run identity, the Sep 2–22 SB3 refactor, dependency
drift. One config value explains the symptom. Dependency drift is not *excluded*
— the May run used an unrecorded dependency set, so "recovered" means "recovered
on today's stack".

## Next steps

1. Redo the sweep one variable at a time, treating the **ratio** as the primary
   factor: `chain_break_penalty ∈ {0.0, -0.01, -0.1}` ×
   `timeout_penalty ∈ {-0.1, -1.0}` × seeds `{3, 7, 42}`.
2. Quantify the match — onset step, asymptotic return, variance. "Mirrors the
   curves" needs to become numbers before it goes anywhere.
3. Log orbs-consumed-per-step **and timeout rate**. It should be *high* at `0.0`
   and lower at `-0.1`; that is the measurement which turns the mechanism above
   into evidence.
4. Note that `max_steps` is 100 at 5x5, 6x6 and 7x7 alike, and the chain length
   is 5 at all three. Only the grid area grows, so the timeout rate rises
   mechanically with size. Log it separately or "worse at 7x7" will be partly
   "times out more", not purely worse spatial reasoning.
5. Optionally, load the May seed-3 checkpoint into current code.
6. Verify the provenance transcription above against git history *before*
   `reproduction_package/` is removed.

## A note on the config

`configs.yaml` aliases `tier_consumption_penalty` to `step_penalty` via
`&step_penalty` / `*step_penalty`, so changing one silently rescales the other.
It moved 1000× in this window. See the config-traps section of `AGENTS.md` —
break the alias before running any further penalty sweep. `timeout_penalty` is
deliberately a literal and not an alias, and a test enforces that.
