# RPPO spatial-scenario regression

Bug hunt, separate from the v1.0.0 list in `todo.md`. Different lifetime: this
either closes or turns into a thesis erratum.

## Status: cause identified, seed 3 recovered

**`chain_break_penalty: 0.0` reproduces the thesis-era training curves on seed 3
RPPO.** The regression was a reward-configuration value, not code drift.

The relationship is non-monotonic, and the best setting is the one that looks
wrongest:

| `chain_break_penalty` | outcome |
|---|---|
| `-0.01` | flatlines |
| `-0.1` (the thesis-era value) | learns, onset ~7M |
| `0.0` | **matches thesis-era curves** |

Mechanism, **hypothesis not yet established**: at a nonzero penalty the agent
learns an aversion to entering any terminal state, and since consuming an orb out
of order *is* terminal, that aversion generalises to avoiding orbs — and avoiding
orbs means never performing the only action that pays. At `0.0` a
terminal-after-orb state is reward-neutral, so the only thing left to learn from
is the completed chain, and there is no learned aversion to work against.

The prediction made before the run — that `0.0` would be *worse*, because it
strengthens the fail-fast incentive — was wrong. The fail-fast incentive is
pointed at the right behaviour: fast termination *via orb contact*. The `-0.1`
failure mode was not fail-fast, it was orb avoidance.

**Treat the mechanism as unsettled.** A chain break terminates the episode at
`0.0` and at `-0.1` alike; the only difference is 0.1 on a scale where success
pays +10. A gap that small producing a complete behavioural flip is plausible —
terminal signals are learned early and consistently, so a reliable `-0.1` on
every terminal transition may be a stronger early prior than its nominal size
suggests — but it is equally consistent with run-to-run variance, and single runs
cannot separate the two. RPPO on a POMDP is high-variance by construction, and
seed 7 flatlining at `-0.1` is direct evidence of that spread.

Before any of this goes in a write-up it needs seeds, not one run per value.

Caveats, all real:
- Verified on **seed 3 only**. Seeds 42 and 7 not re-run.
- One run per configuration, so config and run variance are fully confounded. A
  matched-seed sweep is the minimum that separates them.
- The comparison rests on a visual read of the curves, not a numeric match.

## Symptom (historical)

During the mid-thesis period, RPPO solved the spatial scenario on seed 3, starting
to learn at ~1M steps. That is the only reproducible run. With
`chain_break_penalty: -0.1` and timeout `-1` (matching May), RPPO started learning
at ~7M on seeds 3 and 42, and seed 7 flatlined for the full 8.5M. PPO reproduced
May-era results. Curves looked more stochastic than May-era curves.

Seeds 7 and 42 for the May run were never recorded, so "only seed 3" may be an
artefact of what was kept rather than what happened.

## Established

**The environment has not drifted.** `scripts/replay_probe.py` replays a fixed
seeded action sequence through a given revision. May 16 (`15c225f`) vs the current
working tree, 300 single-chain episodes:

```
obs         identical over 6359 steps
ep_lens     identical over 300 entries
terminated  identical over 6059 entries
truncated   identical over 6059 entries
rewards     6 of 6059 steps differ, each by exactly +9.000
```

**The good run's config is in the repo**, at
`reproduction_package/spatial_scenario/5x5/rppo.yaml`. Compared knob-by-knob
against `configs.yaml` it is identical on 25+ values — grid, tiers, all scoring
flags, `perception`, `max_steps`, every penalty, growth factor, base reward.
Exactly one differs:

| | good run | now |
|---|---|---|
| `chain_break_penalty` | `-0.1` | `0.0` |

So this is close to a single-variable experiment, and the variable is already
known. It is not a code regression.

**Provenance caveat.** The link between the good run and `15c225f` is the artifact
filename `..._seed3_RPPPO_260516_09-56-33.zip`. That is a timestamp, not a
verified link. The replay probe measures behaviour and cannot say which code
produced that checkpoint. Loading the checkpoint and checking which config it
responds to would close this, and is the one thing the harness cannot substitute
for.

## Ruled out

- **Environment/observation drift** — byte-identical given `-0.1`.
- **`OrbFactory._normalize_counts` inversion** — `orb_factory.py` is unchanged
  between May 7 and May 16 (same blob), so the bug predates the working run.
  Latent then, not the regression.
- **`_max_reward_bonus` leak** — real bug, but on the `_step_wise_scoring` path.
  The spatial config uses `max_tier_scoring`. The `+9` the probe measured *was*
  this leak firing, because the probe was pinned to `grid_world_conf.max_tier_scoring`
  while the engine reads the `tier_orb_conf` copy — so the probe ran the wrong
  scoring mode. Not on the spatial path.
- **`truncated` never `True`** — also dead at May 16, so not the regression. See
  below; it may still matter.

## Reward landscape (why inaction is rational)

In `max_tier_scoring` mode:

| outcome | reward | ends episode? |
|---|---|---|
| complete chain | `+consumed_orb.REWARD` | yes |
| correct partial chain | `0.0` | no |
| mis-order | `chain_break_penalty` | yes |
| timeout | `-1` (replaces the step reward) | yes |

Partial progress pays **nothing**, so the value function cannot distinguish
"consumed tiers 1–2 correctly" from "did nothing". Only the completed chain pays.

Worse, inaction is priced *above* failure:

```
mis-consume an orb at step ~10  ->  -0.1  and done
survive to the horizon at 100   ->  -1 + 99*(-0.001)  ~=  -1.1  and done
```

Failing fast is ~11x cheaper than failing slowly, so the return-maximising policy
is to bump an orb and take the chain break. That is "consume willy nilly to save
time". It is the correct optimum of this landscape, not an agent bug.

`gamma` is never overridden, so SB3's default `0.99` applies. Discounting puts the
crossover at roughly a 3–4% believed chance of completing the chain; early in
training, in a POMDP, that is plausibly below threshold.

## Open suspects

1. ~~**`chain_break_penalty` magnitude.**~~ **RESOLVED** — see Status. `0.0` is the
   value that reproduces the May result. Still worth running seeds 42 and 7 at
   `0.0`, and worth establishing whether the `-0.01` flatline was a real effect
   or a third confound.
2. **`truncated` never `True`.** Weakened as an explanation: if `0.0` also
   recovers seed 7, the same mechanism accounts for it and this is no longer
   implicated. Still a genuine handicap — timeouts report as `terminated`, so the
   value function is trained to the terminal value with no bootstrapping.
3. **Run identity.** If a run had `continue_training: true` and the id/glob
   resolved to the wrong checkpoint, it starts from a mismatched policy with no
   error message. Known live instance: `spatial_scenario/6x6/*.yaml` ships
   `id_tag: 'spatial_5x5'`. Verify every run being compared.
4. **Training-side.** The Sep 2–22 SB3 refactor — hyperparameter dicts, LSTM
   policy kwargs (`lstm_hidden_size: 256`, `n_lstm_layers: 1`,
   `shared_lstm: False`), `RecurrentExecutionStrategy` state handling. Now much
   less likely: one config value explains the symptom. Still uninspected.
5. **Dependency drift.** SB3 2.9.0 / sb3-contrib 2.9.0 / torch 2.13.0 now, no
   lockfile. Less likely for the same reason, but not excludable: the May result
   was produced on an unrecorded dependency set, so "recovered" means "recovered
   on today's stack", not "recovered on May's".

## Next steps, cheapest first

- **Re-run seeds 42 and 7 at `0.0`.** The highest-value check. If all three seeds
  recover, this closes as a one-cause fix.
- **Quantify the curve match.** "Mirrors the thesis curves" should become numbers
  (onset step, asymptotic return, run-to-run variance) before it goes in a write-up.
- **Log orbs-consumed-per-step**, not just return. This is the measurement that
  discriminates the mechanism: it should be *high* at `0.0` and materially lower
  at `-0.1`. It turns the explanation above from a story into a measurement.
- **Load the May seed-3 checkpoint** into current code. Closes the provenance gap,
  and it is the one check the replay harness cannot substitute for.
- **Verify checkpoint identity** on every run in the comparison. Free.
- **Diff the resolved hyperparameter dicts and policy kwargs** between `15c225f`
  and now. Cheap, and the only suspect left wholly uninspected.

## History of the values

| date | commit | effective `chain_break_penalty` |
|---|---|---|
| May 7 | `d168a4f` | `0.0` |
| May 16 | `15c225f` | `-0.1` |
| May 22 | `5bc4e1e` | `0.0` again |
| May 28 | `3ce5eed` | configurable |
| Sep 22 | `d439509` | `-0.01` |
| now | local | **`0.0` — recovers the May result on seed 3** |

Worth sitting with: the May 16 commit the good run is attributed to carries `-0.1`,
but the setting that reproduces the May *result* is `0.0` — the value the
surrounding commits used. Either the good run predates the config it is being
compared against, or the attribution to `15c225f` is wrong. Both land on the same
weak link: the filename timestamp, which is load-bearing for the whole comparison.

Timeout penalty was `-1` from May 16 onward; `timeout_penalty` was introduced in
`d439509` (Sep 22) and reverted locally. Back to May values, and not implicated in
the recovery.

## For the write-up

If the seed sweep holds, the result is stronger than "I fixed a regression". In a
sparse-reward POMDP where partial progress pays nothing, a nonzero penalty on a
*recoverable* error may suppress exploration of the only rewarding action more
than it reinforces correct behaviour — and removing the penalty does not make the
task easier, it removes a signal that was preventing the agent from finding the
solution at all.

That would also collapse the three anomalies into one curve: `-0.01` flatlining,
`-0.1` learning at 7M, and `0.0` learning fast become points on a single
exploration-suppression relationship rather than three separate bugs.

**The claim depends entirely on that sweep.** If seed 42 at `0.0` is also slow, or
seed 7 at `-0.1` occasionally learns, then what you have is high run variance plus
one config value, and the honest write-up is about variance rather than about
penalties. Both are publishable. They are not the same paper.

## A trap in the config

```yaml
step_penalty: &step_penalty -0.001
tier_consumption_penalty: *step_penalty
```

The anchor couples them. `step_penalty` went `-1.0 -> -0.001` in this window, so
`tier_consumption_penalty` silently moved 1000x with it. Break the alias before
running any penalty sweep.
