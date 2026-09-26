# RPPO spatial-scenario regression

Closed: cause identified. Bug hunt, separate from the v1.0.0 list in `todo.md`.

## Result

**`chain_break_penalty: 0.0` reproduces the thesis-era training curves on seed 3
RPPO.** Seed 7 also appears to learn, where it previously flatlined for the full
8.5M steps. The regression was a reward-configuration value, not code drift.

| `chain_break_penalty` | outcome |
|---|---|
| `-0.01` | flatlines |
| `-0.1` (the value in the May config) | learns, onset ~7M |
| `0.0` | matches thesis-era curves |

## Mechanism (hypothesis, not established)

At a nonzero penalty the agent learns an aversion to entering terminal states,
and since consuming an orb out of order *is* terminal, that aversion generalises
to avoiding orbs — and avoiding orbs means never performing the only action that
pays. At `0.0` a terminal-after-orb state is reward-neutral, so the only thing
left to learn from is the completed chain, with no learned aversion against it.

Treat this as unsettled. Termination is identical at `0.0` and `-0.1`; only the
0.1 differs, on a scale where success pays +10. A gap that small producing a
complete behavioural flip is plausible — terminal signals are learned early and
consistently — but equally consistent with run-to-run variance, and single runs
cannot separate the two. RPPO on a POMDP is high-variance by construction.

**Do not write the mechanism up before the seed sweep.** If seed 42 at `0.0` is
also slow, or seed 7 at `-0.1` occasionally learns, the honest result is variance
plus one config value, not penalties suppressing exploration. Different paper.

## Why it was the config and not the code

`scripts/replay_probe.py` replays a fixed seeded action sequence through a given
commit. May 16 (`15c225f`) vs the current tree, 300 single-chain episodes:

```
obs         identical over 6359 steps
ep_lens     identical over 300 entries
terminated  identical over 6059 entries
truncated   identical over 6059 entries
rewards     6 of 6059 steps differ, each by exactly +9.000
```

The environment had not drifted. And the good run's config is still in the repo
at `reproduction_package/spatial_scenario/5x5/rppo.yaml`: identical to
`configs.yaml` on 25+ values — grid, tiers, all scoring flags, `perception`,
`max_steps`, every penalty, growth factor, base reward — with exactly one
difference, `chain_break_penalty`. So it was close to a single-variable
experiment, and the variable was already known.

## Ruled out

Worth keeping so nobody re-investigates them.

- **Environment / observation drift** — byte-identical given `-0.1`.
- **`OrbFactory._normalize_counts` inversion** — `orb_factory.py` is unchanged
  between May 7 and May 16 (same blob), so it predates the working run. Latent
  then, not this. Still a real bug; see `todo.md`.
- **`_max_reward_bonus` leak** — on the `_step_wise_scoring` path; the spatial
  config uses `max_tier_scoring`. The `+9` the probe measured *was* this leak
  firing, because the probe was pinned to `grid_world_conf.max_tier_scoring` while
  the engine reads the `tier_orb_conf` copy. Not on the spatial path.
- **`truncated` never `True`** — also dead at May 16, so not this. Still a real
  handicap (no bootstrapping at the horizon) and the next thing to look at.

No longer live as suspects: run identity, the Sep 2–22 SB3 refactor, dependency
drift. One config value explains the symptom. Dependency drift is not *excluded*
— the May run used an unrecorded dependency set, so "recovered" means "recovered
on today's stack".

## Remaining uncertainty

**Provenance.** The link between the good run and `15c225f` is the artifact
filename `..._seed3_RPPPO_260516_09-56-33.zip`. That is a timestamp, not a
verified link, and it is load-bearing: `15c225f` carries `-0.1`, yet `0.0` is
what reproduces its *result*. Either the run predates the config it is compared
against, or the attribution is wrong. Loading the checkpoint and checking which
config it responds to would settle it, and is the one check the replay harness
cannot substitute for.

**The seed-3 residual.** `0.0` is close but not exact — the May run learned a
little faster. Consistent with the provenance gap above. Whether that gap is the
penalty or just run variance is what the seed sweep decides.

## Next steps

1. Finish the seed sweep: 42 and 7 at `0.0`, matched seeds, plus 7 at `-0.1`.
2. Quantify the match — onset step, asymptotic return, variance. "Mirrors the
   curves" needs to become numbers before it goes anywhere.
3. Log orbs-consumed-per-step. It should be *high* at `0.0` and lower at `-0.1`;
   that is the measurement which turns the mechanism above into evidence.
4. Optionally, load the May seed-3 checkpoint into current code.

## A note on the config

`configs.yaml` aliases `tier_consumption_penalty` to `step_penalty` via
`&step_penalty` / `*step_penalty`, so changing one silently rescales the other.
It moved 1000× in this window. See the config-traps section of `AGENTS.md` —
break the alias before running any further penalty sweep.
