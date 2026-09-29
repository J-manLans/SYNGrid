# Scenario as a first-class domain concept

The Scenario refactor, and what it deliberately did not do.

## The problem

SYNGrid is a benchmark, so what varies between experiments *is* the work. That
variation had no name. A scenario was reconstructed, independently, at every
site that needed one:

| Site | How it inferred the scenario |
|---|---|
| `OrbFactory.create_orbs` | `if single_chain_mode` → a different pool construction |
| `GridWorld.reset` | `if single_chain_mode` → fill the field, or place one orb |
| `GridWorld.perform_droid_action` | three more conditionals, incl. `delay_mode` |
| `BasePerception._get_observable_orb_count` | `curriculum_training` × `single_chain_mode` |
| `BasePerception._sort_orbs_by_manhattan_dist_to_droid` | `single_chain_mode` again |
| `episode_termination.check_episode_end` | `single_chain_mode`, `max_tier_scoring`, `curriculum_training` |

Six sites, one bag of booleans, no owner. The failure mode was not that the
flags were wrong but that two sites could read the same config and conclude
different things — and did. `max_tier_scoring` existed in `GridWorldConf` and in
`TierConf`; the digestion engine read one and episode termination read the
other, and the YAML anchor that was supposed to keep them equal was the only
thing stopping a silent divergence. `GridWorldConf`'s validator overwrote
`max_active_orbs` with `max_tier` through `object.__setattr__` on a frozen
model, so the number in the YAML was routinely a lie — the scaling scenarios
all said `max_active_orbs: 3` and ran with 5 and 6.

## The shape of the answer

```text
SYNGridEnv ──▶ Scenario ──┬──▶ OrbPopulation   how the pool is built
                           ├──▶ SpawningRules   how the field behaves
                           ├──▶ ObservationRules how wide the observation is
                           └──▶ TerminationRules when the episode ends
```

`Scenario` is a composition root and holds no behaviour. It is a frozen
dataclass of a name, a kind, and four collaborators. The domain hierarchy —
`Goal → Tier Chain → {Spatial, Delay, Tier Scaling}` alongside `Continuous` — is
expressed by *which rules get composed*, not by a class hierarchy. Spatial,
Delay and Tier Scaling share almost no mechanics; the only real fork in the tree
is goal-versus-continuous, and even that is two rule objects rather than two
types. Inheriting `Goal → TierChain → Spatial` would assert a relationship the
code does not have.

One decision worth stating because it looks like an omission: **digestion is not
scenario-owned.** All three scoring modes are available to every scenario, and
the engine selects between them from the orb, identically everywhere. The issue
listed digestion as a candidate boundary; reading the actual behaviour said no,
and inventing a per-scenario digestion rule to fill the diagram would have been
worse than the gap. What the engine *did* get is `ScoringMode` — one enum
replacing three mutually exclusive booleans plus the duplicate copy — and public
accessors, because episode termination had been reaching into `_pending_reward`
from another package.

## What went away

Config fields that identified a scenario, all removed rather than deprecated:

| Field | Why it could not stay |
|---|---|
| `single_chain_mode` | named a mechanism, not a scenario; duplicated in 3 blocks |
| `delay_mode` | delay is what makes `goal_tier_chain_delay` a *different scenario*. A flag restating that could only contradict the scenario name |
| `termination_on_max_tier` | validated, then read by nothing. A completed chain ended the episode because the scoring mode said so |
| `curriculum_training` | a scenario property in practice, and it only ever reached the observation when a chain was driving the world |
| `max_tier_scoring` (world copy) | the duplicate the engine could not see |
| `step_wise/threshold/max_tier_scoring` | collapsed into `TierConf.scoring: ScoringMode` |
| `PerceptionConf.{max_tier,max_active_orbs,curriculum_training,single_chain_mode}` | world-derived counts; the scenario states them |

`GridWorldConf`'s whole validator went with them. It existed to reject
combinations of scenario flags, and the scenario now rejects them against
itself — a tier chain refuses `de_spawn_tiers` with a message that says why
("a chain that loses a link to a timer is not a chain"), rather than a config
model asserting a rule about a flag that named nothing.

## Two behaviours preserved that look like bugs

Both are pinned by tests, because both read as defects and a future reader would
"fix" them.

**The spatial scenario's observation is wider than its world.** Under curriculum
the observation is sized for `tiers` (5) orb slots while the distance sort that
fills them still caps at `max_tier` (3). Two slots exist and stay zero forever.
The old code computed these two numbers in two different places from two
different expressions; `ObservationRules` keeps them as two named fields rather
than collapsing them, and the asymmetry is the point of the naming.

**`continuous` ignores `curriculum_training`.** Recorded because it is easy to
assume otherwise. The setting only ever reached the observation when
`single_chain_mode` was also set, so a continuous scenario with curriculum on is
the same world with it off. The behaviour lock found this: two scenarios
produced a byte-identical digest.

## What the refactor cost

Two globals went away, both found by tests that could not be written otherwise:

- **`TierOrb.max_tier` is now a constructor argument.** It was a class attribute
  written by `OrbFactory` before every pool was built, so constructing a
  `TierOrb` was only legal immediately after constructing a factory — the code
  that decides *which orbs a scenario has* could not be exercised without
  standing up the thing that calls it. Two worlds in one process also
  overwrote each other's ceiling.
- **`BaseOrb._life_span` stays global.** It is genuinely world-wide (the grid's
  Manhattan diameter) and a perception snapshots it at construction, which
  remains a latent trap. Out of scope here; it is now the *only* one.

`SpawningRules` also had to be made comparable by value. Its three strategies
are stateless frozen dataclasses, because otherwise two rules built for the same
world compared unequal purely for holding different strategy instances, and a
scenario became impossible to assert about.

## Verification

Behaviour preservation is not asserted, it is measured. `scripts/scenario_lock.py`
records a fingerprint per scenario from a fixed action tape and committed
digests in `scripts/scenario_lock_baseline.json`; `--verify` exits non-zero on
any drift. 17 scenarios, all reproducing their pre-refactor digests exactly.

Two findings from building it are worth keeping:

- **`delay=30` at 60 max-steps means `timeout_penalty / 2` is never executed by
  any shipped scenario.** The delay cooldown always eats the horizon, so the
  "field exhausted with time left" branch is dead in every archived config.
  `tier_chain_delay_fast` reaches it (26 hits).
- **`continuous + delay` was reachable but never configured.** `delay_mode` was
  an independent flag, so nothing had recorded what that combination did. It is
  now the `continuous_delay` scenario, and its digest was recorded against
  `cleanup-after-versioning` via a worktree — pinned to what the old flag
  combination did, not to whatever the refactor produced.

The original `replay_probe.py` was independently pointed at the pre-refactor
worktree and the recordings compared, at 300 episodes on both roots:

| | rewards | terminated | ep_lens | obs |
|---|---|---|---|---|
| `goal_tier_chain_spatial` | identical, 7714 | identical, 7714 | identical, 300 | identical, 8014 steps |
| `continuous` | identical, 18000 | identical, 18000 | identical, 300 | identical, 18300 steps |

Reward sums `-8922.900` and `-19960.000` on both sides, delta `+0.000`. Both
probe scripts keep a two-sided signature shim, so they still drive
pre-refactor revisions.

### `replay_sweep.sh` cannot cross this boundary, and that is structural

Worth recording so nobody re-derives it. The sweep resolves `--config-src` from
the current checkout, so its baseline row — the pre-refactor commit — is fed the
post-refactor `test_configs.yaml` and dies on `missing`. The reason is not
fixable by pinning: **one config file cannot satisfy both schemas.** Give the old
code the new file and it fails on missing `single_chain_mode`; give the new code
the old file and it fails on missing `scenario`. `extra='ignore'` does not help —
an unknown `scenario` is dropped silently by the old schema, but it is the
*required* keys that break.

So a cross-boundary sweep needs two config files expressing the same world, which
is what the table above is: `goal_spatial_pre.yaml` against
`scenarios/tier_chain_spatial.yaml`, and the pre-refactor `test_configs.yaml`
against `scenarios/continuous_step_wise.yaml`. The sweep remains the right tool
for sweeps *within* one schema.

## Known gaps

- **`Cancellation` does not exist.** It appears in the intended hierarchy but has
  no implementation anywhere in the codebase. It is absent here rather than
  invented.
- **Effect orbs do not exist.** `core/orbs/effects/` holds only a `dummy_file`.
  The `Continuous` scenario's children in the hierarchy are tier and negative
  orbs only.
- **Run ids are unchanged.** `_set_models_base_id` still derives identity from
  perception, grid size and enabled orb types rather than from the scenario
  name. Folding the scenario in would orphan every existing checkpoint path,
  and artifact naming is a compatibility surface the issue did not ask to move.
  Worth doing deliberately, with a migration.
- **`GridWorldConf.max_active_orbs` is now only read by continuous scenarios.**
  A goal scenario derives its field size from `max_tier` and ignores it. It is
  kept so the field stays expressible, and `tests/scenario/test_registry.py`
  pins that the derived value wins.
