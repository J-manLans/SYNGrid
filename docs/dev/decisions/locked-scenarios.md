# Decisions: Locked scenarios

The values and rules particular to each locked scenario: dense's values, the ablations, and delay.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Dense locks with sparse's reward values, so only the scoring mode differs (2026-11-04)

**Decision:** Scaling dense has sparse's `base_reward` (1.0), `growth_factor` (0.5) and `step_penalty` (-0.001). The
thesis values (2.5, 1.685 and 0.0) are used only to verify dense against the thesis runs, and are not what dense is
locked with.

**Why:** Dense asks whether the silence stops learning. With the thesis values it also paid about 13 times sparse's
completed chain at tier 6 and had no step penalty, so a difference between the two did not say which caused it. With
the same three values, the scoring mode is the only difference, and "the chain's length was not the problem, the
silence was" is a claim the two scenarios can back. Dense learned on these values at tier 7 on 5x5 and on 6x6.

**Rejected:**

- The thesis values as dense's (two differences besides the scoring mode).
- Swapping the ladder and leaving the step penalty (a second difference would remain).
- Training sparse on dense's values instead (only needed if dense had failed on sparse's).

## The ablations kept the step penalty, and sparse's chain break penalty (2026-10-27)

**Decision:** Spatial and scaling sparse keep `step_penalty: -0.001`, and scaling sparse keeps
`chain_break_penalty: -0.1`, the thesis values. They are locked with the scenarios' other rules.

- Spatial on 5x5 with no step penalty did not learn: mean reward about 1.2 against a ceiling of about 9.02, where
  the run with the penalty reached about 8.5.
- Scaling sparse at tier 4 with no step penalty reached its maximum faster.
- Scaling sparse at tier 4 with no chain break penalty did slightly worse.

**Why the step penalty matters although it is small:** PPO normalises advantages in each minibatch, so the learner
sees how rewards differ, not how large they are. Early on almost every step pays 0, and the step penalty is nearly
the only thing that tells a short episode from an aimless one. Where completions are frequent from the start, the
discount is enough, which fits sparse at tier 4; where they are rare, the penalty carries the agent to its first
completions, which fits spatial. This explanation fits both results and is not tested on its own.

**Why sparse keeps it although tier 4 was faster without:** `max_tier` is a tunable, and the locked rules must
hold at every tier. Long chains make early completions rare, which is where spatial failed. The thesis ran -0.001 at
tiers 4 to 7.

**Why sparse keeps its break penalty:** a break ends the episode, so without a penalty it is a free way out, cheaper
than walking to the -1 timeout.

**Spatial's break penalty was not ablated.** It was tuned in the thesis era: under a lantern, eating an unknown
orb is how the droid explores, and a cheap break keeps that cheap. A larger penalty taught the droid to avoid orbs
and run out the clock.

**Rejected:** no step penalty in sparse on the tier 4 result alone; no step penalty in any goal scenario (refuted by
spatial); a tier 6 run before deciding (keeping the penalty is safe whatever it shows).

## Delay ends as the family does (2026-11-06)

**Decision:** Delay uses `SingleTierChainTermination`, as every scenario in the family does: a broken chain ends the
episode at `chain_break_penalty`, and there is no ending for eating every tier orb. The termination is one
`evaluate`, with no methods for a variant to override, and a scenario's entry names no termination.

**Why:** Delay started on legacy's final ending, a subclass that kept a broken chain open and ended at half the
timeout penalty when every tier orb was eaten, and the ablations walked it towards the family's. Legacy kept the
break open on 2026-05-28, when the break penalty was 0, so ending on a break would have been a free way out; the
penalty became -1.0 two days later and the rule was never revisited, and the "last orb eaten" ending was added after
the thesis runs. With a break ending the episode, delay learned at every delay from 30 to 390, and the half-timeout
ending could no longer fire at tier 3, since a break comes before a third orb is eaten. The subclass then said what
the base said. The two overridable methods were there for it alone, and nothing is written for a scenario that does
not exist.

**Rejected:**

- Legacy's open break: written when breaking cost nothing.
- The half-timeout ending for every tier orb eaten: unreachable once a break ends the episode.
- Keeping the template for a later variant: a variant adds the hook it needs when it exists.

**Revisit if:** a scenario in the family needs an ending of its own.

## Delay's values: a free silence, and a break at half the timeout (2026-11-06)

**Decision:** Delay's step penalty is 0, its `chain_break_penalty` is -0.5 and its timeout is -1.0. It is fully
observable: its registry entry allows `vector_markovian`, and the composite and pixel perceptions join when they are
rebuilt. Its ladder is spatial's (base 1.0, growth 1.5), on tier 3 and 5x5.

**Why:** If waiting costs something and ending is cheap, an agent learns to end, so the silence is free. With a
break ending the episode, the break penalty is most of what teaches the chain's order at a long delay: the completed
chain's pay comes hundreds of steps after the first orb and its credit is faint, while the break's cost comes at
once. At -0.1 the droid learned at once at every delay but stayed careless at long ones; at -0.5 it first learned to
stop eating wrong orbs, then passed -0.1 at 260 and 390 and ended level at 130 (PPO, seed 3, 50 spare steps). The
silence is what delay hides; a window on top would mix spatial's memory test into delay's.

**Rejected:**

- -0.1: a break that costs almost nothing teaches the order only through the pay it misses, which a long delay
  hides.
- Spatial's step penalty: the silence would cost steps.
- Any perception that does not see the whole grid: a window would test spatial too.

**Revisit if:** -0.5 does worse than -0.1 at 30 or 70, which have not run at -0.5; or the frozen orbs on the
field change what a wrong move costs.

## Delay's episode length follows from the delay and the grid (2026-11-06)

**Decision:** Delay's `max_steps` is set by its config model, never by a config, and a config that states it is
rejected. An episode gets one silence after each tier orb but the last, plus a fixed number of spare steps: `(max_tier -
1) * delay + 5 * (rows + cols)`, which is 50 spare steps on 5x5. The spare steps follow the grid and not the delay. The
5 is a constant on the model. Delay's tunables are the delay, the negative orb's switch and the perception; the grid is
not one.

**Why:** The thesis paired each delay with a `max_steps` by hand, and the spare steps came out as 90, 60, 40, 30 and
70 for delays of 30, 70, 130, 260 and 390. A comparison across delays then also compared clocks: at the same seed, a
delay of 390 learned faster than 260, which had the least spare time of the five. A longer delay needs no more time,
since the droid sees a frozen orb thaw and waiting is free; what grows with the delay is the credit the first eat has
to carry, which is what delay measures. Spare steps that grew with the delay would give a longer delay more room for
mistakes. The walking a silence cannot cover, and how far a wrong turn takes the droid, grow with the grid's side, so
the spare steps follow `rows + cols` rather than the area. A rule in the model cannot drift from the delay as a
hand-set number can.

**Rejected:**

- The thesis pairs: they mix the clock into the delay's axis.
- `max_steps` by hand from a formula: each run depends on the arithmetic.
- Spare steps as a config value: a measurement constant a run could change.
- Spare steps that grow with the delay: a longer delay would get more room for mistakes.
- Spare steps that follow the grid's area: generous where nothing needs it, four times 5x5's on 10x10.
- The model overwriting a stated `max_steps`: a value written in a config would be silently ignored.

**Revisit if:** delay learns worse at 50 spare steps than at 70, or a grid size other than 5x5 shows the spare
steps too tight or too loose.

## Delay's frozen orbs stay on the field and block their cell (2026-11-06, revised 2026-11-11)

**Built by:** phase 7a½, after delay is locked, between the observation pass's alignment (7a) and its new perceptions
(7b).

**Decision:** A frozen tier orb stays on the field in its own cell, drawn with a frozen skin, and thaws there when
the delay is over. It blocks its cell: the droid cannot move onto it. Blocking is a property of its own, which an
orb can have without being frozen; freezing an orb turns it on, and it turns off when the delay is over. A move into
a blocking orb is stopped as a move into the grid's edge is, but blocking costs nothing of itself: the grid's edge
always costs the boundary penalty, and a blocking orb costs nothing. A move into a frozen orb costs nothing either:
the droid stays where it is, the orb stays frozen, and the step penalty is all it pays. Since a frozen orb never
leaves its cell, the rule that a released orb waits for its cell to clear goes.

The observation marks a frozen orb as frozen, and carries no countdown.

It comes in as delay's next version, so its spread is run again.

**Why:** What the grid shows has to make sense to the agent and to a person watching (see the questions in
`docs/vision.md`). An orb that vanishes and comes back in the same cell does not look frozen to a viewer, and to the
agent it looks the same as an eaten orb or one that was never there. On the field, frozen is a state of its own,
told apart from both. The end of the gap stays hidden: the agent sees what it is waiting for and where, not when it
thaws.

A move into a frozen orb has no price of its own because the world already charges one: the move goes nowhere, so
it is a wasted step. In delay the agent has to wait for the thaw, and bumping into a frozen orb is waiting in place,
which costs what waiting anywhere else costs. A price on it would teach the agent to wait somewhere else, which is
not what delay measures, and would add a reward value to tune and to check against legacy, where a frozen orb was an
empty cell. The mark and the skin already say "not yet"; a price would say nothing more. What a bump costs comes from
what the orb is, and a frozen orb does not hurt. An orb that does, such as a fire orb, is a different orb.

Without the mark, a frozen orb and a thawed one in the same cell look the same, and the same move into it is blocked
or pays the orb. A stateless agent cannot tell the two apart, and an agent with memory has to count the steps, which
makes delay a memory test. With the mark, the agent knows that the orbs are frozen and not when they thaw, as it did
when the field emptied, and also where they are. A countdown would show the end of the gap, which is what delay
hides.

What delay needs is built as general capabilities of the core, none of them about delay, so each is there for the
next mechanic:

- **An orb state for "on the field, but not edible".** Frozen is the first. An effect orb that freezes others, or a
  hazard that disables an orb for a while, uses the same state.
- **A cell the droid cannot enter.** The world's first obstacle. Because blocking is its own property, an orb can
  block without being frozen, which is the first brick of a layout such as a wall around a goal orb (see
  `docs/vision.md`).
- **A move into something solid.** The droid's move already stops at the grid's edge; a blocking orb extends the
  stop into the grid, without the edge's charge.
- **An orb's state in the observation.** Once a perception marks one state, a later state arrives on the same
  channel.

An empty cell cost nothing because it borrowed "absent" to mean "frozen". That overload carries over to nothing; the
wiring above is paid once, and the next mechanic gets it for free.

**Rejected:**

- Building it before delay is locked: the ablations ran on the field emptying, so the spread would be run again
  before anything was locked, and the observation mark belongs after 7a.
- The field emptying while the orbs are frozen (decided 2026-11-05 and reversed here). It does not read as frozen
  to a viewer, and an empty cell mixes a frozen orb with an eaten one. Its reason, that an agent with orbs in sight
  and no way to eat them would first have to learn that it cannot eat yet, is answered by a frozen orb being a state
  of its own that the agent can tell apart.
- A frozen orb the droid can walk over without eating it: the move stays simple, but a droid passing through a solid
  orb does not make sense to watch.
- A bite: a penalty of around 0.1 for a move into a frozen orb (decided 2026-11-06 and reversed on 2026-11-11). The
  blocked move is already a wasted step, the bite would teach the agent to wait away from the orb, and it adds a
  reward value to tune.
- Blocking costing the boundary penalty: a wall costs nothing, and the boundary penalty is the grid's edge's.
- A frozen orb that looks like any other, learned from bumping into it: the same view would sometimes be blocked
  and sometimes pay.
- A countdown in the observation: it shows the end of the gap.
- Blocking as part of the frozen state: a wall orb would then have to be frozen to block.

**Revisit if:** an agent cannot learn delay with the frozen orbs in sight.
