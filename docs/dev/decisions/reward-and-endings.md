# Decisions: Reward and endings

Energy and score, who pays what and when an episode ends, and the reward values the ablations settled.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Energy and score are two numbers, and every reward goes through the droid (2026-10-16)

**Decision:** The droid keeps both, and every reward the agent is paid goes through it.

- **Energy** is what the droid has left. It starts at `max_energy`, stays between 0 and `max_energy`, and the episode
  ends when it reaches 0, in every scenario. One config value is the start and the cap.
- **Each reward is clamped as it comes in.** An orb eaten in the tick whose step drains the droid to 0 pays into
  energy after the step's cost, whatever it pays: a completed chain, a threshold break's pending worth or any other
  payout. A positive payout leaves the droid alive after the step, so energy does not end the episode, and the
  droid walks on unless termination ends the episode for another reason, as a completed or broken chain does in the
  single tier chain family. A negative payout leaves energy at 0.
- **Score** is the episode's return, the sum of every reward the environment hands back. It starts at 0 in `reset`,
  has no bounds and no config, and equals what SB3's `Monitor` logs.

| Path | What it carries |
|---|---|
| `take_step` | the step penalty and the boundary penalty |
| `consume_orb` | what digestion pays for the eaten orb |
| `on_termination(terminal_reward)` | what digestion is owed (the engine's `undigested`) plus what termination says the ending costs; to the score only |

The environment's step: the droid acts, termination decides, and if `outcome.terminated` the environment adds
`droid.on_termination(outcome.terminal_reward)`. Termination decides on the state before anything is settled, so a
settled payout cannot change whether the episode ends. An orb eaten in the step is not settled: it is paid in the
tick, before termination decides. The observation, the HUD and the metrics are read after, so the last
frame shows the final score and the energy the episode ended with.

**Why:** With one door for every reward, energy and score cannot drift from what the agent was paid, and every
reader handed the world reaches both. The cost: a reward from somewhere new needs its own path, decided then. The
ending's payout leaves energy alone because the episode is over; with it in energy, a droid that ran out while holding
a threshold chain showed energy again on its last frame.

**Rejected:**

- One number that was both (the old `starting_score`): the two meanings want different bounds and starts.
- Only penalties moving energy: energy responds to both.
- No cap, with a `max_score` to bound the observation: a cap gives an exact bound and removed three config classes.
- The score kept by the environment: energy still missed what termination added, and the scenario's readers could not
  reach it.
- Termination adding the timeout and what is owed to the reward itself, or calling the droid in `evaluate`: the
  score missed them, and a rule that decides would also change the world.
- A general `receive(reward)` on the droid: a general door makes "every reward goes through the droid" too easy to
  satisfy. Settling and paying the ending are one event, so one call.
- `end_episode` as the name: it covers truncation, where nothing is settled.
- A held chain reviving a droid that runs out while steps remain: it lets a settled payout change whether the
  episode ends. An orb eaten in the draining step is not this: it is paid in the tick, like any eaten orb.
- Clamping energy once per tick, on the sum of the step's cost and the orb's payout, so an orb eaten in the draining
  step pays off the overdraft first: energy is the droid's own state and need not keep step with the score, which is
  charged in full. What is forgiven is at most one step's cost, each time the droid eats on the step that drains it
  (0.001 in spatial and the scaling scenarios, nothing in delay). It would also need the world to tell the droid
  where a tick ends.

**Revisit if:** a continuous scenario needs a droid to bank rewards beyond the cap, which the old uncapped number
allowed.

## A truncated episode settles nothing (2026-10-25)

**Decision:** `on_termination` runs only when an episode is terminated. On a truncation nothing is settled: what a
digester holds is not paid and no `terminal_reward` is taken in. Only a continuous scenario truncates.

**Why:** At a truncation the learner bootstraps from the last observation (SB3 adds `gamma * V(terminal_obs)`), and
that value stands for the rest of the episode, including what the droid holds. Paying it out as well counts it twice.
A termination is a real end; what the droid holds is paid then or never. The value counts what the droid holds only if
the agent sees or remembers it, which is the open question on chain progress.

**Rejected:** settling on either ending; settling on truncation for a learner that treats it as terminal (that learner
is wrong about Gymnasium's contract).

**Revisit if:** a continuous scenario has something that should be paid at the cut. A new algorithm does not reopen
it, but check that it bootstraps at truncation.

## Digestion pays for orbs, and termination pays for the clock (2026-10-15)

**Decision:**

| Moment | Paid by | Pays |
|---|---|---|
| Chain break | the tier digester | `chain_break_penalty`, plus what the chain had gathered in threshold |
| Completion | the tier digester | what the mode says the chain is worth |
| Any ending | the droid's `on_termination` | what its digesters say they are owed |
| Timeout | termination, through `on_termination` | `timeout_penalty`, on top of the row above |

Termination only adds to the step's reward, and returns `EpisodeOutcome.terminal_reward` and changes nothing. The
clock is charged only when it alone ended the episode: a chain completed or broken on the last step, or energy running
out on it, pays no `timeout_penalty`. What is owed is settled at every ending, so threshold and step-wise total the
same over an episode. `undigested` is abstract, so every mode has to state it.

There is no `completion_reward`. `timeout_penalty` is in `goal_conf`, since any goal with a deadline has it;
`chain_break_penalty` is in `tier_orb_conf`. Both are required, with no default.

A goal scenario never truncates: its deadline is part of the task, so running out of steps is a failed attempt and
sets `terminated`. A scenario with no objective truncates on the clock, which makes it continuous.
`EpisodeOutcome.truncated` defaults to false. One termination class serves all three modes.

**Why:** A digester does the same in a goal and a continuous scenario; only whether termination ends the episode
differs. What reaching the goal pays is in one place.

**Rejected:**

- A `completion_reward` in termination: completing is caused by eating an orb, what it pays follows the mode, and a
  scenario where completion does not end the episode has no termination to pay it.
- A rule per mode inside termination, or a termination per mode: a mode conditional in termination, or the mode
  chosen in two places that can disagree.
- Nothing settled when energy reaches 0 (legacy): a threshold chain would be paid at every ending but one.
- The timeout replacing the step's reward (legacy): the last step's penalties and orb were lost.
- Threshold paying no `timeout_penalty` (legacy): the clock running out is the same event in every mode.
- An outcome type without `truncated` for goal scenarios: the environment hands Gymnasium both flags every step.
- Optional penalties or defaults of 0: a forgotten `timeout_penalty` would train with none.

**Revisit if:** a goal is something no single digester sees, such as surviving or clearing the field; a
scenario-level completion reward earns its place there.

## The reward ladder is one formula on the shared tier orb block (2026-10-11)

**Decision:** A tier orb is worth `base_reward * tier ** growth_factor`, both fields of the shared `TierOrbConf`, not
rounded. Scaling dense has no config classes of its own. Spatial uses a base of 1 and a growth factor of 1.5, so its
three tiers are worth about 9.0 together, near legacy's fixed 10.

**Why:** One formula with no setting that is ignored. The saved thesis configs all used non-linear growth.

**Rejected:** a `linear_reward_growth` switch (legacy), which left the growth factor required and ignored; rounding
(legacy, one branch only); the ladder on dense's own block; a base of 2 with linear growth for spatial.

## Reward normalisation stays off (2026-10-11)

**Decision:** Training keeps `norm_reward=False` on `VecNormalize`. Observations are normalised; rewards are not.

**Why:** The rewards are designed to sit in a small range. With sparse rewards the running estimate is tiny until the
first success and shifts when it comes. The 81 saved thesis runs were trained without it, and the owner recalls it as
deliberate.

**Revisit if:** a continuous scenario's return grows from many chains, or a chain's total goes well beyond a few tens.
