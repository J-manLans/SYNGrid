# A held chain that keeps the droid alive (2026-10-16)

**Returns when:** a stochastic goal scenario, with a continuous threshold tier chain beside the goal, is planned.

**Idea:** In threshold scoring a held chain is worth something not yet paid. A droid that runs out of energy while
holding one could have the chain cashed into energy and go on, so the chain doubles as a reserve.

If it is ever built, it happens in the world and not at termination: after the droid acts and eats, a droid at 0
asks its digestion to cash out, and termination then reads energy as usual. Seen as a revive at termination it needs
a check for steps left (`terminated and steps_left > 0`), which reads poorly and lets a payout undo an ending. Seen as
a cash-out before termination looks, no clock check is needed: if the clock ran out too, the episode ends on the clock
and nothing is owed.

**Where it came from:** Deciding on 2026-10-16 that the ending's payout goes to the score and not to energy, so a
droid that ran out of energy stays at 0 on its last frame. See "Energy and score are two numbers" in
`design-decisions.md`.

**Why it might matter:** Probably as a reason not to build it. The droid can already cash out without the rule, by
eating a tier orb out of order while it still has moves left: a threshold break pays `chain_break_penalty` plus what
the chain had gathered. Learning when to do that is the long-horizon trade-off the benchmark wants to test, and an
automatic cash-out at 0 would hand the agent the answer. That deliberate cash-out only matters where a break does not
end the episode, in a family where chains repeat. In a goal tier chain a break ends it, and running out of energy pays
the chain without the penalty, so breaking on purpose never helps there.

If it is built anyway, it is a new path for a reward into the droid and changes what threshold teaches, so it would be
a scenario of its own. It fits a continuous family better than a goal one: a cash-out would have to end the chain
without counting as a break or a completion, and a goal episode would then go on with a chain that cannot finish.

**Where it earns its place (2026-10-16):** a highly stochastic goal scenario, with a continuous threshold tier chain
running beside the goal and the cash-out at 0 turned on. Orb rewards are uncertain, so the droid has to take chances,
and one payout can take it to 0 on a single step. A deliberate break cannot save it then, because the droid cannot
know beforehand which orb is the bad one. The only defence is to already carry an undigested chain, so the cash-out is
no longer an answer handed to the agent but what makes insurance a strategy. The agent learns how much chain is worth
carrying against the step costs of building it, and that a large held chain makes risky orbs affordable.

The goal ends the episode, and a break or a completion of the side chain ends nothing, which is a termination of the
scenario's own. Open when it is built:

- Whether a cash-out pays the whole chain, with anything above `max_energy` lost or left undigested, or only what
  keeps the droid alive. Paying only what is needed makes the chain a battery and changes what carrying it is worth.
- What a cash-out counts as. It is not a break or a completion, so it likely needs a counter of its own.
- What "stochastic" means: a reward drawn per orb, or kinds whose values are hidden. That decides whether the agent
  can learn to read the risk at all.
