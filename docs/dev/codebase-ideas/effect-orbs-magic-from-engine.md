# Effect orbs get their magic from the engine (2026-10-27)

**Returns when:** effect orbs are built. The roadmap lists them under "Not on this roadmap".

**Idea:** When effect orbs are built, the combining happens in the digestion engine and nowhere else. Digesters keep
not knowing about each other. The engine already sees all of them, so it is the one place that can let one kind's orb
change what another kind's orb pays. This is a lean, not a decision. Effect orbs are not built, and the roadmap lists
them under "Not on this roadmap".

**Where it came from:** The owner remembers deciding this in an earlier talk, of which no record was found. It was
talked through again on 2026-10-27. It answers the "Revisit if" of "Digesters do not talk to each other" in
`design-decisions.md`, and the roadmap's list of what effect orbs need from the engine.

**How it could work.** The engine's step becomes a short pipeline with a fixed order:

1. **Tick.** Once per step, the engine calls an optional `tick()` on every digester that has one. This is how a timed
   effect counts down or wears off.
2. **Route.** The consumed orb goes to the digester of its kind, as today, and that digester returns a reward.
3. **Modify.** The engine hands the reward to each digester that has the optional modifier capability, in a fixed
   order, and each returns the adjusted reward. A "double the next tier payout" orb is an effect digester that holds
   the doubling and applies it here. The tier digester never hears of it.
4. **Reach the world.** An effect that has to touch the field (clear it, freeze it, move an orb) is handed a narrow,
   one-way view of the world, to read from and to ask through. It does not hold the world.

The capabilities are small optional protocols next to `OrbDigester`, such as `Tickable` and `RewardModifier`. The
engine checks which a digester has when it is built. A digester with none of them is the digester of today, so
nothing already written changes. This is the list in the roadmap: a tick, a stage before routing, a way to change
what another kind's orb pays, a way to reach the world.

**Why it fits the rule.** A digester can still be read and tried alone. The only thing that knows about combinations
is the engine, and it knows them as stages in a fixed order, not as digesters calling each other. A step's reward can
be explained from one place by reading the pipeline top to bottom.

**Two things to settle first:**

- **Payouts that skip the pipeline.** `undigested()` is settled by the droid when the episode ends. If a modifier is
  meant to scale a held chain payout too, that payout has to pass through the modify stage as well. Today it does
  not.
- **Order matters.** Two modifiers do not commute: doubling and then subtracting is not subtracting and then
  doubling. The order has to be written down, by scenario or by a fixed rule, and not left to how the list was built.

Open when it is built: whether modify runs before routing (so an effect can change which digester gets the orb) or
only after (so it can only change the reward); where the order of modifiers is stated; and what the narrow view of
the world exposes.

**Field effects through a watcher (2026-11-05).** Step 4 above, an effect that touches the field, may fit a watcher
better than a view of the world handed through the engine: the engine lives inside the droid, which holds no world,
while a watcher is handed the world and the eaten orb's kind every step. A reward effect could also be paid by a watcher
the way termination pays its ending, through the droid and added to the step's reward, but a held payout
(`undigested()`) never reaches the step result, and two homes for reward effects would need one chosen. The owner's
direction: an orb kind's effects stay contained to the kind, with explicit channels where something has to communicate.
A kind could carry its own watcher in its bundle (see "A bundle that carries everything its orb kind brings",
`bundle-carries-its-kind.md`), so its field effect travels with it and the scenario never has to know. See "The world's
step returns a result, and a scenario's watcher reacts to it" in `design-decisions.md`.
