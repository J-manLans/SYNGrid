# Decisions: Observation

What the perceptions show, and how an observation leaves the environment.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## The kept perceptions observe only what earned its place (2026-10-23)

**Decision:** `vector_markovian` and `vector_fog_of_war` observe the droid's position and one block per orb slot
(active flag, row, column, identity), and no global values: no steps left, score, chain tier or orb timer. The single
tier chain family sets `global_features=()`.

**Why:** The owner ablated an observation like `vector_markovian_easy` one value at a time and removed every value
without which the agent did as well or better. A value the agent does not need costs input and gives the policy
something to fit.

**Rejected:** the global values in front with an optional timer per orb (the ablation's starting point).

**Revisit if:** a scenario has to be solved from a value these leave out, such as threshold's gathered sum. When the
legacy perceptions are rewritten, this is the reason not to add their global values to the two kept ones.

## An orb's timer is not observed (2026-10-23)

**Decision:** The agent is not shown how long an orb stays or how long until it returns. It sees an orb when it is on
the field. A rewritten perception gets no timer by default.

**Why:** Delay tests whether an agent can bridge a gap it cannot see the end of. A countdown would let it read the gap
off the input, and the memory question would shrink to nothing; withholding it is the measurement. Spatial works the
same way: the agent remembers what it is not shown. That a stateless agent cannot see an orb vanish is the
stateless-versus-memory difference the benchmark measures. Chain progress is not a timer but the agent's own history.

**Rejected:** showing the time left on an orb, and for a missing orb the time until it returns.

## The HUD and the observation share readers and keep their own lists (2026-10-23)

**Decision:** A number the HUD shows and the observation carries is read by the same module-level reader, and each
consumer lists it in its own tuple: a `HudBar` or `HudStat` in `hud`, a `GlobalFeature` in
`ObservationRules.global_features`. Readers every scenario can use are in `scenario/blocks/readers.py`; a family's own
sit in its `builders.py`.

**Why:** The two diverge on purpose: the HUD shows energy, score, steps left and the chain, and the observation none.
The HUD needs a label and a bound only for a bar; the observation always needs a bound and a fixed order, and score
has no upper bound for a `Box`.

**Rejected:** one list with a flag per consumer (the flags grow); a shared base type (it couples the two files for two
saved arguments; worth it if a third consumer appears); the readers in `hud.py` (the observation would seem to depend
on the HUD).

## The environment hands out a copy of each observation (2026-11-07)

**Decision:** A perception fills one buffer and returns it every step, and `ObservationHandler.get_observation`
returns a copy of it. Every observation that leaves the environment is an array of its own, so one a caller keeps does
not change with the next step.

**Why:** Gymnasium asks for new observation data on every call. From 1.4.0 its passive checker warns when two
observations share an object, and `check_env`'s reset-determinism check fails on a shared buffer, because a later
reset overwrites an observation it compares. The copy is one small allocation a step, and the perceptions keep their
buffers.

**Rejected:**

- A new array in each perception: every perception would change, and the rewritten ones would each have to remember.
- The copy in the environment's `reset` and `step`: two places for what the handler does in one.
- Keeping the shared buffer: `check_env` fails, and anything that keeps two observations must copy them itself.

**Revisit if:** a perception returns something `.copy()` does not copy through, such as a dictionary of arrays.

## `vector_markovian_easy` is not rewritten (2026-11-11)

**Decision:** Of legacy's perceptions, `vector_markovian_easy` is not brought back in phase 7b. It goes with legacy.

**Why:** It is `vector_markovian` with global values in front (steps left, score, chained tiers) and an optional timer
per orb. The ablation behind "The kept perceptions observe only what earned its place" removed those values one at a
time, and the agent did as well or better without them, so what sets it apart is noise. It is also not a perception
of its own once the scenario lists what the agent sees: its global values are `global_features`, and its timer would
be an orb feature, so it is the vector format with a longer list (see "A perception is a format, and the scenario
says what the agent knows").

**Rejected:** rewriting it as a third vector perception.

**Revisit if:** a scenario has to be solved from a global value. That value then goes in the scenario's
`ObservationRules.global_features`, read by a kept perception, not in a perception of its own.

## The composite perceptions are rewritten, as one (2026-11-11)

**Decision:** `composite_markovian` and `composite_fully_pomdp` come back in phase 7b as one composite format. They
differ only in global values and a timer, which the scenario's lists now carry.

**Why:** They carry the same information as the vectors, but split into named parts: the droid, the orbs and the
global values each have a space of their own. That structure is worth keeping for a network that reads each part on
its own terms. Until there is a custom feature extractor, SB3 flattens the `Dict` back into one vector, so the
structure pays off only once there is one.

**Rejected:** dropping them as copies of the vectors.

**Revisit if:** no custom extractor is planned, so the composites stay flattened copies of the vectors.

## An orb's identity is the radix scalar, and tier order is meant to be seen (2026-11-12)

**Decision:** The identity the agent sees is one number per orb, built from the mixed-radix formula on category,
kind and tier, with tier in the top place, so the identity grows with tier. The agent sees it as its rank among the
scenario's orbs, the dense identity (see "An orb's identity is looked up in a table the scenario builds"), which keeps
the order. That the agent can read the chain's order from it is intended, not a leak. It is one orb feature in the
slot.

**Why:** The thesis tried three designs. Identity began as three fields, category, type and tier. After the move to
distance sorting, one-hot gave cleaner and faster convergence, but every tier was an identity of its own, so the
width grew with tier depth and performance fell from tier 3 to tier 10. The radix scalar kept one-hot's single clean
value at a fixed width, and the thesis named its cost: it is a magnitude, not a pure identity, and its limitations
section notes it may hand the stateless agent the order. The delay scenario even relied on the agent reading the
order from the identity.

The vision settles that cost. Tier depth makes a task longer for an agent but not harder for a person, who sees the
sequence at once, and what the agent sees should match what a person watching sees. A readable rank is what a person
gets, so it is not a leak.

The remaining cost is that the magnitude means rank for tier orbs and nothing between kinds. With effect orbs the
catalogue grows to around fifteen kinds, but a scenario will mostly hold about three, and a small network separates
three values easily. The owner's call is that the radix handles that.

Background, from the talk on 2026-11-11, so the reasoning can be found again:

- **MiniGrid** gives the agent a 7x7 view in which each cell holds three integers: object type, colour and state
  (open, closed, locked). It is a grid of facts, not pixels; pixels come from an optional wrapper. Most baselines run
  a small CNN over it and treat the integers as plain numbers, magnitude and all. It is close to the thesis's early
  three fields: each field means one thing, none carries a meant order.
- **MiniHack**, on the NetHack Learning Environment, gives each cell a glyph, an integer up to 5991 for every
  distinct entity, and baselines run it through an embedding layer, a learned vector per id, before a CNN. An
  embedding reads no magnitude, so nothing leaks and the width does not grow. Its own authors note the weak spot:
  NetHack's glyph ids are not well suited to learning, as more than half of them are of a type that never shows up
  in play. An embedding also learns every id on its own, so neighbouring tiers share nothing until training finds it,
  and ids the agent seldom sees, such as high tiers, learn poorly. It needs a custom network.
- **SYNGrid's radix** is a small id space designed for its job, with the one ordered property placed on purpose. It
  pays with a magnitude that means nothing between kinds, where MiniHack pays with a large, mostly empty id space and
  an embedding.

**Rejected:**

- Tier as a number of its own beside one flag per kind in the scenario: honest about which value is ordered, but
  wider, and its advantage at three kinds is small. It is the first thing to try if this is reopened.
- An embedding of the id in every observation: it needs a custom extractor for every run. It is the composite's way
  out instead (see the 7b note in the roadmap).
- MiniGrid's separate integers per field: the thesis moved away from composing identity from several fields.
- One-hot over identities: its width grows with tier depth.
- Hiding the order with shuffled ids: it would make the agent learn what a person sees at once.

**Revisit if:** a scenario holds clearly more than about three kinds, or an effect-orb scenario learns slower than
expected. The cheap check is an ablation on that scenario and its seeds: the radix against tier plus kind flags.

## An orb's identity is looked up in a table the scenario builds (2026-11-12)

**Decision:** The dense identity stays: the agent sees the orb's rank among the radix identities of the scenario's
orbs, 1 to N, in radix order. It no longer lives in the orb's meta. The meta holds kind and tier, its radix is a pure
function of them, and it is frozen. The scenario builds a table from radix identity to dense identity once, from the
kinds and tiers it holds, and the identity orb feature reads the orb's dense id from it. The world no longer
remaps.

**Why:** The dense id is a fact about the set of orbs, not about one orb, so a meta that knows only itself cannot
compute it. That is why the world remapped after every orb was built and wrote the result back into each meta, which
kept `IDENTITY` from being `Final`. What the agent sees is the observation's business; the renderer and digestion
never read the dense id. The table numbers the same orbs the same way the world does today, so the observation
should not change.

Where the remap came from is undecided. Two accounts:

- **The thesis's lessons notes** record it as a lesson of its own: the mixed-radix identity flatlined the LSTM under VecNormalize warm-up, and the dense
  sequential remap fixed it. They call it a narrower, debugging-flavoured lesson, kept apart from the dimensionality
  story of one-hot against scalar.
- **The owner recalls** it coming in with the one-hot experiments, where each identity needed a column of its own and
  dense ids number those columns, and staying when one-hot was dropped because an ablation showed it learning a
  little faster. The owner tried one-hot, and it did not scale, since tiers 1, 2 and 3 were each a kind of their own;
  how the remap fits with that is not clear.

`git log -S` does not settle it. The remap first appears in `ab70676` (2026-05-13), the same commit that brought in
the radix identity, which replaced three fields (category, type and tier). No one-hot encoding was ever committed
under `src/`, so the one-hot experiments may not have been committed, and nothing in the code mentions VecNormalize
or an LSTM.

The remap stays, for the even spacing it gives the scalar. Whether the plain radix does as well is for the experiment
named under "Revisit if".

**Rejected:**

- Dropping the remap (decided on 2026-11-11 and reversed here, once the radix scalar was kept): the only evidence
  there is says it helped.
- The meta handed its dense id when it is built: possible, since the scenario knows its kinds before any orb exists,
  but it keeps an observation concern in a core object.
- The world remapping and writing into the meta, as before: the meta cannot be frozen.

**Revisit if:** spatial, trained in 7a with the radix against the remapped radix on the same seeds, shows the remap
earns nothing. The same orb gets different dense ids in different scenarios, so revisit it too if one model has to
move between scenarios.

## A perception is a format, and the scenario says what the agent knows (2026-11-11)

**Decision:** There is one perception class per format: vector, composite, grid and pixel. What the agent gets to
know is the scenario's, in its `ObservationRules`: the view radius, the global features, the orb features and the
slot count. The base perception picks the orbs within the view radius, sorts them by distance and reads the feature
lists; a format only lays out the result.

- **Visibility is one number.** A view radius of none shows every orb; a radius of 1, 2 and up shows the orbs within
  it. Today's `vector_markovian` is the vector format with no radius, and `vector_fog_of_war` is the vector format
  with a radius of 1. The two are spoken of as the full view and the limited view, not as Markovian and POMDP: even
  the full view is not Markovian once a delay hides time, and legacy's `fully_pomdp` meant "no global values".
- **The `Perception` enum names formats.** This is how "A locked scenario fixes the visibility, and the encoding is
  a tunable" (`scenarios-and-config.md`) is built: the radius is the scenario's, the format is the run's.
- **Every format but pixel reads the same lists.** The vector and the composite write orb features as the columns of
  a slot; the grid writes them as channels of a cell, where the cell is the position and there are no slots and no
  sort. Pixel reads no features: an orb's kind, tier and state show in its sprite, so pixel follows a new mechanic
  through the renderer. The view radius applies to all four.

**Why:** Legacy's seven perceptions were each a visibility paired with an encoding, plus a few extras, so they grew
as a product and every encoding repeated the same filter. With the extras in the scenario's lists, a new mechanic is
one reader and no new class, and the difficulty of a scenario can be read in one place. A scenario carries only the
columns it needs, as "The kept perceptions observe only what earned its place" asks.

**Rejected:**

- A class for each pair of visibility and encoding: the classes grow as a product, and the filter is repeated.
- Visibility as subclasses, or as a selection class of its own: the radius is a number, and "none" covers the full
  view.
- Fog of war kept as a named perception a scenario points at: it is the vector format with a radius of 1.

**Revisit if:** one trained model has to move between scenarios, as in curriculum training or transfer. Scenarios
listing their own columns give them observations of different shapes, as the slot count already does.

## An orb slot's columns are orb features the scenario lists (2026-11-11)

**Decision:** Every orb slot holds the orb's active flag and its position. Everything else about the orb is an orb
feature: a reader and a bound, as a `GlobalFeature` is, listed by the scenario in its `ObservationRules`. The
identity is one orb feature, read from the scenario's table (see "An orb's identity is looked up in a table the
scenario builds").

**Why:** It is the shape `global_features` already has, so there is nothing new to learn. A new mechanic adds one
reader, and only the scenarios that have the mechanic list it, so the observations of the others do not change.
Active and position are fixed because every orb has them, and empty slots are told apart by the active flag.

**Rejected:** one general slot that carries every orb feature in every scenario: a column that is always zero where
a mechanic is missing, and every observation changing whenever a mechanic is added.

**Revisit if:** the same as for "A perception is a format, and the scenario says what the agent knows".
