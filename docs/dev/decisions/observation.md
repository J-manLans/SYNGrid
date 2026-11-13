# Decisions: Observation

What the agent sees and how it is laid out: what earns a place, the perception formats, the orb features the scenario lists, and how an observation leaves the environment.

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
  a tunable" (`locking-and-forks.md`) is built: the radius is the scenario's, the format is the run's.
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
