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
