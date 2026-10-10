# Ideas About the Codebase

Things noticed while rebuilding that aren't problems to solve now, just
worth revisiting later. Low bar for entry: half-formed is fine.

---

## Entry template

**Idea:**
**Where it came from:**
**Why it might matter:**

---

## A locked scenario keeps only its axis in the YAML (2026-10-10)

**Idea:** Once a scenario is declared done, its values move out of the YAML and are fixed in its config models. The
YAML keeps only what sits on the scenario's difficulty axis (grid size for spatial, chain length and grid for
scaling, the delay for delay) and the perception. Penalties, rewards, `max_steps` and `max_energy` are no longer
tunable for that scenario. The continuous sandbox scenario stays fully tunable, for every parameter that matters and
can't break something.

Varying a locked scenario means changing its axis value, not registering a new scenario per variation, so there is
no `spatial_6x6` beside `spatial_5x5`.

**Where it came from:** Reshaping the config models on 2026-10-10. Dense got defaults on its own classes so its YAML
could leave values out, which is the same mechanism a locked scenario would use for all of its fixed values.

**Why it might matter:** The scenario name is a run's identity. While every value is in the YAML, two runs with the
same name can describe different worlds, which is the reason scoring mode is already kept out of config. A short YAML
also shows at a glance what is meant to be varied.

Not yet, because the values aren't settled: the learning problem is open, and comparing with legacy is easier with
every value in view. To decide when it is done: whether a fixed value is a default that a YAML could still override,
or is removed as a field so it can't be. If overrides stay possible, they should mark the run id as non-standard.
