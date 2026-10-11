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

**Idea:** Once a scenario is declared done, its values move out of the YAML and are fixed in its builder, next to the
scoring mode the builder already chooses. The YAML and the config models keep only what sits on the scenario's
difficulty axis (grid size for spatial, chain length and grid for scaling, the delay for delay) and the perception.
Penalties, rewards, `max_steps` and `max_energy` are no longer fields of that scenario's config, so a YAML cannot set
them. The continuous sandbox scenario stays fully tunable, for every parameter that matters and
can't break something.

Varying a locked scenario means changing its axis value, not registering a new scenario per variation, so there is
no `spatial_6x6` beside `spatial_5x5`.

**Where it came from:** Reshaping the config models on 2026-10-10. Dense got defaults on its own classes so its YAML
could leave values out. Those dense classes were removed on 2026-10-11, since every scenario now sets its own
penalties.

**Why it might matter:** The scenario name is a run's identity. While every value is in the YAML, two runs with the
same name can describe different worlds, which is the reason scoring mode is already kept out of config. A short YAML
also shows at a glance what is meant to be varied.

Decided by the owner on 2026-10-11: the fixed values go in the builder, not in the config models and not in the
YAML. A fixed value is then no longer a field, so it cannot be overridden and a forgotten YAML line cannot load a
value nobody chose. The config models describe only what can be tuned.

Turned down: fixed values as defaults on the scenario's config models. A default can still be overridden, which
brings back two runs with the same name describing different worlds.

Not yet. A scenario is locked after it has been rebuilt and has reproduced its legacy results, because until then
the values aren't settled and comparing with legacy is easier with every value in view. Until then every scenario
states all of its values in its YAML.

Still to work out when it is done: the shared config models have fields a locked scenario no longer sets
(`chain_break_penalty` on `TierOrbConf`, `timeout_penalty` on `GoalConf`, `max_steps`), while the sandbox scenario
still needs them. So a locked scenario needs narrower models than the sandbox, or the builder needs another way to
be handed those values.
