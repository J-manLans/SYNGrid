# Orbs that block without being frozen (2026-11-06)

**Returns when:** a scenario needs cells that block, or orbs placed on purpose, such as the vision's wall around
a goal orb.

**Idea:** An orb kind whose orbs block their cell for good: a wall. With blocking a property of its own, a wall is an
orb that blocks and is never eaten, and walking into it costs nothing. A layout of them can shape a scenario, such
as the vision's wall of deadly orbs around a goal orb.

**Where it came from:** The owner and the agent, while deciding that delay's frozen orbs block their cell. See
"Delay's frozen orbs stay on the field and block their cell" in `design-decisions.md`.

**Why it might matter:** The vision says a scenario's character may come from how its orbs are arranged, and that
nothing can describe a layout today. Blocking cells are the first piece of that. Placing walls on purpose, and not at
random, is the larger part and is not touched.
