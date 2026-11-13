# Phase 10: The sandbox

**Produces:** the continuous family and the sandbox scenario, with its screen in the GUI.

The sandbox is the scenario where every value can be set and orbs of every kind can be added. Its field is fixed, as
decided in "The sandbox is a fixed field, its tier orbs whole chains". It comes after the GUI (phase 9), so its
config is shaped by the form it is shown in.

**Settle first** (see `open-questions.md`, phase 10): whether a broken chain ends a continuous episode; whether a
`ScoringMode` enum routes the sandbox's scoring; and, from phase 7, whether chain progress is observed where tier orbs
return.

1. **The continuous family** in `scenario/families/continuous/`, an empty package today: its config models, its entry
   type and its catalogue. Its termination truncates on the clock, since a scenario with no objective is
   continuous (see "Digestion pays for orbs, and termination pays for the clock").
2. **Step-wise run for the first time.** `StepWiseDigester` is written and has never run.
3. **The sandbox scenario:** a count for each orb kind, and tier orbs as a max tier and a number of whole chains.
   Every value is tunable, so it has no lock.
4. **Its screen in the GUI:** every value editable, and orbs added from a list of every kind.

**Done when:** a sandbox setup made in the GUI trains, and the agent learns.
