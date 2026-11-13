# The GUI's version 2: creating a scenario (2026-10-12)

**Returns when:** the GUI's version 1 (phase 9) and the sandbox (phase 10) are done.

**Idea:** Version 2 of the GUI has a "create scenario" step: choose continuous or goal, define the goal, and save the
result as a scenario. Version 1 stops before it: a user who finds a good setup in the sandbox turns it into a scenario
by hand, in the code. What version 1 is is decided, in "SYNGrid has a GUI, built on the registry".

**Where it came from:** The owner's picture of the GUI, talked through on 2026-10-12 while asking whether the
locked-scenario models would also serve a GUI.

**Why it might matter:** It is the only part of the GUI that goes against the current design. A scenario is an entry
in code, in its family's catalogue, and a member of `ScenarioName`, so a scenario made in a GUI would have to be data
that a general entry composes from. Nothing is shaped for that now. The blocks (termination as a protocol with a file
per strategy, `OrbBundle`) are what such a composer would choose from.

Turning a sandbox setup into a goal scenario is not a copy of its config: the sandbox is continuous and has its own
world model, so the goal and the termination are decided by hand.
