# Decisions: The GUI

What the GUI is, what its version 1 holds, and how its screens go. The config models it reads are in
`locking-and-forks.md`; the game's own HUD is in `runners-and-rendering.md`.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## SYNGrid has a GUI, built on the registry (2026-10-12, settled 2026-11-13)

**Built by:** phase 9, and phase 10 for the sandbox's screen.

**Decision:** SYNGrid has a GUI: the control panel the vision describes, where a scenario is picked and set before a
run. Its version 1:

- The first step is picking a scenario: one of the registered ones, or the sandbox.
- For a locked scenario only its axis can be edited, and its other values are shown read-only.
- The values are on one side and a preview of the scenario on the other. The preview follows the form, so a larger
  grid for spatial shows as a larger grid.
- In the sandbox every value meant to be tunable can be edited, and orbs are added from a list of every existing
  kind.
- Version 1 stops there. A user who finds a good setup in the sandbox turns it into a scenario by hand, in the code.

It is built with pygame_gui, which themes its widgets through a JSON file. The game's HUD stays in plain pygame (see
"The renderer takes plain values, and the HUD is the scenario's list").

It is built on what the scenario layer already has. `SCENARIOS` is the list of scenarios, and an entry's config class
is what a user may set, so a form is made from it with no special case for locked scenarios or the sandbox.
`ConfigManager.load_config` ends in `model_class(**raw)`, so a dict from a form takes the path a YAML takes. A
`Scenario` holds nothing that changes and `build_world()` returns a new world on each call, so the preview builds a
world from the form on every edit.

**Why:** The vision's two ways in, tinkering at the panel and picking a hard scenario to play, both start at a panel,
and the panel is where a person feels they are touching a machine. The lock and the tunables were shaped so that a
form could be drawn from them.

**Rejected:**

- Version 2's "create scenario" step in version 1: a scenario is code today, an entry in its family's catalogue and a
  member of `ScenarioName`, and nothing is shaped for a scenario that is data. It stays an idea (see
  `codebase-ideas/gui-version-2.md`).
- pygame_gui for the game's HUD: it is built for interactive widgets, and the HUD is part of the game's look.

## Forking a locked scenario is the same screen, unlocked (2026-10-29)

**Built by:** phase 9, the forks step.

**Decision:** Picking a locked scenario shows its axis as editable: for spatial, the grid size, the negative orb's
switch, and the perception as a dropdown of the perceptions with its visibility. The rest of the base config is greyed
out, or hidden. An unlock button makes every value editable, or visible; the perception stays a dropdown with the
same list, and the negative orb becomes its switch with its two values under it. Cancel throws the edits away and
writes nothing. Save asks for a name and writes the fork.

Tunable is then a property of the mode, not of the scenario: locked for runs that can be compared, unlocked for
exploring. What a fork is, and that it is saved outside `src/`, is in "A fork is a full config with a name and a
parent".

**Why:** It came from talking through where to put the limit on what is tunable, by voice on 2026-10-26, and the
owner picturing the screen on 2026-10-29. Until the GUI exists, a fork is a YAML written by hand.
