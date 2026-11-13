# A GUI on top of the registry (2026-10-12)

**Returns when:** work on a GUI starts, which no phase of the roadmap holds. Picking a checkpoint returns sooner,
when a run continues another, such as a curriculum stage.

**Idea:** The owner's picture of the GUI the project is heading for.

- The first step is picking a scenario: one of the registered ones, or the continuous sandbox.
- For a locked scenario only its axis can be edited. Its other values are shown read-only.
- The values are on one side and a preview of the scenario on the other. The preview follows the form, so a larger
  grid for spatial shows as a larger grid.
- In the sandbox every value that is meant to be tunable can be edited, and orbs are added from a list of every
  existing kind.
- Version 1 stops there. A user who finds a good setup in the sandbox turns it into a scenario by hand, in the code.
- Version 2 has a "create scenario" step: choose continuous or goal, define the goal, and save the result as a
  scenario.

**Where it came from:** Talking through whether the locked-scenario models would also serve a GUI, on 2026-10-12.
Nothing in the repo is written for a GUI; the only mention is a TODO in `config/config_manager.py`.

**Why it might matter:** It is a second consumer of the config models, and it asks a few things of them.

What already fits:

- `SCENARIOS` is the list of scenarios, and an entry's config class is what a user may set for that scenario. A form
  can be made from it without a special case for locked scenarios or the sandbox.
- `ConfigManager.load_config` ends in `model_class(**raw)`, so a dict from a form takes the same path as a dict from
  a YAML.
- A `Scenario` holds nothing that changes and `build_world()` returns a new world on each call, so a preview can
  build a world from the form on every edit.
- Each orb kind is a `*_orb_conf` block on a world model. A sandbox world model that declares every kind as optional
  is the list of existing orbs, and adding an orb is switching a block on. It allows one block per kind.

What it asks for:

- An error has to name a field the user can see. See the open points in the design decision.
- A range on a locked axis would serve the form, the validation and the preview at once.
- The explanations of the fields are in class docstrings. A form would want them per field.
- The renderer's settings have no place in the new models yet, and a preview needs them.

Version 2 is the only part that goes against the current design. A scenario is an entry in code, in its family's
catalogue, and a member of `ScenarioName`, and a scenario made in a GUI would have to be data that a general entry
composes from. Nothing is shaped for that now. The blocks (termination as a protocol with a file per strategy,
`OrbBundle`) are what such a composer would choose from.

Turning a sandbox setup into a goal scenario is not a copy of its config: the sandbox is continuous and has its own
world model, so the goal and the termination are decided by hand.

The owner wants to learn pygame_gui and would use it for this GUI. It is built for interactive widgets and themes
them through a JSON file, which is what a form of fields, dropdowns and buttons needs. The game HUD stays in plain
pygame, see "The renderer takes plain values, and the HUD is the scenario's list" in `design-decisions.md`.

**Picking a checkpoint (2026-11-04).** For eval and for continuing a run, such as a curriculum stage, the user picks
a checkpoint from a list. What it is named does not matter, only whether it fits the run being set up. Today a
checkpoint is found by rebuilding the run id from the config, so a continued stage on a new grid means renaming the
parent's files by hand. Found while talking it through:

- What has to fit is the observation's shape and the algorithm, with the policy it brings. Bounds that differ, as a
  new grid gives, already pass: the loaded `VecNormalize` keeps the space it was saved with and checks shapes only.
  Accepting that on purpose is a choice to write down then. The `VecNormalize` statistics saved beside the
  checkpoint come with it.
- A curriculum over the grid fits today. Over tiers in a continuous scenario it fits, since the slot count is
  `max_active_orbs`, if identities stay put when the pool grows: the world renumbers them by sorting, and a kind that
  sorts earlier would shift every number after it. Over tiers in a single chain it does not fit, since the slot count
  follows the orbs; it needs a fixed slot count, as legacy's `tiers` setting gave.
- A list can grey out the checkpoints that do not fit.

See "A run that continues another gets its own id, and records its parent" in `design-decisions.md`.

**Forking a locked scenario (2026-10-29).** What a fork is, and where it is saved, is decided (see "A fork is a full
config with a name and a parent" in `design-decisions.md`); this is how the screen could go. Picking spatial shows its
axis as editable: the grid size, the negative orb's switch, and the perception as a dropdown of the perceptions with
its visibility. The rest of the base config is grayed out, or hidden. An unlock button makes every value editable, or
visible; the perception stays a dropdown with the same list, and the negative orb becomes its switch with its two
values under it. Cancel throws the edits away and writes nothing. Save asks for a name and writes the fork. Tunable
then becomes a property of the mode and not of the scenario: locked for runs that can be compared, unlocked for
exploring. Until the GUI exists, a fork is a YAML written by hand. It came from talking through where to put the limit
on what is tunable, by voice on 2026-10-26, and the owner picturing the screen on 2026-10-29.
