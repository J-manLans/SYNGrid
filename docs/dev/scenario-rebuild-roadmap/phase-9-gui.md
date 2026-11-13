# Phase 9: The GUI

**Produces:** the GUI's version 1: a scenario picked from the registry, its values in a form beside a preview that
follows it, forks of a locked scenario, and a checkpoint picked from a list.

What it is and how its screens go is decided in `decisions/gui.md`. It comes after the renderer's rework (8a and 8b),
since the preview draws through the renderer, and before the sandbox (phase 10), which adds its own screen. The
screens are laid out with the sandbox's in mind: every value editable, and orbs added from a list of every kind.

**Settle first** (see `open-questions.md`, phase 9 and "Locking a scenario"): whether the GUI starts a run or
writes the config a run reads; how a run points at the checkpoint it loads; where the user folder for forks is;
whether a locked axis has a range; and where a rule that spans fields is checked.

1. **Picking a scenario:** one of `SCENARIOS`, with what it tests, as its catalogue's comment says.
2. **The form:** a locked scenario's tunables editable, from its tunables model, and the rest of its base config
   read-only, from `base_conf()`. A field's explanation is shown beside it; today the explanations are in class
   docstrings, and the form wants them per field.
3. **Errors that name a field the user can see.** A check on the full model names a block today.
4. **The preview:** a world built from the form on each edit, drawn by the renderer. The renderer's settings have no
   place in the config models yet, and the preview needs them.
5. **Forks:** the unlock button, saving a fork under a name of its own, and cancel, as decided in "Forking a locked
   scenario is the same screen, unlocked".
6. **Picking a checkpoint** from a list, for eval and for a run that continues another, with the checkpoints that do
   not fit greyed out. What has to fit is in `open-questions.md`, under "How a run points at the checkpoint it loads".

**Done when:** every locked scenario can be picked, tuned within its axis, previewed and forked in the GUI, and an
eval picks its checkpoint from the list.
