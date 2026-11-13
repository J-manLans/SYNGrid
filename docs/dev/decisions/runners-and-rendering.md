# Decisions: Runners and rendering

The runners, the algorithms, human play, the renderer and the window.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## The algorithm in the runner config is an enum, and a user's runner is added to it (2026-11-06)

**Decision:** `alg` in the runner config is an `Algorithm` enum (`PPO`, `FSPPO`, `RPPO`), and the `RUNNER` table is
keyed by it, as `ScenarioName` and `SCENARIOS` are. A wrong name fails when the config loads. The field is
`Annotated[Algorithm, Field(strict=False)]`, since a strict enum field rejects YAML's plain string. The run id uses
the member's `.value`.

`BaseAgentRunner` is the extension point for other users' runners. A user adds their own runner by subclassing it,
adding a member to `Algorithm` and an entry to `RUNNER`, in their own copy of the source.

**Why:** A typo is caught at load, and the valid names are stated where they are used. SYNGrid is installed by
cloning it and installing it editable, so a user's extension is an edit to their copy, as a new scenario is.

**Rejected:**

- `alg: str` checked in `build_runner`: a typo found only after the scenario was built.
- The member in an f-string: a `str` enum formats differently from Python 3.11, and checkpoints are found by that
  id.
- A registration function with `alg` as a string checked against the registry: it serves a package that is
  imported without being edited, which SYNGrid is not.

**Revisit if:** SYNGrid is published as a library that users install and import without editing its source.

## An SB3 algorithm is a spec in the runner table (2026-11-06)

**Decision:** SYNGrid's SB3 algorithms share one runner, `SB3Runner`. Each is an `SB3Spec` in `RUNNER`: the SB3 class,
the device, what it adds to the shared hyperparameters, and an optional frame stack. Calling a spec with the scenario
and the runner config makes its runner, so `RUNNER` holds a spec for an SB3 algorithm and a class for a user's own
runner alike. The eval loop hands every model the recurrent state and the episode starts, which a policy that is not
recurrent ignores.

**Why:** The runner classes per algorithm were near copies that differed in those four values. SB3's `predict`
takes the state and the episode starts for every algorithm, so the eval strategies per algorithm solved nothing.

**Rejected:**

- A runner subclass per SB3 algorithm: a file of copies for four values.
- An execution strategy per algorithm: `predict` already takes the state for all of them.
- Merging `BaseAgentRunner` into the SB3 runner: it is the extension point for users' runners.
- A `RunnerBundle` holding the scenario and the runner config: two fields and no behaviour, in the signature of every
  user's runner.

## The policy's name comes from the algorithm (2026-10-13)

**Decision:** `resolve_policy(perception, algorithm)` returns the one name in the algorithm's `policy_aliases` that
starts with the perception's base policy, and raises if there is not exactly one. The SB3 runner calls it; a
spec says nothing about the policy. The table from a perception's leading word to a base policy stays:
`vector` to `Mlp`, `composite` to `MultiInput`, `grid` to `Cnn`.

**Why:** It removes the LSTM special case, and naming the algorithm names the policy family.

**Rejected:** a `use_lstm` flag (the next variant needs another); a suffix or variant per runner (it restates what the
algorithm class says); perceptions named after SB3's policies (SB3's wording would reach the config).

**Revisit if:** an algorithm offers two policies with one base, or a perception's name stops starting with its kind.

## Human play is not an agent runner (2026-10-13)

**Decision:** `HumanRunner` is a plain class with `play()`. With `human_control` set, `app.main` runs it and returns
before an agent runner is built. Human control overrides the runner config, with no check.

**Why:** Human or agent is a choice about the whole run, made once, in `main`.

**Rejected:** `HumanRunner` as a `BaseAgentRunner` with a `train()` that raises (it inherited what it could not use).

## The renderer takes plain values, and the HUD is the scenario's list (2026-10-21)

**Built by:** phase 8a, for the fog overlay, and phase 8b, for the stacked HUD and the theme file. The rest is built.

**Decision:** The renderer is handed plain values and never reads the world or a config.

- From the world, through the environment: the droid's position, the orbs on the field and `grid_dimensions`.
- From the scenario, through `scenario.hud`: everything else the HUD shows, as static elements (a label, a kind, a
  maximum for a bar, a reader). The environment calls the readers and hands the values over.
- A scenario says what to show and in what order, never a pixel. Placement belongs to the renderer and its theme
  file. The images are constants in the renderer, the frame rate comes from the env's metadata; there is no
  `RendererConf`. An agent run sets the frame rate with `common_conf.fps`, which paces a human-mode window and a
  recorded video's playback, and the env takes its own copy of the metadata for it. Human play caps at a fixed 60,
  since nothing animates between steps and a lower cap only delays the next move.
- The fog overlay dims cells farther from the droid than a view radius on `ObservationRules`, the same one the
  perception reads.
- The HUD is to be stacked from drawn pieces (a top cap, a tiling row piece, a bottom cap), drawn by element kind,
  so a new element adds a row and no number. The theme file holds anchors inside each piece.
- The game HUD stays in plain pygame; pygame_gui is for the config GUI.

**Why:** The renderer stays free of `core/` and config, the HUD's look is edited as data, and a new scenario adds
elements without a change in the renderer.

**Rejected:** a generic labelled-rows layout (the HUD is part of the look and feel); a placement per scenario
(rendering in the scenario layer); energy, score and steps as fixed furniture; pygame_gui for the game HUD (built for
interactive UI); slots painted into one picture (it caps the count).

## Runners own the window's input, the renderer draws, and the env closes the window (2026-10-25)

**Decision:** The renderer reads no input. It draws a frame, an episode summary or the pause screen, sets the title
and closes the window. Each runner reads its own keys.

- `HumanRunner` blocks on pygame's event queue, takes the arrow keys as actions and quits on Escape or the close
  button. When an episode ends it shows a summary of the HUD values (from the env's `get_hud_data()`), the metrics and
  the last reward; Space plays again, Escape quits, other keys are ignored.
- Agent runs with a human renderer share run controls in `BaseAgentRunner`: Space pauses, Escape or the close button
  quits. Eval calls them before each step; training through `RunControlsCallback`, added only when training renders
  to a window.
- Quitting returns, it does not raise: eval returns from its loop, and the callback returns False from `_on_step`,
  SB3's way to end `learn()`, after which training saves.
- `SYNGridEnv.close()` closes the window, and every runner closes its env in a `finally`.

**Why:** The renderer has one reason to change, how things look. Stopping and closing use what SB3 and Gymnasium
already offer, so the window closes the same way however a run ends.

**Rejected:** pause read in the renderer's update with a flag for the human runner (input back in the renderer); a
summary between eval episodes (the `VecEnv` resets first, and the CSV is eval's summary); `pygame.quit()` in each quit
path (a crash left pygame running); `sys.exit()` (it skipped SB3's `on_training_end` and the last flush); the human
runner reading `scenario.hud` itself (two places to read the HUD); polling for keys (a busy CPU core).

## A run that continues another gets its own id, and records its parent (2026-11-04)

**Built by:** phase 9, the checkpoint step.

**Decision:** A run that starts from another run's checkpoint, as a curriculum stage does, is a run of its own. It
gets its own id from its own scenario and runner config, and it records which checkpoint it came from.

**Why:** A curriculum moves to a new grid, and the id carries the grid, so a stage cannot share its parent's id.
Today a checkpoint is found by the current run's id, so continuing on a new grid means renaming the parent's files
by hand and typing its step count into `agent_steps`. A model that went through 5x5 and 6x6 first would also carry
the same name as one trained from scratch on 7x7, and only a tag remembered by hand told them apart.

**Rejected:** extending the parent's id and folder (the id would name the first stage's grid, not the one trained on).

## No config snapshot until the GUI builds one (2026-11-09)

**Decision:** SYNGrid saves no copy of a run's config. The `snapshot` block in `global_config.yaml`,
`SnapshotConf` and `ConfigManager.save_snapshot` are gone, and `app.dispatch` only trains or evaluates.

**Why:** The snapshot copied the run's YAML byte for byte, which for a locked scenario is only the tunables, and it
required an id that nothing read. The GUI's scenario builder is expected to need something different, and will build
it then. Until then, a base config edit is committed before training on it, so a run's values can be looked up in git
by its date.

**Rejected:** fixing the snapshot now, to save the merged full config or the tunables beside their base config's
name: work the GUI is expected to replace.

**Revisit if:** runs need their config saved before the GUI exists.
