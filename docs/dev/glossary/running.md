# Running

- **Runner:** what trains or evaluates an agent on a scenario.
- **SB3 spec:** an `SB3Spec` in `RUNNER`: one SB3 algorithm as SYNGrid runs it, its class, device, extra
  hyperparameters and frame stack. Calling it makes the SB3 runner.
- **Run id:** the name of a run's outputs, made of the perception, the seed, the scenario tag, the runner's optional
  tag and the algorithm.
- **Human play:** a person plays a scenario by hand through `HumanRunner`. It trains nothing and records nothing,
  and shows a summary of the episode when it ends.
- **Run controls:** the keys an agent run with a human renderer reads between two steps: Space pauses, and Escape or
  the close button quits. Eval and training share them.
