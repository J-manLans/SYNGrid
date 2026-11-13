# The world

- **World:** the `GridWorld`: the grid, the droid, the orbs and the step clock.
- **Droid:** what the agent moves. It eats orbs and owns digestion.
- **Field:** the orbs that are on the grid.
- **Fixed field:** every orb is stated and present, as in a tier chain.
- **Sampled field:** orbs are drawn at random from a pool, and each kind's weight sets the mix. None is written.
- **Energy:** what the droid has left. It starts at `max_energy`, stays between 0 and `max_energy`, and the episode
  ends when it reaches 0.
- **Score:** how the episode went: the episode's return, the sum of every reward the environment has handed back,
  termination's included. The droid keeps it beside energy, since every reward goes through the droid. It starts at
  0 and has no bounds.
- **Settle:** the droid taking in what its digestion has left undigested when the episode is terminated, in `on_termination`.
- **Terminal reward:** what termination says the ending itself costs or pays, `EpisodeOutcome.terminal_reward`. The
  environment hands it to the droid's `on_termination`. In a goal scenario it is the timeout penalty or 0.
- **Reward:** what one step pays. A negative reward is a penalty.
- **Penalty:** a config value that must be 0 or negative. Every field whose name ends in `_penalty` is checked.
- **Step clock:** the count of steps in an episode, up to `max_steps`.
- **Timeout:** the step clock running out. It ends the episode and costs `timeout_penalty`.
- **Terminated, truncated:** Gymnasium's two ways for an episode to end. Terminated means the task is over, reached
  or failed. Truncated means the clock stopped an episode that had no task to fail. A goal scenario never truncates:
  its deadline is part of the task, so running out of steps sets `terminated`. A continuous scenario is where
  `truncated` is used. `EpisodeOutcome.truncated` defaults to false, so a goal strategy does not name it.
