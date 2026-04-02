# Gymnasium Terminology

Core concepts and vocabulary for understanding the Gymnasium framework.

## Environment
The simulation/game world. It manages state, accepts actions, and returns observations/rewards. In Gymnasium, every environment inherits from `gym.Env`.

## Agent
The learner/player that interacts with the environment by taking actions and receiving observations/rewards.

## Episode
One complete run/playthrough from `reset()` until `terminated=True` or `truncated=True`. Agents train over many episodes.

## Step
One iteration within an episode. Agent chooses an action, environment executes it and returns (observation, reward, terminated, truncated, info).

## reset()
Gymnasium method that initializes/resets the environment to start state and returns the initial observation. Called at the beginning of each episode.

## step(action)
Core Gymnasium method. Executes one action in the environment and returns: `(observation, reward, terminated, truncated, info)`.

## Observation
The data/information the agent receives from the environment at each step. Could be numbers, images, arrays, etc. What the agent uses to decide its next action.

## Action
The choice/command the agent sends to the environment. Could be discrete (0, 1, 2...) or continuous (float values).

## Observation Space
Gymnasium `Space` object defining what observations are possible. Example: `Box(0, 255, (84, 84, 3))` for 84x84 RGB images.

## Action Space
Gymnasium `Space` object defining what actions are possible. Example: `Discrete(4)` for 4 discrete actions (up, down, left, right).

## Reward
Numerical feedback signal from the environment after each step. Positive = good, negative = bad. The agent's goal is to maximize cumulative reward.

## Terminated
Boolean flag indicating the episode ended due to reaching a terminal state (natural end, goal reached, failure).

## Truncated
Boolean flag indicating the episode ended due to reaching a time limit, not a natural terminal state. Important for proper RL training.

## Info
Dictionary returned by `step()` containing extra metadata/debugging info that doesn't affect learning (e.g., performance metrics).

## Render Mode
How the environment displays itself. `"human"` = live visualization, `"rgb_array"` = returns pixel array, `None` = no rendering.

## render()
Gymnasium method that visualizes/displays the environment state based on the render mode.

## seed(seed_value)
Sets the random seed for reproducibility. Ensures same sequence of random events across runs.

## close()
Cleans up the environment (release orbs, close windows). Good practice to call at the end.

## Space
Gymnasium abstraction for defining valid observations/actions. Common types: `Discrete`, `Box`, `MultiBinary`, `MultiDiscrete`.

## Discrete Space
Action/observation space with integer choices (0, 1, 2, ..., n-1). Example: 4-directional movement.

## Box Space
Continuous action/observation space within bounds. Example: `Box(low=-1.0, high=1.0, shape=(2,))` for 2D continuous control.

## Wrapper
Gymnasium pattern to modify environment behavior (add preprocessing, logging, reward shaping). Wraps an environment and modifies its inputs/outputs.

## VectorEnv
Gymnasium abstraction for running multiple environment instances in parallel for efficient batch training.

## Gymnasium Registry
Central catalog of all registered environments. New environments are registered so users can instantiate them with `gymnasium.make("EnvironmentName-v0")`.
