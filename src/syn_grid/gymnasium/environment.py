from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from syn_grid.core.droid.droid_action import DroidAction
from syn_grid.gymnasium.observation_space.observation_handler import (
    ObservationHandler,
)
from syn_grid.gymnasium.utils.episode_logging.keys import SYN_STATS_KEY
from syn_grid.rendering.pygame_renderer import PygameRenderer
from syn_grid.scenario.scenario import Scenario


class SYNGridEnv(gym.Env):
    """
    SYNGrid reinforcement learning environment.

    A discrete grid-world environment for benchmarking single-agent RL.
    """

    # ================= #
    #       Init        #
    # ================= #

    # Metadata required by Gym.
    # "human" for Pygame visualization.
    # render_fps caps the update rate of render(); each call corresponds to one logic step, not the
    # full game framerate. Simply put: render_fps controls the speed of the environment’s logic,
    # while a sub-loop in the renderer would handle smooth animation between steps.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}  # noqa: RUF012 - Required by Gymnasium API

    def __init__(
        self,
        scenario: Scenario,
        render_mode: str | None = None,
    ):
        # Set up bench environment;
        self.render_mode = render_mode
        self.world = scenario.build_world()
        self._hud = scenario.hud
        self._termination = scenario.termination
        self._metrics = scenario.metrics

        if self.render_mode in self.metadata["render_modes"]:
            self.renderer = PygameRenderer(
                self.world.grid_dimensions,
                self._hud,
                render_mode,
                self.metadata["render_fps"],
            )

        # Set up Gymnasium environment:

        # Gymnasium requires us to define action_space — which is the agent's possible
        # actions. Training code can call action_space.sample() to randomly select an action.
        self.action_space = spaces.Discrete(len(DroidAction))

        # Same goes with observation_space: this provides the agent with a structured view
        # of the world that it uses to decide its actions.
        self._observation_handler = ObservationHandler(
            scenario.observation, self.world.grid_dimensions, self.world.max_identity
        )
        self.observation_space = self._observation_handler.setup_obs_space()

    # ======================== #
    #    Gymnasium contract    #
    # ======================== #

    def reset(self, *, seed=None, options=None):
        # Gymnasium requires this call to control randomness and reproduce scenarios.
        super().reset(seed=seed)

        # Reset the environment.
        self.world.reset(self.np_random)
        self._observation_handler.reset()

        if self.render_mode == "human":
            self.render()

        self.obs = self._observation_handler.get_observation(self.world)

        # Return observation and info (not used)
        return self.obs, {}

    def step(self, action: int):
        info: dict[str, Any] = {}

        # Perform action and adjust variables affected by it
        reward = self.world.perform_droid_action(DroidAction(action))
        outcome = self._termination.evaluate(self.world)

        # A truncation is not settled: the learner bootstraps what the droid holds from the last
        # observation.
        if outcome.terminated:
            reward += self.world.droid.on_termination(outcome.ending_reward)

        # The metrics are read after the droid is settled, so they see how the episode ended.
        # TODO: still think I should only do this if csv output is on. Basically, I will most use
        # it when evaluating a trained agent, so storing csv data for each training run seems like
        # unneeded computation for a benchmark that aims at being lightweight.
        if outcome.terminated or outcome.truncated:
            info[SYN_STATS_KEY] = self._get_episode_metrics()

        if self.render_mode in self.metadata["render_modes"]:
            self.render()

        self.obs = self._observation_handler.get_observation(self.world)

        # Return observation, reward, terminated, truncated and info
        return (
            self.obs,
            reward,
            outcome.terminated,
            outcome.truncated,
            info,
        )

    def render(self) -> np.ndarray | None:
        orbs = self.world.orbs_on_field()

        frame = self.renderer.render(
            self.world.droid.position,
            [True] * len(orbs),
            [orb.position for orb in orbs],
            [orb.META for orb in orbs],
            self._get_hud_data(),
        )

        if self.render_mode == "rgb_array":
            return frame

    # ================== #
    #       Helpers      #
    # ================== #

    # === Gymnasium contract === #

    def _get_hud_data(self) -> dict[str, float]:
        return {
            hud_element.label: hud_element.read(self.world) for hud_element in self._hud
        }

    def _get_episode_metrics(self) -> dict[str, float]:
        return {metric.name: metric.read(self.world) for metric in self._metrics}
