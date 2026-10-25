import sys
from typing import Any, Final, cast

import pygame

from syn_grid.core.droid.droid_action import DroidAction
from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.gymnasium.utils.episode_logging.keys import SYN_STATS_KEY
from syn_grid.scenario.scenario import Scenario


class HumanRunner:
    """
    Play a scenario by hand through a real SYNGridEnv.

    For inspecting a configuration, not for learning. It trains nothing,
    writes no checkpoints, and records no metrics. Its job is to read
    human input, pass actions to the environment, and stop when the
    episode ends. When it ends, it shows a summary of the episode over the
    last frame until a key is pressed: the scenario's HUD values, its
    metrics and the last step's reward.
    """

    # ================= #
    #       Init        #
    # ================= #

    _KEY_ACTIONS: Final[dict[int, DroidAction]] = {
        pygame.K_LEFT: DroidAction(0),
        pygame.K_DOWN: DroidAction(1),
        pygame.K_RIGHT: DroidAction(2),
        pygame.K_UP: DroidAction(3),
    }

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self._env = cast(SYNGridEnv, make(scenario, "human").unwrapped)

    # ================= #
    #        API        #
    # ================= #

    def play(self) -> None:
        """Play one episode, blocking until it ends or the player quits."""

        try:
            self._env.reset()

            while True:
                action = self._get_user_action()
                _, reward, terminated, truncated, info = self._env.step(action.value)

                if terminated or truncated:
                    self._show_summary(reward, info)
                    break
        finally:
            self._env.close()

    # ================= #
    #      Helpers      #
    # ================= #

    def _show_summary(self, reward: float, info: dict[str, Any]) -> None:
        """Show the episode's summary and wait for a key."""

        world = self._env.world
        summary = {
            **{element.label: element.read(world) for element in self._scenario.hud},
            **info[SYN_STATS_KEY],
            "last reward": reward,
        }

        self._env.renderer.draw_summary(summary)
        self._next_key()

    def _get_user_action(self) -> DroidAction:
        """Blocks until an arrow key is pressed and returns its action."""

        while True:
            action = self._KEY_ACTIONS.get(self._next_key())

            if action is not None:
                return action

    def _next_key(self) -> int:
        """
        Blocks until a key is pressed and returns it.

        Escape or closing the window quits. Other input is ignored.
        """

        while True:
            event = pygame.event.wait()

            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                return event.key
