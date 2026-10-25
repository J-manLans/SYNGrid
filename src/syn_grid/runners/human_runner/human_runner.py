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
    human input and pass actions to the environment, one episode after
    another. When an episode ends, it shows a summary over the last frame:
    the scenario's HUD values, its metrics and the last step's reward.
    Space then plays the next episode, and Escape or closing the window
    quits, as it does during an episode.
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
        self._env = cast(SYNGridEnv, make(scenario, "human").unwrapped)

    # ================= #
    #        API        #
    # ================= #

    def play(self) -> None:
        """Play episodes one after another, blocking until the player quits."""

        try:
            while True:
                player_quit = self._play_episode()
                if player_quit:
                    break
        finally:
            self._env.close()

    # ================= #
    #      Helpers      #
    # ================= #

    def _play_episode(self) -> bool:
        """
        Play one episode and show its summary.

        Returns:
            Whether the player quit, during the episode or on its summary.
        """

        self._env.reset()

        while True:
            action = self._get_user_action()

            # None is the player quitting.
            if action is None:
                return True

            _, reward, terminated, truncated, info = self._env.step(action.value)

            if terminated or truncated:
                return self._show_summary(reward, info)

    def _show_summary(self, reward: float, info: dict[str, Any]) -> bool:
        """
        Show the episode's summary and wait for Space or quitting. Other keys are ignored.

        Returns:
            Whether the player quit instead of playing again.
        """

        summary = {
            **self._env.get_hud_data(),
            **info[SYN_STATS_KEY],
            "last reward": reward,
        }

        self._env.renderer.draw_summary(summary)

        while True:
            key = self._next_key()

            if key is None:
                return True
            if key == pygame.K_SPACE:
                return False

    def _get_user_action(self) -> DroidAction | None:
        """Blocks until an arrow key is pressed and returns its action, or None on quitting."""

        while True:
            key = self._next_key()

            if key is None:
                return None

            action = self._KEY_ACTIONS.get(key)
            if action is not None:
                return action

    def _next_key(self) -> int | None:
        """
        Blocks until a key is pressed and returns it.

        Escape or closing the window returns None, which is the player quitting. Other input is
        ignored.
        """

        while True:
            event = pygame.event.wait()

            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return None
            if event.type == pygame.KEYDOWN:
                return event.key
