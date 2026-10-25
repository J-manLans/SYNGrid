from typing import Any, cast

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
                action = self._env.renderer.get_user_action()

                if action is None:
                    continue

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
        self._env.renderer.wait_for_key()
