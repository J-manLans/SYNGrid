from typing import cast

from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.scenario.scenario import Scenario


class HumanRunner:
    """
    Play a scenario by hand through a real SYNGridEnv.

    For inspecting a configuration, not for learning. It trains nothing,
    writes no checkpoints, and records no metrics. Its job is to read
    human input, pass actions to the environment, and stop when the
    episode ends.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, scenario: Scenario):
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

                _, _, terminated, truncated, _ = self._env.step(action.value)

                if terminated or truncated:
                    break
        finally:
            self._env.close()
