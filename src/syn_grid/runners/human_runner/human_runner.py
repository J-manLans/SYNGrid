from typing import cast

from syn_grid.gymnasium.environment import SYNGridEnv
from syn_grid.gymnasium.utils.env_factory import make
from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner


class HumanRunner(BaseAgentRunner):
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

    def __init__(self, agent_bundle: AgentBundle):
        self._env = cast(
            SYNGridEnv,
            make(
                "human",
                agent_bundle.scenario,
                agent_bundle.world_conf,
                agent_bundle.obs_conf,
            ).unwrapped,
        )

    # ================= #
    #        API        #
    # ================= #
    def train(self) -> None:
        raise ValueError(
            "HumanRunner does not support training; set training=False in your config."
        )

    def eval(self) -> None:
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
