from typing import Any, Final

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner_v2 import BaseSB3Runner
from syn_grid.runners.agent_runners.sb3.execution_strategy import (
    StatelessExecutionStrategy,
)
from syn_grid.runners.agent_runners.sb3.policy_resolver import resolve_policy


class FrameStackPPO(BaseSB3Runner[PPO]):
    """PPO with a fixed window of stacked observations for short-term memory."""

    # ================= #
    #       Init        #
    # ================= #

    _N_STACK: Final[int] = 4
    _HYPER_PARAMETERS: Final[dict[str, Any]] = {
        **BaseSB3Runner._SHARED_HYPER_PARAMETERS,
        "device": "cpu",
    }

    def __init__(self, agent_bundle: AgentBundle):
        policy = resolve_policy(agent_bundle.obs_conf.observation_handler.perception)
        hyper_parameters = {"policy": policy, **self._HYPER_PARAMETERS}

        super().__init__(
            agent_bundle,
            hyper_parameters,
            PPO,
            execution_strategy=StatelessExecutionStrategy(),
        )
        print("Initializing stateless PPO with frame stacking...")

    # ================= #
    #       Hooks       #
    # ================= #

    def _build_env(self, render_mode: str | None, sub_dir: str) -> VecEnv:
        env = super()._build_env(render_mode, sub_dir)
        return VecFrameStack(env, self._N_STACK)
