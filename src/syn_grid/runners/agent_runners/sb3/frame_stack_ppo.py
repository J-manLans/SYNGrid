from typing import Any, Final

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.runners.agent_runners.sb3.execution_strategy import (
    StatelessExecutionStrategy,
)


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

    def __init__(self, runner_bundle: RunnerBundle):
        super().__init__(
            runner_bundle,
            self._HYPER_PARAMETERS,
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
