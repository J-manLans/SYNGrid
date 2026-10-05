from typing import Any, Final

from stable_baselines3 import PPO

from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.runners.agent_runners.sb3.execution_strategy import (
    StatelessExecutionStrategy,
)
from syn_grid.runners.agent_runners.sb3.policy_resolver import resolve_policy


class StatelessPPO(BaseSB3Runner[PPO]):
    """PPO with no memory beyond the current observation."""

    # ================= #
    #       Init        #
    # ================= #

    _HYPER_PARAMETERS: Final[dict[str, Any]] = {
        **BaseSB3Runner._SHARED_HYPER_PARAMETERS,
        "device": "cpu",
    }

    def __init__(self, runner_bundle: RunnerBundle):
        policy = resolve_policy(
            runner_bundle.scenario.observation.perception
        )
        hyper_parameters = {"policy": policy, **self._HYPER_PARAMETERS}

        super().__init__(
            runner_bundle,
            hyper_parameters,
            PPO,
            execution_strategy=StatelessExecutionStrategy(),
        )
        print("Initializing stateless PPO...")
