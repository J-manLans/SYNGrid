from typing import Any, Final

from sb3_contrib import RecurrentPPO

from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.runners.agent_runners.sb3.execution_strategy import (
    RecurrentExecutionStrategy,
)
from syn_grid.runners.agent_runners.sb3.policy_resolver import resolve_policy


class LstmPPO(BaseSB3Runner[RecurrentPPO]):
    """Recurrent PPO with an LSTM-based episodic memory."""

    # ================= #
    #       Init        #
    # ================= #

    _HYPER_PARAMETERS: Final[dict[str, Any]] = {
        **BaseSB3Runner._SHARED_HYPER_PARAMETERS,
        "device": "cuda",
        "policy_kwargs": {
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "shared_lstm": False,
        },
    }

    def __init__(self, runner_bundle: RunnerBundle):
        policy = resolve_policy(
            runner_bundle.scenario.perception, use_lstm=True
        )
        hyper_parameters = {"policy": policy, **self._HYPER_PARAMETERS}

        super().__init__(
            runner_bundle,
            hyper_parameters,
            RecurrentPPO,
            execution_strategy=RecurrentExecutionStrategy(),
        )
        print("Initializing RecurrentPPO...")
