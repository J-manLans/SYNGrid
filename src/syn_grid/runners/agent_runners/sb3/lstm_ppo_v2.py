from typing import Any, Final

from sb3_contrib import RecurrentPPO

from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner_v2 import BaseSB3Runner
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

    def __init__(self, agent_bundle: AgentBundle):
        policy = resolve_policy(
            agent_bundle.obs_conf.observation_handler.perception, use_lstm=True
        )
        hyper_parameters = {"policy": policy, **self._HYPER_PARAMETERS}

        super().__init__(
            agent_bundle,
            hyper_parameters,
            RecurrentPPO,
            execution_strategy=RecurrentExecutionStrategy(),
        )
        print("Initializing RecurrentPPO...")

    # No _build_env override needed: the default DummyVecEnv + VecNormalize
    # pipeline in BaseSB3Runner is exactly what LSTM trains on too. The only
    # thing that differs for this agent is eval-time action selection, which
    # is handled by RecurrentExecutionStrategy above.
