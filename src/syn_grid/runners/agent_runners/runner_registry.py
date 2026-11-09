"""
Which runner each algorithm in the runner config gets.

`RUNNER` maps every `Algorithm` to what makes its runner from a scenario and a runner config: an `SB3Spec` for
the algorithms SYNGrid trains with SB3, or a `BaseAgentRunner` subclass for one of your own.
Adding a runner of your own is a member in `Algorithm` (`config/models/runner_models.py`) and an
entry here.
"""

from collections.abc import Callable

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from syn_grid.config.models.runner_models import Algorithm, RunnerConf
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.sb3.sb3_runner import SB3Spec
from syn_grid.scenario.scenario import Scenario

RUNNER: dict[Algorithm, Callable[[Scenario, RunnerConf], BaseAgentRunner]] = {
    # PPO with no memory beyond the current observation.
    Algorithm.PPO: SB3Spec(PPO, device="cpu"),
    # PPO with a fixed window of stacked observations, for short-term memory.
    Algorithm.FSPPO: SB3Spec(PPO, device="cpu", frame_stack=4),
    # Recurrent PPO, with an LSTM for episodic memory.
    Algorithm.RPPO: SB3Spec(
        RecurrentPPO,
        device="cuda",
        hyper_parameters={
            "policy_kwargs": {
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "shared_lstm": False,
            }
        },
    ),
}


def build_runner(scenario: Scenario, runner_conf: RunnerConf) -> BaseAgentRunner:
    """
    Make the runner registered for the configured algorithm.

    Raises:
        KeyError: If the configured algorithm has no registered runner.
    """

    alg = runner_conf.common_conf.alg
    if alg not in RUNNER:
        raise KeyError(
            f"No runner registered for algorithm '{alg.value}'. "
            f"Available: {[a.value for a in RUNNER]}"
        )

    return RUNNER[alg](scenario, runner_conf)
