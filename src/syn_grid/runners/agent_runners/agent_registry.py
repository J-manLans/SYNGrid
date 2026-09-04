from syn_grid.runners.agent_runners.sb3 import StatelessPPO, LstmPPO, FrameStackPPO
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.config.models import AgentConfig, ObsConfig, WorldConfig

from typing import Type

ALGORITHMS: dict[str, Type[BaseAgentRunner]] = {
    "PPO": StatelessPPO,
    "FSPPO": FrameStackPPO,
    "RPPO": LstmPPO,
}


def build_runner(
    agent_conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig
) -> BaseAgentRunner:
    """
    Instantiate the agent runner class registered for the configured algorithm.

    Args:
        agent_conf: Agent configuration, including which algorithm to run.
        obs_conf: Observation space configuration.
        run_conf: World/environment configuration.

    Returns:
        An instance of the BaseAgentRunner subclass registered under
        `agent_conf.global_agent_conf.alg`.

    Raises:
        KeyError: If the configured algorithm has no registered runner.
    """

    alg = agent_conf.global_agent_conf.alg
    if alg not in ALGORITHMS:
        raise KeyError(
            f"No runner registered for algorithm '{alg}'. Available: {list(ALGORITHMS)}"
        )

    return ALGORITHMS[alg](agent_conf, obs_conf, run_conf)
