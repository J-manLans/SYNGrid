from syn_grid.config.models import AgentConfig, ObsConfig, WorldConfig
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.sb3 import FrameStackPPO, LstmPPO, StatelessPPO

ALGORITHMS: dict[str, type[BaseAgentRunner]] = {
    "PPO": StatelessPPO,
    "FSPPO": FrameStackPPO,
    "RPPO": LstmPPO,
}


def build_runner(
    world_conf: WorldConfig, obs_conf: ObsConfig, agent_conf: AgentConfig
) -> BaseAgentRunner:
    """
    Instantiate the agent runner class registered for the configured algorithm.

    Args:
        agent_conf: Agent configuration, including which algorithm to run.
        obs_conf: Observation space configuration.
        world_conf: World/environment configuration.

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

    return ALGORITHMS[alg](world_conf, obs_conf, agent_conf)
