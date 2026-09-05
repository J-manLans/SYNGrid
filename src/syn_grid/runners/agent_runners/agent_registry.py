from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.sb3 import FrameStackPPO, LstmPPO, StatelessPPO

ALGORITHMS: dict[str, type[BaseAgentRunner]] = {
    "PPO": StatelessPPO,
    "FSPPO": FrameStackPPO,
    "RPPO": LstmPPO,
}


def build_runner(agent_bundle: AgentBundle) -> BaseAgentRunner:
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

    alg = agent_bundle.agent_conf.global_agent_conf.alg
    if alg not in ALGORITHMS:
        raise KeyError(
            f"No runner registered for algorithm '{alg}'. Available: {list(ALGORITHMS)}"
        )

    return ALGORITHMS[alg](agent_bundle)
