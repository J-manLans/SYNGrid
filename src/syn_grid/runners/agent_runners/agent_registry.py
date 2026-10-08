from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.sb3 import FrameStackPPO, LstmPPO, StatelessPPO
from syn_grid.runners.human_runner.human_runner import HumanRunner

RUNNER: dict[str, type[BaseAgentRunner]] = {
    "PPO": StatelessPPO,
    "FSPPO": FrameStackPPO,
    "RPPO": LstmPPO,
}


def build_runner(human_control: bool, runner_bundle: RunnerBundle) -> BaseAgentRunner:
    """
    Instantiate the agent runner class registered for the configured algorithm.
    If human play mode is enabled we return that early.

    Args:
        agent_conf: Agent configuration, including which algorithm to run.
        obs_conf: Observation space configuration.
        world_conf: World/environment configuration.

    Returns:
        An instance of the BaseAgentRunner subclass registered under
        `agent_conf.global_agent_conf.alg` or a `HumanRunner` instance.

    Raises:
        KeyError: If the configured algorithm has no registered runner.
    """

    if human_control:
        return HumanRunner(runner_bundle)

    alg = runner_bundle.runner_conf.common_conf.alg
    if alg not in RUNNER:
        raise KeyError(
            f"No runner registered for algorithm '{alg}'. Available: {list(RUNNER)}"
        )

    return RUNNER[alg](runner_bundle)
