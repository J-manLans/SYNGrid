from syn_grid.config.models.runner_models import Algorithm
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.sb3 import FrameStackPPO, LstmPPO, StatelessPPO

RUNNER: dict[Algorithm, type[BaseAgentRunner]] = {
    Algorithm.PPO: StatelessPPO,
    Algorithm.FSPPO: FrameStackPPO,
    Algorithm.RPPO: LstmPPO,
}


def build_runner(runner_bundle: RunnerBundle) -> BaseAgentRunner:
    """
    Instantiate the agent runner class registered for the configured algorithm.

    Args:
        runner_bundle: The scenario and the runner configuration, including
            which algorithm to run.

    Returns:
        An instance of the BaseAgentRunner subclass registered under
        `runner_bundle.runner_conf.common_conf.alg`.

    Raises:
        KeyError: If the configured algorithm has no registered runner.
    """

    alg = runner_bundle.runner_conf.common_conf.alg
    if alg not in RUNNER:
        raise KeyError(
            f"No runner registered for algorithm '{alg.value}'. "
            f"Available: {[a.value for a in RUNNER]}"
        )

    return RUNNER[alg](runner_bundle)
