# agent_bundle.py

from dataclasses import dataclass

from syn_grid.config.models.runner_models import RunnerConf
from syn_grid.scenario.scenario import Scenario


@dataclass
class RunnerBundle:
    """
    Bundled configuration needed for the agent.

    ``scenario`` is the selected scenario. It travels with the bundle because
    a runner has to build the scenario's environment, and because a run's
    identity follows from the scenario.
    """

    scenario: Scenario
    runner_conf: RunnerConf
