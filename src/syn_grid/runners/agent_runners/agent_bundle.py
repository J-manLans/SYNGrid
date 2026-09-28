# agent_bundle.py

from dataclasses import dataclass

from syn_grid.config.models import AgentConfig, ObsConfig, WorldConfig


@dataclass
class AgentBundle:
    """Bundled configuration needed for the agent.

    ``scenario`` is the selected scenario's name. It travels with the bundle
    because a runner has to name the environment it is about to build, and
    because a run's identity should follow from the scenario rather than be
    reassembled from the parameters that happen to describe it.
    """

    scenario: str
    world_conf: WorldConfig
    obs_conf: ObsConfig
    agent_conf: AgentConfig
