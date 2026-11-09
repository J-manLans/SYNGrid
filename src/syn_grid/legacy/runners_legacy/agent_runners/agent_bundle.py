# agent_bundle.py

from dataclasses import dataclass

from syn_grid.legacy.config_legacy.models import AgentConfig, ObsConfig, WorldConfig


@dataclass
class AgentBundle:
    """Bundled configuration needed for the agent."""

    world_conf: WorldConfig
    obs_conf: ObsConfig
    agent_conf: AgentConfig
