"""
Scenario selection.

This module is the whole of "which scenario is this". A config names one, a
builder composes the pieces for it, and nothing downstream re-derives identity
from the config. The builders live in their family's folder, such as
``tier_chain/builders.py``. Registering a new scenario means writing its builder
there and adding it to ``SCENARIO_BUILDERS`` here.

Skeleton: structure and signatures only. Each docstring says what the function
composes; the bodies are left out.
"""


from collections.abc import Callable

from syn_grid.config.models.common_models import ScenarioConf
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.scenario.scenario import Scenario
from syn_grid.scenario.goal.tier_chain.builders import (
    build_tier_chain_spatial,
    build_tier_chain_scaling_sparse,
    build_tier_chain_scaling_dense,
    build_tier_chain_delay
)


# ============ #
#   Registry   #
# ============ #


ScenarioBuilder = Callable[[str, ScenarioConf], Scenario]


SCENARIO_BUILDERS: dict[ScenarioName, ScenarioBuilder] = {
    ScenarioName.GOAL_TIER_CHAIN_SPATIAL: build_tier_chain_spatial,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: build_tier_chain_scaling_sparse,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: build_tier_chain_scaling_dense,
    ScenarioName.GOAL_TIER_CHAIN_DELAY: build_tier_chain_delay,
}


def build_scenario(scenario: ScenarioName, scenario_conf: ScenarioConf) -> Scenario:
    """Look the name up in `SCENARIO_BUILDERS` and run its builder."""
    ...
