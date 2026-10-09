"""
Scenario selection.

This module is the whole of "which scenario is this". A config names one, a
builder composes the rules for it, and nothing downstream re-derives identity
from booleans. The builders live in their family's folder, such as
``tier_chain/builders.py``. Registering a new scenario means writing its builder
there and adding it to ``SCENARIO_BUILDERS`` here; the environment, the world,
the orb factory and the perceptions do not change.

The names are the domain's, not the config's. ``single_chain_mode`` described
a mechanism; "tier chain" describes the scenario that uses it, which is why the
old flag name is not carried forward.
"""

from syn_grid.config.models.common_models import ScenarioConf
from syn_grid.config.models.global_models import ScenarioName
from collections.abc import Callable
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
    """Resolve a scenario name into the rules that define it."""

    try:
        builder = SCENARIO_BUILDERS[scenario]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIO_BUILDERS)}"
        ) from None

    return builder(scenario.value, scenario_conf)
