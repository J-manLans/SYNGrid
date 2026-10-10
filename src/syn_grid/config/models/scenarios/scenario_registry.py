"""
Which configuration model belongs to which scenario name.

The dispatch table and nothing else. Each family's models live in that family's
file; this is the one place that has to know all of them, because it is the join
between a name in a config and a schema to validate it against.

The mapping is deliberately many-to-one where the schema is. Four tier-chain
names point at one hierarchy because they are configured identically and differ
only in rules -- which is what makes `tier_chain.py` a family file rather than
four files. A name gets its own model when it needs a field the family lacks,
and that subclass lives beside the family it departs from.

What this must never do is infer a scenario from the values it is given.
"""

from syn_grid.config.models.scenarios.common_models import ScenarioConf
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenarios.tier_chain_models import (
    TierDelayScenarioConf,
    TierDenseScenarioConf,
    TierScenarioConf,
)

SCENARIO_MODELS: dict[ScenarioName, type[ScenarioConf]] = {
    ScenarioName.GOAL_TIER_CHAIN_SPATIAL: TierScenarioConf,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: TierScenarioConf,
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: TierDenseScenarioConf,
    ScenarioName.GOAL_TIER_CHAIN_DELAY: TierDelayScenarioConf,
}


