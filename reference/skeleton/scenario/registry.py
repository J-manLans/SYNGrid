"""
Scenario selection.

This module is the whole of "which scenario is this". A config names one, its
entry here says which config class validates its config file and which builder composes
its pieces, and nothing downstream re-derives identity from the config. The
config classes and the builders live in their family's folder, such as
``goal/tier_chain/config.py`` and ``goal/tier_chain/builders.py``. Registering a
new scenario means writing its builder there and adding an entry to
``SCENARIOS`` here.

Skeleton: structure and signatures only. Each docstring says what the function
composes; the bodies are left out.
"""


from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.config.models.common_scenario_models import ScenarioConf
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.scenario.scenario import Scenario
from syn_grid.scenario.goal.tier_chain.builders import (
    build_tier_chain_spatial,
    build_tier_chain_scaling_sparse,
    build_tier_chain_scaling_dense,
    build_tier_chain_delay
)
from syn_grid.scenario.goal.tier_chain.config import (
    TierDelayScenarioConf,
    TierDenseScenarioConf,
    TierScenarioConf,
)


# ============ #
#   Registry   #
# ============ #


ScenarioBuilder = Callable[[str, ScenarioConf], Scenario]


@dataclass(frozen=True)
class ScenarioEntry:
    """What a scenario name stands for: the config class its config file is
    validated against, and the builder that turns that config into a
    `Scenario`."""

    conf_class: type[ScenarioConf]
    builder: ScenarioBuilder


SCENARIOS: dict[ScenarioName, ScenarioEntry] = {
    ScenarioName.GOAL_TIER_CHAIN_SPATIAL: ScenarioEntry(
        TierScenarioConf, build_tier_chain_spatial
    ),
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_SPARSE: ScenarioEntry(
        TierScenarioConf, build_tier_chain_scaling_sparse
    ),
    ScenarioName.GOAL_TIER_CHAIN_TIER_SCALING_DENSE: ScenarioEntry(
        TierDenseScenarioConf, build_tier_chain_scaling_dense
    ),
    ScenarioName.GOAL_TIER_CHAIN_DELAY: ScenarioEntry(
        TierDelayScenarioConf, build_tier_chain_delay
    ),
}


def build_scenario(scenario: ScenarioName, scenario_conf: ScenarioConf) -> Scenario:
    """Look the name up in `SCENARIOS` and run its entry's builder."""
    ...
