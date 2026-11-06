"""
Scenario selection.

This module is the whole of "which scenarios exist". It collects every family's catalogue into
`SCENARIOS`, and looks a name up there. What a scenario is, its config, visibility, lock and what
its family composes it from, is its entry in its family's ``scenarios.py``, such as
``families/goal/single_tier_chain/scenarios.py``. The entry itself is in ``entry.py``.

Registering a scenario means adding its entry to its family's catalogue. Registering a family
means adding its catalogue here. Nothing downstream re-derives identity from the config.
"""

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import ScenarioConf
from syn_grid.scenario.entry import ScenarioEntry
from syn_grid.scenario.families.goal.single_tier_chain.scenarios import (
    SINGLE_TIER_CHAIN_SCENARIOS,
)
from syn_grid.scenario.scenario import Scenario

# ============ #
#   Registry   #
# ============ #

SCENARIOS: dict[ScenarioName, ScenarioEntry] = {
    **SINGLE_TIER_CHAIN_SCENARIOS,
}


def build_scenario(scenario: ScenarioName, scenario_conf: ScenarioConf) -> Scenario:
    """Look the name up in `SCENARIOS` and build the scenario through its entry."""

    try:
        entry = SCENARIOS[scenario]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIOS)}"
        ) from None

    return entry.build(scenario.value, scenario_conf)
