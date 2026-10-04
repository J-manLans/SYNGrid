"""
Scenario: the first-class domain concept that owns a benchmark scenario.

The benchmark varies by scenario; this package is where a scenario lives.
"""

from syn_grid.scenario.registry import SCENARIO_BUILDERS, build_scenario
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import OrbPopulation
from syn_grid.scenario.rules.spawning import SpawningRules
from syn_grid.scenario.rules.termination import (
    EpisodeOutcome,
    TerminationRules,
)
from syn_grid.scenario.scenario import Scenario, ScenarioType

__all__ = [
    "SCENARIO_BUILDERS",
    "EpisodeOutcome",
    "ObservationRules",
    "OrbPopulation",
    "Scenario",
    "ScenarioType",
    "SpawningRules",
    "TerminationRules",
    "build_scenario",
]
