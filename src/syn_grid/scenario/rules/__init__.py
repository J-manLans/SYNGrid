"""
Scenario rules.

Each module here owns one question a scenario has to answer about the world.
They are separate because they vary independently: a scenario can change how
orbs are drawn without changing when an episode ends, and vice versa. Folding
them into one object would mean a delay scenario and a curriculum scenario
sharing a constructor argument neither of them uses.
"""

from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import (
    OrbPopulation,
    TierChainPopulation,
    WeightedPopulation,
)
from syn_grid.scenario.rules.spawning import (
    AfterAction,
    LeaveFieldAlone,
    ReactivateAllOrbs,
    RefillOrbPool,
    SpawningRules,
)
from syn_grid.scenario.rules.termination import (
    COMPLETION_CEILING,
    ContinuousTermination,
    EpisodeOutcome,
    GoalTermination,
    TerminationRules,
)

__all__ = [
    "COMPLETION_CEILING",
    "AfterAction",
    "ContinuousTermination",
    "EpisodeOutcome",
    "GoalTermination",
    "LeaveFieldAlone",
    "ObservationRules",
    "OrbPopulation",
    "ReactivateAllOrbs",
    "RefillOrbPool",
    "SpawningRules",
    "TerminationRules",
    "TierChainPopulation",
    "WeightedPopulation",
]
