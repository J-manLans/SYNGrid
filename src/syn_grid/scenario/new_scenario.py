"""
Scenario: the fully composed simulation used by a SYNGrid benchmark.

A Scenario is the composition root of the runtime. Scenario builders construct
the concrete world and its scenario-specific components, then assemble them
here.

The Scenario itself contains no scenario logic. It simply owns the composed
simulation and the rules used by the Gymnasium adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.termination import TerminationRules


@dataclass(frozen=True)
class Scenario:
    """
    A fully composed SYNGrid scenario.

    The builder decides which concrete components make up the scenario.
    GridWorld runs the simulation; observation and termination rules define
    how that simulation is exposed as a benchmark task.
    """

    name: str
    tag: str
    world: GridWorld
    observation: ObservationRules
    termination: TerminationRules
