"""
Scenario: the domain concept that owns a benchmark scenario's rules.

SYNGrid is a benchmark, so what varies between experiments -- the goal, the orb
field, what the agent can observe, when an episode ends, how reward is scored
-- is the substance of the work, not incidental configuration.

``Scenario`` is a recipe and nothing more: an identity, the pieces the
Gymnasium side needs, and a way to build the world. It holds no behaviour and no
state of its own, so one scenario can be handed to any number of environments
and each builds a world nobody else touches.

Everything that is specific to a scenario and needed outside the world is a
field here. Adding a scenario is a new builder in its family's folder,
registered in ``registry.py``, plus a new module beside the builder if it
introduces a genuinely new mechanic. Nothing outside this package needs to learn
the scenario's name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.blocks.hud import HudElement
from syn_grid.scenario.blocks.metrics import Metric
from syn_grid.scenario.blocks.observation import ObservationRules
from syn_grid.scenario.blocks.termination import TerminationRules


@dataclass(frozen=True)
class Scenario:
    """A named scenario, its rules, and how to build its world.

    Attributes:
        scenario_name: the scenario's registered name, as written in the
            config's ``scenario:`` key. This is the scenario's identity and it
            is what the config selects on.
        scenario_tag: where this run sits on the scenario's own axis, for run
            ids: the grid for spatial, tier and grid for tier scaling, the delay
            for delay.
        observation: what the agent sees and how wide the observation is.
        termination: when the episode ends and what the last step pays.
        hud: what the HUD shows for this scenario, in order.
        metrics: what this scenario reports at the end of an episode.
        build_world: returns a new world, with its own droid, orbs and
            digestion, on every call.
    """

    scenario_name: str
    scenario_tag: str
    observation: ObservationRules
    termination: TerminationRules
    hud: tuple[HudElement, ...]
    metrics: tuple[Metric, ...]
    build_world: Callable[[], GridWorld]
