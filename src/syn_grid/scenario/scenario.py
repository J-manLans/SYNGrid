"""
Scenario: the domain concept that owns a benchmark scenario's rules.

SYNGrid is a benchmark, so what varies between experiments -- the goal, the orb
field, what the agent can observe, when an episode ends, how reward is scored
-- is the substance of the work, not incidental configuration. Before this
existed, that knowledge was reconstructed from unrelated booleans at each site
that needed it, which meant two sites could read the same config and conclude
different things.

``Scenario`` is a recipe and nothing more: an identity, the rules the
Gymnasium adapter needs, and a way to build the world. It holds no behaviour
and no state of its own, so one scenario can be handed to any number of
environments and each builds a world nobody else touches.

Adding a scenario is a new builder in ``registry.py`` plus, if it introduces a
genuinely new mechanic, a new rules module. Nothing outside this package needs
to learn the scenario's name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.core.grid_world import GridWorld
from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.termination import TerminationRules


@dataclass(frozen=True)
class Scenario:
    """A named scenario, its rules, and how to build its world.

    Attributes:
        name: the scenario's registered name, as written in the config's
            ``scenario:`` key. This is the scenario's identity and it is what
            the config selects on.
        tag: where this run sits on the scenario's own axis, for run ids: the
            grid for spatial, tier and grid for tier scaling, the delay for
            delay.
        observation: what the agent sees and how wide the observation is.
        termination: when the episode ends and what the last step pays.
        build_world: returns a new world, with its own droid, orbs and
            digestion, on every call.
    """

    scenario_name: str
    scenario_tag: str
    observation: ObservationRules
    termination: TerminationRules
    build_world: Callable[[], GridWorld]
