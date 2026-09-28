"""
Scenario: the domain concept that owns a benchmark scenario's rules.

SYNGrid is a benchmark, so what varies between experiments -- the goal, the orb
field, what the agent can observe, when an episode ends, how reward is scored
-- is the substance of the work, not incidental configuration. Before this
existed, that knowledge was reconstructed from unrelated booleans at each site
that needed it, which meant two sites could read the same config and conclude
different things.

``Scenario`` is a composition root and nothing more: an identity plus the rule
objects that implement it. It holds no behaviour of its own, and the domain
hierarchy is expressed by which rules get composed rather than by a class
hierarchy. Goal/Tier Chain and Continuous share almost no mechanics, and
Spatial/Delay/Tier Scaling differ only in the rules they are given, so
inheritance would be a claim the code does not support.

Adding a scenario is a new builder in ``registry.py`` plus, if it introduces a
genuinely new mechanic, a new rules module. Nothing outside this package needs
to learn the scenario's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from syn_grid.scenario.rules.observation import ObservationRules
from syn_grid.scenario.rules.population import OrbPopulation
from syn_grid.scenario.rules.spawning import SpawningRules
from syn_grid.scenario.rules.termination import TerminationRules


class ScenarioKind(Enum):
    """The two roots of the scenario hierarchy.

    ``GOAL`` scenarios have an objective and end when it is met or lost.
    ``CONTINUOUS`` scenarios have no objective and end only when the clock runs
    out or the score does.
    """

    GOAL = "goal"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class Scenario:
    """A named scenario and the rules that define it.

    Attributes:
        name: the scenario's registered name, as written in the config's
            ``scenario:`` key. This is the scenario's identity and it is what
            the config selects on.
        kind: which root of the hierarchy this scenario belongs to.
        population: how the orb pool is built.
        spawning: how the orb field behaves over an episode.
        observation: how much of the orb field the observation holds.
        termination: when the episode ends and what the last step pays.
    """

    name: str
    kind: ScenarioKind
    population: OrbPopulation
    spawning: SpawningRules
    observation: ObservationRules
    termination: TerminationRules

    @property
    def is_goal(self) -> bool:
        return self.kind is ScenarioKind.GOAL
