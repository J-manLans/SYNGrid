"""
What a scenario lets the agent see.

The scenario owns how many orb slots an observation is built to hold, because
that is a property of the world rather than of the perception encoding. It
also states which perception encodes the world, which global values the
observation carries and the bounds they are scaled against, so the Gymnasium
adapter is told everything it needs and never reads a config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.config.models.scenario.common_models import Perception

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


@dataclass(frozen=True)
class GlobalFeature:
    """One global value in the observation, with its bound.

    Attributes:
        high: the upper bound the value is encoded against; the perception
            declares its ``Box`` from it.
        read: returns the current value from the world it is handed.
    """

    high: float
    read: Callable[[GridWorld], float]


@dataclass(frozen=True)
class ObservationRules:
    """How the agent's observation is built.

    Attributes:
        perception: which perception encodes the world.
        observation_slot_count: sizes the observation vector.
        sort_limit: caps the distance sort that chooses which orbs are written
            into those slots.
        global_features: the global values this scenario's observation carries,
            in order. A scenario leaves out what its agent should not see.
    """

    perception: Perception
    observation_slot_count: int
    sort_limit: int
    global_features: tuple[GlobalFeature, ...]
