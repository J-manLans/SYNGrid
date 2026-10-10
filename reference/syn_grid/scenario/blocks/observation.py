"""
What a scenario lets the agent see.

The scenario owns how many orb slots an observation is built to hold, because
that is a property of the world rather than of the perception encoding. It
also states which perception encodes the world and the bounds that encoding is
scaled against, so the Gymnasium adapter is told everything it needs and never
reads a config.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models.scenarios.common_models import Perception


@dataclass(frozen=True)
class ObservationRules:
    """How the agent's observation is built.

    Attributes:
        perception: which perception encodes the world.
        max_steps: the episode's step budget.
        max_score: the upper bound a score is encoded against, or None for
            a scenario whose observation does not carry the score.
        observation_slot_count: sizes the observation vector.
        sort_limit: caps the distance sort that chooses which orbs are written
            into those slots.
        max_tier: the upper bound a tier value is encoded against, for the
            perceptions that give tier its own channel.
    """

    perception: Perception
    max_steps: int
    max_score: int | None # would go away
    observation_slot_count: int
    sort_limit: int
    max_tier: int
    # global_features: tuple[GlobalFeature, ...]

    @dataclass(frozen=True)
    class GlobalFeature:
        high: float                              # static: the Box bound
        read: Callable[[GridWorld], float]

