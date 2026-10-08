"""
When an episode ends, and what the last step is worth.

A termination rule belongs to an objective, not to a kind of scenario. Tier
chain and continuous are the two that exist. Another goal scenario brings its
own rule behind ``TerminationRules`` rather than extending one of these: there
is no shared base class, because the two have too little in common to earn one.

A rule decides when the episode ends and sets the one reward the clock causes.
A reward caused by consuming an orb comes from digestion and is passed through
untouched.

``truncated`` is always False. Running out of steps is a punishment and sets
``terminated``, so no value function is bootstrapped at the horizon. That is a
deliberate decision with a real cost, pinned by a test so it is not later
"fixed" by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# ################## #
#     Interface      #
# ################## #


@dataclass(frozen=True)
class EpisodeOutcome:
    terminated: bool
    truncated: bool
    reward: float


class TerminationRules(Protocol):
    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        ...