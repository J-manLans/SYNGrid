"""
When an episode ends, and what the last step is worth.

A termination rule belongs to an objective, not to a kind of scenario. This
file is the interface only. Each strategy lives in the folder of the scenario
family that uses it, as ``families/goal/tier_chain/termination.py`` does, and imports
only what that family needs. There is no shared base class, because the
strategies have too little in common to earn one.

A rule decides when the episode ends, and only adds to the step's reward: what
the clock causes, and what is still owed when the episode ends. A reward caused
by consuming an orb comes from digestion and is passed through untouched.

An episode ends in one of two ways. It is terminated when the task is over, and
truncated when the clock stops an episode that has no task to fail. A goal
scenario never truncates: its deadline is part of the task, so running out of
steps is a punishment and sets ``terminated``, and no value function is
bootstrapped at the horizon. That is a deliberate decision with a real cost.
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
    """What a termination rule decides for one step.

    Attributes:
        terminated: the task is over, reached or failed.
        reward: what the step pays, with what the ending adds.
        truncated: the clock stopped an episode that has no task to fail. A
            goal scenario leaves it at False.
    """

    terminated: bool
    reward: float
    truncated: bool = False


class TerminationRules(Protocol):
    def evaluate(
        self, world: GridWorld, steps_left: int, reward: float
    ) -> EpisodeOutcome:
        """Decide whether the episode ends this step, and what the step pays."""
        ...
