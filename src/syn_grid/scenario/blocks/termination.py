"""
When an episode ends, and what the last step is worth.

A termination rule belongs to an objective, not to a kind of scenario. This
file is the interface only. Each strategy lives in the folder of the scenario
family that uses it, as ``families/goal/tier_chain/termination.py`` does, and imports
only what that family needs. There is no shared base class, because the
strategies have too little in common to earn one.

A rule decides when the episode ends, and what the ending itself costs, such as
the clock running out. It changes nothing. A reward caused by consuming an orb
comes from digestion, and what digestion is still owed when the episode ends is
settled by the droid, so neither passes through the rule.

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
        ending_reward: what the ending itself costs or pays, on top of what the
            droid took in this step. It is 0 when the episode goes on.
        truncated: the clock stopped an episode that has no task to fail. A
            goal scenario leaves it at False.
    """

    terminated: bool
    ending_reward: float = 0.0
    truncated: bool = False


class TerminationRules(Protocol):
    def evaluate(self, world: GridWorld) -> EpisodeOutcome:
        """Decide whether the episode ends this step, and what the ending adds."""
        ...
