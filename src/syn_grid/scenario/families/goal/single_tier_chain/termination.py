"""
When a single tier chain episode ends.

This is the single tier chain implementation of ``TerminationRules``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.scenario.blocks.termination import EpisodeOutcome

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# #################### #
# Termination strategy #
# #################### #


@dataclass(frozen=True)
class SingleTierChainTermination:
    """Ends a single tier chain episode, and states what the ending costs.

    An episode ends when the droid ends the task, or when the steps run out. The droid ends the
    task when it is out of energy, when the chain is broken, or when the chain is completed.

    The ending costs the timeout penalty when the steps run out and the droid did not end the
    task, and nothing otherwise. What the running chain is owed is digestion's, and the droid
    settles it.

    Attributes:
        timeout_penalty: what the clock running out costs.
    """

    timeout_penalty: float

    def evaluate(self, world: GridWorld) -> EpisodeOutcome:
        tier_digester = world.droid.get_digester(TierDigester)

        droid_ended_task = (
            world.droid.energy <= 0
            or tier_digester.chains_broken > 0
            or tier_digester.chains_completed > 0
        )
        out_of_steps = world.steps_left <= 0

        if not (droid_ended_task or out_of_steps):
            return EpisodeOutcome(terminated=False)

        # The clock is charged only when it alone ended the episode. A task ended on the last step
        # is not also a timeout.
        if not droid_ended_task:
            return EpisodeOutcome(terminated=True, ending_reward=self.timeout_penalty)

        return EpisodeOutcome(terminated=True)
