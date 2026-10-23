"""
When a tier chain episode ends.

This is the tier-chain implementation of ``TerminationRules``.
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
class TierChainTermination:
    """Ends a tier chain episode, and states what the ending costs.

    An episode ends when the droid is out of energy, when the chain is broken,
    when the chain is completed, or when the steps run out.

    The ending costs the timeout penalty when the steps run out and nothing
    else ended the episode, and nothing otherwise. What the running chain is
    owed is digestion's, and the droid settles it.

    Attributes:
        timeout_penalty: what the clock running out costs.
    """

    timeout_penalty: float

    def evaluate(self, world: GridWorld) -> EpisodeOutcome:
        tier_digester = world.droid.get_digester(TierDigester)

        # Gather ending states
        out_of_energy = world.droid.energy <= 0
        chain_ended = (
            tier_digester.chains_broken > 0 or tier_digester.chains_completed > 0
        )
        out_of_steps = world.steps_left <= 0

        # Check if episode ended
        if not (out_of_energy or chain_ended or out_of_steps):
            return EpisodeOutcome(terminated=False)

        # Decide what should be returned depending on the termination reason.
        # The clock is charged only when it alone ended the episode. A chain completed or broken
        # on the last step, or energy running out on it, is not also a timeout.
        if out_of_steps and not (out_of_energy or chain_ended):
            return EpisodeOutcome(terminated=True, ending_reward=self.timeout_penalty)

        return EpisodeOutcome(terminated=True)
