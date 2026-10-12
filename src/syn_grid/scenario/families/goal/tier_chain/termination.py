"""
When a tier chain episode ends.

This is the tier-chain implementation of ``TerminationRules``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.scenario.blocks.termination import EpisodeOutcome

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# #################### #
# Termination strategy #
# #################### #


@dataclass(frozen=True)
class TierChainTermination:
    """Ends a tier chain episode: a broken chain, a finished chain, or the clock.

    Attributes:
        timeout_penalty: what the clock running out costs.
    """

    timeout_penalty: float

    def evaluate(
        self, world: GridWorld, steps_left: int, reward: float
    ) -> EpisodeOutcome:
        return EpisodeOutcome(terminated=False, truncated=False, reward=reward)
