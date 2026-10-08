from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.core.digestion.tier_digester import TierOrbDigester
from syn_grid.scenario.rules.termination.termination import EpisodeOutcome


if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld

# ###################### #
#  Termination Strategy  #
# ###################### #

@dataclass(frozen=True)
class ContinuousTermination:
    """Ends a continuous episode: only the score running out, or the clock.

    There is no chain to break and nothing to complete, so the clock is the
    only interesting boundary. A timeout still settles a partially accumulated
    reward.
    """

    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        terminated = world.droid.score <= 0

        if steps_left <= 0:
            reward = world.droid.digestion_engine.get(TierOrbDigester).pending_reward
            terminated = True

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)