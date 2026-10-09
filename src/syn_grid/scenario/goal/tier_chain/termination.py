from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.core.digestion.tier_digester import (
    ChainBroken,
    ChainCompleted,
)
from syn_grid.scenario.blocks.termination import EpisodeOutcome

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# ###################### #
#  Termination Strategy  #
# ###################### #


@dataclass(frozen=True)
class TierChainTermination:
    """Ends a tier chain episode: a broken chain, a finished chain, or the clock.

    The only reward set here is the timeout. A break and a completion keep the
    reward digestion gave them.

    Attributes:
        timeout_penalty: paid for reaching the step limit unfinished.
    """

    timeout_penalty: float # TODO: think this should be optional

    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        engine = world.droid.digestion_engine

        # The score ending is switched off for tier chains. Thesis: it adds
        # nothing here. A wall hit is already punished when it happens, the
        # score can only fall because the one positive reward ends the episode,
        # and under fog of war the agent cannot see the score that would end it.
        # TODO: once the repo is stable, do one training run without it, then
        # either delete this line or restore it.
        # terminated = world.droid.score <= 0
        terminated = False

        if engine.count(ChainBroken) > 0 or engine.count(ChainCompleted) > 0:
            terminated = True

        elif steps_left <= 0:
            reward = self.timeout_penalty
            terminated = True

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)