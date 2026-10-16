"""
Step-wise scoring: every tier orb eaten in order pays right away.
"""

from typing import Final

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb


class StepWiseDigester(TierDigester):
    """Pays each orb of a chain when it is eaten.

    An orb that grows or completes the chain pays its own reward. A tier orb
    eaten out of order pays the chain break penalty and not its reward.

    A tier 1 orb that breaks a longer chain starts a new one, so it pays the
    penalty and its reward. Eaten while the chain is at tier 1 it pays the
    penalty alone.

    A chain that is running when the episode ends is owed nothing, because
    each of its orbs has been paid.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, max_tier: int, chain_break_penalty: float):
        super().__init__(max_tier)

        self._CHAIN_BREAK_PENALTY: Final[float] = chain_break_penalty

    # ================= #
    #        API        #
    # ================= #

    def undigested(self) -> float:
        return 0.0

    # ================= #
    #      Scoring      #
    # ================= #

    def _score_progress(self, orb: BaseOrb) -> float:
        return orb.REWARD

    def _score_break(self, orb: BaseOrb) -> float:
        restarts_chain = (
            orb.META.TIER == self._BASE_TIER and self.chain_tier != self._BASE_TIER
        )

        if restarts_chain:
            return self._CHAIN_BREAK_PENALTY + orb.REWARD

        return self._CHAIN_BREAK_PENALTY

    def _score_completion(self, orb: BaseOrb) -> float:
        return orb.REWARD
