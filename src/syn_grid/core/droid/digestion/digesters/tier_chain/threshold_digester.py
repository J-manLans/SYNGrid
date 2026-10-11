"""
Threshold scoring: a chain pays what it has gathered when it ends.
"""

from typing import Final

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb


class ThresholdDigester(TierDigester):
    """Holds a chain's worth while it grows, and pays it when the chain ends.

    A growing chain pays nothing. The orb that completes it pays what all the
    orbs in the chain are worth together. A tier orb eaten out of order pays
    the chain break penalty plus what the chain had gathered so far.

    A tier 1 orb eaten while the chain is at tier 1 pays the penalty alone.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, max_tier: int, chain_break_penalty: float):
        super().__init__(max_tier)

        self._CHAIN_BREAK_PENALTY: Final[float] = chain_break_penalty

    # ================= #
    #      Scoring      #
    # ================= #

    def _score_progress(self, orb: BaseOrb) -> float:
        return 0.0

    def _score_break(self, orb: BaseOrb) -> float:
        repeated_base_tier = (
            self.chained_tiers == self._BASE_TIER and orb.META.TIER == self._BASE_TIER
        )

        if repeated_base_tier:
            return self._CHAIN_BREAK_PENALTY

        return self._CHAIN_BREAK_PENALTY + self.chain_reward

    def _score_completion(self, orb: BaseOrb) -> float:
        return self.chain_reward + orb.REWARD
