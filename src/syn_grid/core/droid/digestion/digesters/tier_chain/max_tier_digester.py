"""
Max-tier scoring: a chain pays only when it is completed.
"""

from typing import Final

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb


class MaxTierDigester(TierDigester):
    """Pays a chain's whole worth at the last tier, and nothing before it.

    A growing chain pays nothing. The orb that completes it pays what all the
    orbs in the chain are worth together. A tier orb eaten out of order pays
    the chain break penalty, and what the chain had gathered is lost.
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
        return self._CHAIN_BREAK_PENALTY

    def _score_completion(self, orb: BaseOrb) -> float:
        return self.chain_reward + orb.REWARD
