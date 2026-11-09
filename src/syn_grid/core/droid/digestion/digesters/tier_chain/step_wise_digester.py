"""
Step-wise scoring: every tier orb eaten in order pays right away.
"""

from typing import Final, override

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb


class StepWiseDigester(TierDigester):
    """Pays each orb of a chain when it is eaten.

    An orb that grows or completes the chain pays its own reward. A tier orb
    eaten out of order pays the chain break penalty and not its reward.

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

    @override
    def undigested(self) -> float:
        return 0.0

    # ================= #
    #      Scoring      #
    # ================= #

    @override
    def _score_progress(self, orb: BaseOrb) -> float:
        return orb.REWARD

    @override
    def _score_break(self, orb: BaseOrb) -> float:
        return self._CHAIN_BREAK_PENALTY

    @override
    def _score_completion(self, orb: BaseOrb) -> float:
        return orb.REWARD
