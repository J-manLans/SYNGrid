"""
How a tier chain fills the orb pool.

A tier chain has exactly one orb per tier and no weighting anywhere. This is
the tier-chain implementation of ``OrbPopulation``.
"""

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb
from syn_grid.core.orbs.synergy.chain.tier_orb import TierOrb


# ======================= #
#   Population Strategy   #
# ======================= #


class TierChainPopulation:
    """
    One orb per tier, tiers 1...max_tier.

    A tier chain is a fixed sequence, so the pool is the sequence. `max_tier` is the pool size,
    which is why a tier chain spawns `max_tier` orbs at reset rather than a configured count.
    """

    def __init__(
        self,
        max_tier: int,
        *,
        base_reward: float,
        growth_factor: float,
        linear_reward_growth: bool,
        cool_down: int,
    ) -> None:
        self._max_tier = max_tier
        self._base_reward = base_reward
        self._growth_factor = growth_factor
        self._linear_reward_growth = linear_reward_growth
        self._cool_down = cool_down

    def create(self) -> list[BaseOrb]:
        return [
            TierOrb(
                tier,
                self._max_tier,
                self._base_reward,
                self._growth_factor,
                self._linear_reward_growth,
                self._cool_down,
            )
            for tier in range(1, self._max_tier + 1)
        ]
