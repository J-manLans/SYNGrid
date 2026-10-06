from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    OrbCategory,
    OrbMeta,
    SynergyType,
)


class TierOrb(BaseOrb):
    """
    An orb that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 orb a tier 1 and tier 2 must have first been collected in that order without breaking the chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        tier: int,
        max_tier: int,
        base_reward: float,
        growth_factor: float,
        linear_reward_growth: bool,
        cool_down: int,
    ):
        if tier > max_tier:
            raise ValueError("Tier is higher than the allowed max")

        self._linear_reward_growth = linear_reward_growth
        self._tier_base_reward = base_reward
        self._growth_factor = growth_factor

        super().__init__(
            self._calculate_reward(tier),
            cool_down,
            OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, tier),
        )

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, tier: int) -> float:
        """
        Calculate the reward based on the tier base and growth setting.

        :param multiplier: The factor by which the base reward is scaled.
        """

        if self._linear_reward_growth or tier == 1:
            return self._tier_base_reward * tier
        else:
            return round(self._tier_base_reward * (tier**self._growth_factor))
