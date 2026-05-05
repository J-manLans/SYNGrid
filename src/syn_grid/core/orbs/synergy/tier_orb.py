from syn_grid.config.models import TierConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    OrbMeta,
    OrbCategory,
    SynergyType,
)
from typing import Final


class TierOrb(BaseOrb):
    """
    An orb that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 orb a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    max_tier: int

    def __init__(self, tier: int, conf: TierConf):
        if tier > self.max_tier:
            raise ValueError("Tier is higher than the allowed max")

        self._linear_reward_growth = conf.linear_reward_growth
        self._tier_base_reward = conf.base_reward
        self._growth_factor = conf.growth_factor
        self.step_wise_scoring = conf.step_wise_scoring
        self.threshold_scoring = conf.threshold_scoring
        self.max_tier_scoring = conf.max_tier_scoring

        super().__init__(
            self._calculate_reward(tier),
            conf.cool_down,
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
