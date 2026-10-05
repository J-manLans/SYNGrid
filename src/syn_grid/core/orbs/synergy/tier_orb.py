from syn_grid.config.models.tier_chain import TierOrbConf
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
    To get reward for a tier 3 orb a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, tier: int, conf: TierOrbConf, max_tier: int):
        """``max_tier`` is passed in rather than read off the class.

        It used to be a class attribute written by OrbFactory before every pool
        was built, which meant constructing a TierOrb was only legal immediately
        after constructing a factory -- so the code that decides *which orbs a
        scenario has* could not be exercised without also standing up the thing
        that calls it. Two worlds in one process would also overwrite each
        other's ceiling, which is the same shared-state hazard as the orb
        lifespan.
        """

        if tier > max_tier:
            raise ValueError("Tier is higher than the allowed max")

        # Kept as an instance attribute because the digestion engine compares a
        # consumed orb's tier against the world's ceiling to decide whether the
        # chain is complete.
        self.max_tier = max_tier

        self._linear_reward_growth = conf.linear_reward_growth
        self._tier_base_reward = conf.base_reward
        self._growth_factor = conf.growth_factor
        self.scoring = conf.scoring

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
