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

    _linear_reward_growth: bool
    _TIER_BASE_REWARD: Final[float]
    _GROWTH_FACTOR: Final[float]
    MAX_TIER: int
    step_wise_scoring: bool

    def __init__(self, tier: int, conf: TierConf):
        if tier > self.MAX_TIER:
            raise ValueError("Tier is higher than the allowed max")

        self._linear_reward_growth = conf.linear_reward_growth
        self._TIER_BASE_REWARD = conf.base_reward
        self._GROWTH_FACTOR = conf.growth_factor
        self.step_wise_scoring = conf.step_wise_scoring

        super().__init__(
            self._calculate_reward(tier),
            conf.cool_down,
            OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, tier),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> "TierOrb":
        super()._consume()
        return self

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, multiplier: int) -> float:
        """
        Calculate the reward based on the tier base and growth setting.

        :param multiplier: The factor by which the base reward is scaled.
        """

        if self._linear_reward_growth:
            reward = self._TIER_BASE_REWARD * multiplier
        else:
            reward = int(
                (self._TIER_BASE_REWARD * (self._GROWTH_FACTOR**multiplier)) + 0.5
            )

        return reward
