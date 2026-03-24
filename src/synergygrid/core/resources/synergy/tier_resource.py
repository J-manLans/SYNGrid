from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.resource_meta import (
    ResourceMeta,
    ResourceCategory,
    SynergyType,
)
from typing import Final


class TierResource(BaseResource):
    """
    A resource that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 resource a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    _linear_reward_growth: bool = True
    _TIER_BASE_REWARD: Final[int] = 3
    MAX_TIER: int
    step_wise_scoring_type: bool = False
    _GROWTH_FACTOR: Final[float] = 1.5

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, tier: int, cool_down: int = 10):
        if tier > self.MAX_TIER:
            raise ValueError("Tier is higher than the allowed max")

        super().__init__(
            self._calculate_reward(tier + 1),
            cool_down,
            ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER, tier),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> "TierResource":
        super()._consume()
        return self

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, multiplier: int) -> int:
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
