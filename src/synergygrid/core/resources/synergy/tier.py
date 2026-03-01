from synergygrid.core.resources import (
    BaseTierResource,
    ResourceMeta,
    ResourceCategory,
    SynergyType,
)


class TierResource(BaseTierResource):
    """
    A resource that needs to be collected in tier order to give a reward.

    Example:
    To get reward for a tier 3 resource a tier 0, tier 1 and tier 2 must have first been collected on that order without breaking the chain.
    """

    _linear_reward_growth: bool = False
    _step_wise_scoring_type: bool = True

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, tier, world_boundaries: tuple[int, int], cool_down: int = 10):
        self._REWARD = self._calculate_reward(tier)

        super().__init__(
            world_boundaries,
            cool_down,
            ResourceMeta(
                category=ResourceCategory.SYNERGY, type=SynergyType.TIER, tier=tier
            ),
        )

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> int:
        super()._consume()

        # Two types of scoring:
        # Reward at every correct collected tier resource
        if self._step_wise_scoring_type:
            if super()._resolve_tier_progression():
                return self._REWARD
            return 0

        # Or reward only when the current chain is broken
        if super()._resolve_tier_progression():
            return 0
        return self._REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    def _calculate_reward(self, multiplier: int) -> int:
        reward = 0

        if self._linear_reward_growth:
            reward = self._TIER_BASE_REWARD * multiplier
        # TODO: just for testing if incentive structure for the agent changes if we power up the
        # reward, remove if not necessary
        else:
            reward = int((self._TIER_BASE_REWARD * (1.5**multiplier)) + 0.5)

        return reward
