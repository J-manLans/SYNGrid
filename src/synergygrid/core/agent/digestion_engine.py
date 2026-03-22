from synergygrid.core.resources.base_resource import BaseResource
from synergygrid.core.resources.synergy.tier_resource import TierResource


class DigestionEngine:
    _NO_CHAIN = -1
    _BASE_TIER = 0
    _pending_reward: int
    chained_tiers: int = _NO_CHAIN

    # ================= #
    #        API        #
    # ================= #

    def digest(self, consumed_resource: BaseResource) -> int:
        """
        Process a consumed resource and return the resulting reward.

        - Tiered resources follow progression rules:
            * Step-wise scoring: reward is given only on correct progression.
            * Non-step-wise scoring: reward is given only on incorrect progression.
        - Non-tiered resources always return their base reward.

        :param consumed_resource: The resource being processed.
        :return: The calculated reward.
        """

        # Handle tier-based resources with progression logic
        if isinstance(consumed_resource, TierResource):

            # Step-wise scoring: reward only if progression is correct
            if consumed_resource.step_wise_scoring_type:
                if self._resolve_tier_progression(consumed_resource):
                    return consumed_resource.REWARD
                return 0

            # Non-step-wise scoring: accumulate reward silently on correct progression, and only
            # pay it out when the chain fails
            if self._resolve_tier_progression(consumed_resource):
                self._pending_reward = consumed_resource.REWARD
                return 0
            return self._flush_pending_reward()

        # Non-tier resources: always return base reward
        return consumed_resource.REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    def _resolve_tier_progression(self, consumed_resource: TierResource) -> int:
        current_tier = consumed_resource.meta.tier

        # Correct progression (starting tier or previous tier + 1)
        if self.chained_tiers == current_tier - 1:
            if current_tier == consumed_resource.MAX_TIER:
                # If max tier is reached, reset chain
                self.chained_tiers = self._NO_CHAIN
            else:
                # Otherwise, continue the chain
                self.chained_tiers = current_tier
            return True

        if current_tier == self._BASE_TIER:
            # Restart chain from base tier
            self.chained_tiers = self._BASE_TIER
            return True

        # Invalid progression → reset chain
        self.chained_tiers = self._NO_CHAIN
        return False

    def _flush_pending_reward(self) -> int:
        temp_rew = self._pending_reward
        self._pending_reward = 0
        return temp_rew
