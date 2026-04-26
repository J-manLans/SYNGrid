from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb

from typing import Final


class DigestionEngine:
    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    # ================= #
    #        Init       #
    # ================= #

    def reset(self):
        self.chained_tiers: int = self._NO_CHAIN
        self._pending_reward: float = 0.0

    # ================= #
    #        API        #
    # ================= #

    def digest(self, consumed_orb: BaseOrb, tier_consumption_penalty: float) -> float:
        """
        Process a consumed orb and return the resulting reward.

        - Tiered orbs follow progression rules:
            * Step-wise scoring: reward is given only on correct progression.
            * Non-step-wise scoring: reward is given only on incorrect progression.
        - Non-tiered orbs always return their base reward.

        :param consumed_orb: The orb being processed.
        :return: The calculated reward.
        """

        # Handle tier-based orbs with progression logic
        if isinstance(consumed_orb, TierOrb):
            # Step-wise scoring: reward only if progression is correct
            if consumed_orb.STEP_WISE_SCORING:
                if self._resolve_tier_progression(consumed_orb):
                    return consumed_orb.REWARD
                return tier_consumption_penalty  # small punishment for consuming in wrong order

            # Non-step-wise scoring: accumulate reward silently on correct progression.
            # If the chain breaks, flush the pending reward and return it.
            if self._resolve_tier_progression(consumed_orb):
                # If the consumed orb is of base tier but other orbs have been consumed before it,
                # then flush the pending reward, set pending reward to the base tiers reward and
                # return the flushed reward
                if (
                    consumed_orb.META.TIER == self._BASE_TIER
                    and self._pending_reward is not 0
                ):
                    reward = self._flush_pending_reward()
                    self._pending_reward = consumed_orb.REWARD
                    return reward

                # If the consumed orb is of max tier, add its reward to the pending one, then flush
                # it and return it
                if consumed_orb.META.TIER == consumed_orb.MAX_TIER:
                    self._pending_reward = consumed_orb.REWARD
                    return self._flush_pending_reward()

                # Otherwise, keep on building the pending reward
                self._pending_reward = consumed_orb.REWARD
                return 0

            # If chain is broken just flush the pending reward and return it
            return self._flush_pending_reward()

        # Non-tier orbs: always return base reward and resets the tier chain
        self.chained_tiers = self._NO_CHAIN
        return consumed_orb.REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    def _resolve_tier_progression(self, consumed_orb: TierOrb) -> bool:
        current_tier = consumed_orb.META.TIER

        # Correct progression (starting tier or previous tier + 1)
        if self.chained_tiers == current_tier - 1:
            if current_tier == consumed_orb.MAX_TIER:
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

        # Invalid progression - reset chain
        self.chained_tiers = self._NO_CHAIN
        return False

    def _flush_pending_reward(self) -> float:
        if self._pending_reward == 0.0:
            return 0.0

        temp_rew = self._pending_reward
        self._pending_reward = 0.0
        return temp_rew
