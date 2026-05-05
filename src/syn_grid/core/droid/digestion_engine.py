from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb

from typing import Final


class DigestionEngine:
    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, tier_consumption_penalty: float, reward_multiplier: float):
        self._tier_consumption_penalty = tier_consumption_penalty
        self._reward_multiplier = reward_multiplier

    def reset(self):
        self.chained_tiers = self._NO_CHAIN
        self._pending_reward = 0.0
        self._max_reward_bonus = 0.0

    # ================= #
    #        API        #
    # ================= #

    def digest(self, consumed_orb: BaseOrb) -> float:
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
            if consumed_orb.step_wise_scoring:
                return self._step_wise_scoring(consumed_orb)

            # Non-step-wise scoring: accumulate reward silently on correct progression.
            # If the chain breaks, flush the pending reward and return it.
            return self._threshold_scoring(consumed_orb)

        # Non-tier orbs: always return base reward and resets the tier chain
        self.chained_tiers = self._NO_CHAIN
        self._pending_reward = 0.0
        return consumed_orb.REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    # === Scoring types === #

    def _step_wise_scoring(self, consumed_orb: TierOrb) -> float:
        if self._resolve_tier_progression(consumed_orb):
            return consumed_orb.REWARD
        # small punishment for consuming in wrong order
        return self._tier_consumption_penalty

    def _threshold_scoring(self, consumed_orb: TierOrb) -> float:
        if self._resolve_tier_progression(consumed_orb):
            # Get the pending reward if the consumed orb is a base tier and other orbs except
            # another base tier have been consumed before it
            if (
                consumed_orb.META.TIER == self._BASE_TIER
                and self._pending_reward != 0
                and self._pending_reward != round(consumed_orb.REWARD * self._reward_multiplier)
            ):
                return self._get_pending_reward(consumed_orb.REWARD)

            # If we reached max tier, return the bonus
            if consumed_orb.META.TIER == consumed_orb.max_tier:
                return self._get_max_tier_bonus(consumed_orb.REWARD)

            # Otherwise, keep on building the pending reward and bonus
            self._set_pending_rewards(consumed_orb.REWARD)
            return 0.0

        # Handel pending reward if chain is broken,
        return self._get_pending_reward_at_chain_break()

    def _sparse_scoring(self, consumed_orb: TierOrb):
        if self._resolve_tier_progression(consumed_orb):
            if consumed_orb.META.TIER == consumed_orb.max_tier:
                _, max_bonus = self._flush_rewards()
                return max_bonus + round(consumed_orb.REWARD * self._reward_multiplier)

            self._set_pending_rewards(consumed_orb.REWARD)
            return 0

        self._flush_rewards()
        return 0

    # === Scoring helpers === #

    def _resolve_tier_progression(self, consumed_orb: TierOrb) -> bool:
        current_tier = consumed_orb.META.TIER

        # Correct progression (starting tier or previous tier + 1)
        if self.chained_tiers == current_tier - 1:
            if current_tier == consumed_orb.max_tier:
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

    def _get_pending_reward(self, orb_reward) -> float:
        '''
        Flush the bonus and pending reward, set pending reward
        to the base tiers reward and return the flushed reward
        '''

        pending_reward, _ = self._flush_rewards()
        self._set_pending_rewards(orb_reward)
        return pending_reward

    def _get_max_tier_bonus(self, orb_reward: float) -> float:
        '''
        Flush the bonus and pending reward and return the
        bonus plus the orb's own reward scaled by the multiplier
        '''

        return self._flush_rewards()[1] + round(orb_reward * self._reward_multiplier)

    def _get_pending_reward_at_chain_break(self) -> float:
        '''
        If we haven't chained anything yet, return early. Otherwise flush the bonus and pending reward, then return the pending reward
        '''

        if self._pending_reward == 0.0:
            return 0.0

        return self._flush_rewards()[0]

    def _set_pending_rewards(self, orb_reward: float):
        self._pending_reward = round(orb_reward * self._reward_multiplier)
        self._max_reward_bonus += self._pending_reward

    def _flush_rewards(self) -> tuple[float, float]:
        temp_rew = self._pending_reward
        self._pending_reward = 0.0

        temp_bonus = self._max_reward_bonus
        self._max_reward_bonus = 0.0

        return temp_rew, temp_bonus
