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

            # Threshold scoring: accumulate reward silently on correct progression.
            # If the chain breaks or max tier is reached, flush the pending reward and return it.
            if consumed_orb.threshold_scoring:
                return self._threshold_scoring(consumed_orb)

            # Max tier scoring: only give rewards when reaching max tier, for more controlled
            # scenarios
            if consumed_orb.max_tier_scoring:
                return self._max_tier_scoring(consumed_orb)

        # Non-tier orbs: always return base reward and resets reward state
        self.reset()
        return consumed_orb.REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    # === Scoring types === #

    def _step_wise_scoring(self, consumed_orb: TierOrb) -> float:
        if self.chained_tiers == consumed_orb.META.TIER - 1:
            self.chained_tiers = (
                consumed_orb.META.TIER if not consumed_orb.max_tier else self._NO_CHAIN
            )
            return consumed_orb.REWARD

        self.chained_tiers = self._NO_CHAIN
        # small punishment for consuming in wrong order
        return self._tier_consumption_penalty

    def _threshold_scoring(self, consumed_orb: TierOrb) -> float:
        scaled_reward = round(consumed_orb.REWARD * self._reward_multiplier)
        current_tier = consumed_orb.META.TIER

        if self.chained_tiers == current_tier - 1:
            # Keep on building the tier chain, pending reward and bonus if we're not at max tier
            if current_tier != consumed_orb.max_tier:
                self.chained_tiers = current_tier
                self._set_pending_rewards(scaled_reward)
                return 0.0

            # If we reached max tier, reset the chain and return the bonus
            self.chained_tiers = self._NO_CHAIN
            return self._flush_rewards()[1] + scaled_reward

        # Handel pending reward if chain is broken,
        return self._handle_chain_break(current_tier, scaled_reward)

    def _max_tier_scoring(self, consumed_orb: TierOrb):
        current_tier = consumed_orb.META.TIER

        if self.chained_tiers == current_tier - 1:
            if current_tier != consumed_orb.max_tier:
                self.chained_tiers = current_tier
                return 0.0

            self.chained_tiers = self._NO_CHAIN
            return consumed_orb.REWARD

        self.chained_tiers = (
            self._NO_CHAIN if current_tier != self._BASE_TIER else current_tier
        )

        return 0.0

    # === Scoring helpers === #

    def _set_pending_rewards(self, scaled_reward: float):
        self._pending_reward = scaled_reward
        self._max_reward_bonus += self._pending_reward

    def _handle_chain_break(self, current_tier: int, scaled_reward: float) -> float:
        """
        Handles a broken tier chain and returns any accumulated pending reward.

        A chain break occurs when an orb is consumed out of sequence. Behavior depends
        on the breaking orb's tier:

        - No pending reward: return early with 0.
        - Base tier, previous was higher tier: flush pending reward, restart chain and
        rewards at base tier, return the flushed reward.
        - Base tier, previous was also base tier: consecutive base tier collections are
        not treated as chain breaks — do nothing, return 0.
        - Any other tier: reset chain state, flush and return pending reward.
        """

        if self._pending_reward == 0.0:
            return 0.0

        pending_reward = 0.0

        if current_tier == self._BASE_TIER:
            if self.chained_tiers != current_tier:
                self.chained_tiers = current_tier
                pending_reward = self._flush_rewards()[0]
                self._set_pending_rewards(scaled_reward)
        else:
            self.chained_tiers = self._NO_CHAIN
            pending_reward = self._flush_rewards()[0]

        return pending_reward

    def _flush_rewards(self) -> tuple[float, float]:
        temp_rew = self._pending_reward
        self._pending_reward = 0.0

        temp_bonus = self._max_reward_bonus
        self._max_reward_bonus = 0.0

        return temp_rew, temp_bonus
