from typing import Final

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.tier_orb import TierOrb


class DigestionEngine:
    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    # ================= #
    #        Init       #
    # ================= #

    def __init__(
        self,
        tier_consumption_penalty: float,
        reward_multiplier: float,
        chain_break_penalty: float,
    ):
        self._tier_consumption_penalty = tier_consumption_penalty
        self._reward_multiplier = reward_multiplier
        self._chain_break_penalty = chain_break_penalty

    def reset(self):
        """Start of episode: clear the chain and zero the episode counters."""

        self._reset_chain()
        self._chains_progressed = 0
        self._chains_broken = 0
        self._chains_completed = 0

    def _reset_chain(self):
        """Clear chain state only. Counters cover the whole episode and stay."""

        self.chained_tiers = self._NO_CHAIN
        self._pending_reward = 0.0
        self._max_reward_bonus = 0.0

    # ================= #
    #        API        #
    # ================= #

    @property
    def stats(self) -> dict[str, int]:
        """Episode counters as a fresh dict, safe to merge into the env's info."""

        return {
            "chains_progressed": self._chains_progressed,
            "chains_broken": self._chains_broken,
            "chains_completed": self._chains_completed,
        }

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

            raise ValueError("The scoring type for this orb isn't implemented")

        # Non-tier orbs: always return base reward. If a chain was running it is
        # broken, and any pending reward and bonus disappear with it.
        if self.chained_tiers != self._NO_CHAIN:
            self._mark_broken()
            self._reset_chain()

        return consumed_orb.REWARD

    # ================= #
    #      Helpers      #
    # ================= #

    # === Counters === #

    def _mark_progressed(self):
        self._chains_progressed += 1

    def _mark_broken(self):
        self._chains_broken += 1

    def _mark_completed(self):
        self.chained_tiers = self._NO_CHAIN
        self._chains_completed += 1

    # === Scoring types === #

    def _step_wise_scoring(self, consumed_orb: TierOrb) -> float:
        current_tier = consumed_orb.META.TIER

        if self.chained_tiers == current_tier - 1:
            if current_tier == consumed_orb.max_tier:
                self._mark_completed()
                return consumed_orb.REWARD + self._flush_rewards()[1]
            else:
                self._mark_progressed()
                self.chained_tiers = current_tier
                self._max_reward_bonus += consumed_orb.REWARD

            return consumed_orb.REWARD

        self._mark_broken()
        if current_tier != 1:
            self.chained_tiers = self._NO_CHAIN
            # small punishment for consuming in wrong order
            return self._tier_consumption_penalty
        else:
            self.chained_tiers = current_tier
            # as tier 1 is base for the tier chain it never gives any penalty
            return consumed_orb.REWARD

    def _threshold_scoring(self, consumed_orb: TierOrb) -> float:
        scaled_reward = round(consumed_orb.REWARD * self._reward_multiplier)
        current_tier = consumed_orb.META.TIER

        if self.chained_tiers == current_tier - 1:
            # Keep on building the tier chain, pending reward and bonus if we're not at max tier
            if current_tier != consumed_orb.max_tier:
                self._mark_progressed()
                self.chained_tiers = current_tier
                self._set_pending_rewards(scaled_reward)
                return 0.0

            # If we reached max tier, reset the chain and return the bonus
            self._mark_completed()
            return self._flush_rewards()[1] + scaled_reward

        # Handel pending reward if chain is broken,
        return self._handle_chain_break(current_tier, scaled_reward)

    def _max_tier_scoring(self, consumed_orb: TierOrb):
        current_tier = consumed_orb.META.TIER

        if self.chained_tiers == current_tier - 1:
            if current_tier != consumed_orb.max_tier:
                # Build on current chain
                self._mark_progressed()
                self.chained_tiers = current_tier
                return 0.0

            # Return reward for completed chain
            self._mark_completed()
            return consumed_orb.REWARD

        self._mark_broken()
        self.chained_tiers = (
            self._NO_CHAIN if current_tier != self._BASE_TIER else current_tier
        )

        return self._chain_break_penalty

    # === Scoring helpers === #

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
            self._mark_broken()
            return 0.0

        pending_reward = 0.0

        if current_tier == self._BASE_TIER:
            if self.chained_tiers != current_tier:
                self.chained_tiers = current_tier
                pending_reward = self._flush_rewards()[0]
                self._set_pending_rewards(scaled_reward)
            else:
                return pending_reward
        else:
            self.chained_tiers = self._NO_CHAIN
            pending_reward = self._flush_rewards()[0]

        self._mark_broken()

        return pending_reward

    def _set_pending_rewards(self, scaled_reward: float):
        self._pending_reward = scaled_reward
        self._max_reward_bonus += self._pending_reward

    def _flush_rewards(self) -> tuple[float, float]:
        temp_rew = self._pending_reward
        self._pending_reward = 0.0

        temp_bonus = self._max_reward_bonus
        self._max_reward_bonus = 0.0

        return temp_rew, temp_bonus
