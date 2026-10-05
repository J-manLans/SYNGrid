"""Digester for tier orbs.

The three scoring modes are ported unchanged from the old DigestionEngine. What
changed is who owns them: the scoring mode is now a setting of this digester,
read from the tier config, instead of something every TierOrb carried around
for the engine to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from syn_grid.config.models.tier_chain_models import ScoringMode
from syn_grid.core.digestion.new_digestion import DigestionResult, Event, OrbKind
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbCategory, SynergyType

# ================= #
#      Events       #
# ================= #


@dataclass(frozen=True)
class ChainProgressed(Event):
    pass


@dataclass(frozen=True)
class ChainBroken(Event):
    pass


@dataclass(frozen=True)
class ChainCompleted(Event):
    pass


# ================= #
#     Digester      #
# ================= #


class TierDigester:
    """Scores tier orbs: how a chain progresses, breaks and completes."""

    kind: OrbKind = (OrbCategory.SYNERGY, SynergyType.TIER)

    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        *,
        scoring: ScoringMode,
        max_tier: int,
        tier_consumption_penalty: float,
        reward_multiplier: float,
        chain_break_penalty: float,
    ):
        self._max_tier = max_tier
        self._tier_consumption_penalty = tier_consumption_penalty
        self._reward_multiplier = reward_multiplier
        self._chain_break_penalty = chain_break_penalty

        scorers = {
            ScoringMode.STEP_WISE: self._step_wise_scoring,
            ScoringMode.THRESHOLD: self._threshold_scoring,
            ScoringMode.MAX_TIER: self._max_tier_scoring,
        }
        if scoring not in scorers:
            raise ValueError(f"The scoring mode {scoring} isn't implemented")
        self._score = scorers[scoring]

        self.reset()

    def reset(self) -> None:
        self._events: list[Event] = []
        self._reset_chain()

    def _reset_chain(self) -> None:
        self._chained_tiers = self._NO_CHAIN
        self._pending_reward = 0.0
        self._max_reward_bonus = 0.0

    # ================= #
    #        API        #
    # ================= #

    @property
    def chained_tiers(self) -> int:
        """The tier the current chain has reached, 0 if there is no chain."""
        return self._chained_tiers

    @property
    def pending_reward(self) -> float:
        """Reward accumulated on the current chain but not yet paid out."""
        return self._pending_reward

    def digest(self, orb: BaseOrb) -> DigestionResult:
        self._events = []
        reward = self._score(orb)
        return DigestionResult(reward, tuple(self._events))

    def notice(self, orb: BaseOrb) -> tuple[Event, ...]:
        """Any other kind of orb being consumed breaks a running chain."""

        if self._chained_tiers == self._NO_CHAIN:
            return ()

        self._events = []
        self._mark_broken()
        self._reset_chain()
        return tuple(self._events)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Events === #

    def _mark_progressed(self) -> None:
        self._events.append(ChainProgressed())

    def _mark_broken(self) -> None:
        self._events.append(ChainBroken())

    def _mark_completed(self) -> None:
        self._chained_tiers = self._NO_CHAIN
        self._events.append(ChainCompleted())

    # === Scoring modes === #

    def _step_wise_scoring(self, orb: BaseOrb) -> float:
        current_tier = orb.META.TIER

        if self._chained_tiers == current_tier - 1:
            if current_tier == self._max_tier:
                self._mark_completed()
                return orb.REWARD + self._flush_rewards()[1]
            else:
                self._mark_progressed()
                self._chained_tiers = current_tier
                self._max_reward_bonus += orb.REWARD

            return orb.REWARD

        self._mark_broken()
        if current_tier != 1:
            self._chained_tiers = self._NO_CHAIN
            # small punishment for consuming in wrong order
            return self._tier_consumption_penalty
        else:
            self._chained_tiers = current_tier
            # as tier 1 is base for the tier chain it never gives any penalty
            return orb.REWARD

    def _threshold_scoring(self, orb: BaseOrb) -> float:
        scaled_reward = round(orb.REWARD * self._reward_multiplier)
        current_tier = orb.META.TIER

        if self._chained_tiers == current_tier - 1:
            # Keep on building the chain, pending reward and bonus if not at max tier
            if current_tier != self._max_tier:
                self._mark_progressed()
                self._chained_tiers = current_tier
                self._set_pending_rewards(scaled_reward)
                return 0.0

            # Reached max tier: reset the chain and return the bonus
            self._mark_completed()
            return self._flush_rewards()[1] + scaled_reward

        return self._handle_chain_break(current_tier, scaled_reward)

    def _max_tier_scoring(self, orb: BaseOrb) -> float:
        current_tier = orb.META.TIER

        if self._chained_tiers == current_tier - 1:
            if current_tier != self._max_tier:
                self._mark_progressed()
                self._chained_tiers = current_tier
                return 0.0

            # Reward only for a completed chain
            self._mark_completed()
            return orb.REWARD

        self._mark_broken()
        self._chained_tiers = (
            self._NO_CHAIN if current_tier != self._BASE_TIER else current_tier
        )

        return self._chain_break_penalty

    # === Scoring helpers === #

    def _handle_chain_break(self, current_tier: int, scaled_reward: float) -> float:
        """
        Handles a broken tier chain and returns any accumulated pending reward.

        - No pending reward: return early with 0.
        - Base tier, previous was higher tier: flush pending reward, restart chain and
        rewards at base tier, return the flushed reward.
        - Base tier, previous was also base tier: consecutive base tier collections are
        not treated as chain breaks, do nothing, return 0.
        - Any other tier: reset chain state, flush and return pending reward.
        """

        if self._pending_reward == 0.0:
            self._mark_broken()
            return 0.0

        pending_reward = 0.0

        if current_tier == self._BASE_TIER:
            if self._chained_tiers != current_tier:
                self._chained_tiers = current_tier
                pending_reward = self._flush_rewards()[0]
                self._set_pending_rewards(scaled_reward)
            else:
                return pending_reward
        else:
            self._chained_tiers = self._NO_CHAIN
            pending_reward = self._flush_rewards()[0]

        self._mark_broken()

        return pending_reward

    def _set_pending_rewards(self, scaled_reward: float) -> None:
        self._pending_reward = scaled_reward
        self._max_reward_bonus += self._pending_reward

    def _flush_rewards(self) -> tuple[float, float]:
        temp_rew = self._pending_reward
        self._pending_reward = 0.0

        temp_bonus = self._max_reward_bonus
        self._max_reward_bonus = 0.0

        return temp_rew, temp_bonus
