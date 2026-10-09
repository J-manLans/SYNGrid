"""Digester for tier orbs.

The two scoring modes a tier chain uses are ported from the old DigestionEngine.
Step-wise scoring belongs to the continuous family and returns with it. What
changed is who owns them: the scoring mode is now a setting of this digester,
read from the tier config, instead of something every TierOrb carried around
for the engine to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from syn_grid.core.digestion.digestion import DigestionResult, Event, OrbKind
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbCategory, SynergyType


# #################### #
#    Scoring modes     #
# #################### #


class ScoringMode(Enum):
    """How a tier chain's reward is paid out. Chosen by the scenario's builder."""

    THRESHOLD = "threshold"
    MAX_TIER = "max_tier"


# #################### #
#        Events        #
# #################### #


@dataclass(frozen=True)
class ChainProgressed(Event):
    pass


@dataclass(frozen=True)
class ChainBroken(Event):
    pass


@dataclass(frozen=True)
class ChainCompleted(Event):
    pass


# #################### #
# OrbDigester Strategy #
# #################### #


class TierOrbDigester:
    """Scores tier orbs: how a chain progresses, breaks and completes."""

    # ================= #
    #       Init        #
    # ================= #

    kind: OrbKind = (OrbCategory.SYNERGY, SynergyType.TIER)

    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    def __init__(
        self,
        scoring_mode: ScoringMode,
        max_tier: int,
        completion_reward: float,
        chain_break_penalty: float,
    ):
        self._max_tier = max_tier
        self._completion_reward = completion_reward
        self._chain_break_penalty = chain_break_penalty

        # Pick scoring method for the digest method
        scoring_methods = {
            ScoringMode.THRESHOLD: self._threshold_scoring,
            ScoringMode.MAX_TIER: self._max_tier_scoring,
        }

        if scoring_mode not in scoring_methods:
            raise ValueError(f"The scoring mode {scoring_mode} isn't implemented")

        self._scoring_method = scoring_methods[scoring_mode]

        self.reset()

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

    def reset(self) -> None:
        self._events: list[Event] = []
        self._reset_chain()

    def digest(self, orb: BaseOrb) -> DigestionResult:
        self._events = []
        reward = self._scoring_method(orb)
        return DigestionResult(reward, tuple(self._events))

    def notice(self, orb: BaseOrb) -> tuple[Event, ...]:
        """Only run if other orb exist in the scenario.

        And any other kind of orb being consumed breaks a running chain.
        """

        if self._chained_tiers == self._NO_CHAIN:
            return ()

        self._events = []
        self._mark_broken()
        self._reset_chain()
        return tuple(self._events)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _reset_chain(self) -> None:
        self._chained_tiers = self._NO_CHAIN
        self._pending_reward = 0.0
        self._max_reward_bonus = 0.0

    # === Events === #

    def _mark_progressed(self) -> None:
        self._events.append(ChainProgressed())

    def _mark_broken(self) -> None:
        self._events.append(ChainBroken())

    def _mark_completed(self) -> None:
        self._chained_tiers = self._NO_CHAIN
        self._events.append(ChainCompleted())

    # === Scoring modes === #

    def _threshold_scoring(self, orb: BaseOrb) -> float:
        scaled_reward = round(orb.REWARD)
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
            return self._completion_reward

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
