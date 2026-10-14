"""
The tier digester's base: the chain a droid builds by eating tier orbs in order.

The base tracks the chain, counts what happens to it and adds up what its orbs
are worth. What each of those moments pays is a scoring mode, and each mode is
a subclass in this folder.
"""

from abc import ABC, abstractmethod
from typing import Final

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbKind, SynergyKind

# ################## #
# Digestion Strategy #
# ################## #


class TierDigester(ABC):
    """The rules for tier orbs, without the payouts.

    A chain starts at tier 1 and grows one tier at a time. An eaten tier orb
    does one of three things to it:

    - progress: it is the next tier, and not the last one.
    - completion: it is the next tier, and the last one. The chain is over.
    - break: it is any other tier. A tier 1 orb starts a new chain, and any
      other tier leaves no chain.

    A subclass decides what each of the three pays, and what a running chain
    is owed when an episode ends with nothing eaten to end it.

    Attributes:
        KIND: the orb kind this digester owns.
        chained_tiers: the tier the running chain has reached, 0 if there is no
            chain.
        chain_reward: the summed reward of the orbs in the running chain. A
            break or a completion clears it, and a tier 1 orb that starts a new
            chain starts a new sum.
        chains_progressed: how many times a chain has grown this episode.
        chains_broken: how many times a tier orb was eaten out of order this
            episode.
        chains_completed: how many chains have reached the last tier this
            episode.
    """

    _NO_CHAIN: Final[int] = 0
    _BASE_TIER: Final[int] = 1

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, max_tier: int):
        self.KIND: Final[OrbKind] = SynergyKind.TIER
        self._MAX_TIER: Final[int] = max_tier

        self.reset()

    def reset(self) -> None:
        """Clear the chain and the counters, at the start of an episode."""

        self._clear_chain()
        self.chains_progressed = 0
        self.chains_broken = 0
        self.chains_completed = 0

    # ================= #
    #        API        #
    # ================= #

    def digest(self, orb: BaseOrb) -> float:
        """Move the chain for an eaten tier orb and return what that pays."""

        tier = orb.META.TIER

        if tier is None:
            raise ValueError("A tier digester can only digest an orb with a tier")

        if tier != self.chained_tiers + 1:
            reward = self._score_break(orb)
            self.chains_broken += 1
            self._clear_chain()

            if tier == self._BASE_TIER:
                self._grow_chain(tier, orb.REWARD)
        elif tier == self._MAX_TIER:
            reward = self._score_completion(orb)
            self.chains_completed += 1
            self._clear_chain()
        else:
            reward = self._score_progress(orb)
            self.chains_progressed += 1
            self._grow_chain(tier, orb.REWARD)

        return reward

    @abstractmethod
    def owed_at_episode_end(self) -> float:
        """Return what the running chain is owed if the episode ends now.

        It is what the chain has gathered and digestion has not paid. The
        digester only states it: the droid settles it when the episode ends.
        """

    # ================= #
    #      Helpers      #
    # ================= #

    def _clear_chain(self) -> None:
        """Leave no chain, and nothing summed."""

        self.chained_tiers = self._NO_CHAIN
        self.chain_reward = 0.0

    def _grow_chain(self, tier: int, reward: float) -> None:
        """Move the chain to this tier and add the orb's reward to the sum."""

        self.chained_tiers = tier
        self.chain_reward += reward

    # ================= #
    #      Scoring      #
    # ================= #

    # Each is called before the chain moves, so `chained_tiers` and
    # `chain_reward` are still what they were when the orb was eaten.

    @abstractmethod
    def _score_progress(self, orb: BaseOrb) -> float:
        """Return the reward for an orb that grew the chain."""

    @abstractmethod
    def _score_break(self, orb: BaseOrb) -> float:
        """Return the reward for an orb eaten out of order."""

    @abstractmethod
    def _score_completion(self, orb: BaseOrb) -> float:
        """Return the reward for the orb that completed the chain."""
