"""
When an episode ends, and what the last step is worth.

A termination rule belongs to an objective, not to a kind of scenario. Tier
chain and continuous are the two that exist. Another goal scenario brings its
own rule behind ``TerminationRules`` rather than extending one of these: there
is no shared base class, because the two have too little in common to earn one.

A rule decides when the episode ends and sets the one reward the clock causes.
A reward caused by consuming an orb comes from digestion and is passed through
untouched.

``truncated`` is always False. Running out of steps is a punishment and sets
``terminated``, so no value function is bootstrapped at the horizon. That is a
deliberate decision with a real cost, pinned by a test so it is not later
"fixed" by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from syn_grid.core.digestion.tier_digester import (
    ChainBroken,
    ChainCompleted,
    TierOrbDigester,
)

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# ################## #
#     Interface      #
# ################## #


@dataclass(frozen=True)
class EpisodeOutcome:
    terminated: bool
    truncated: bool
    reward: float


class TerminationRules(Protocol):
    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        ...


# ################## #
#     Strategies     #
# ################## #


@dataclass(frozen=True)
class TierChainTermination:
    """Ends a tier chain episode: a broken chain, a finished chain, or the clock.

    The only reward set here is the timeout. A break and a completion keep the
    reward digestion gave them.

    Attributes:
        timeout_penalty: paid for reaching the step limit unfinished.
    """

    timeout_penalty: float # TODO: think this should be optional

    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        engine = world.droid.digestion_engine

        # The score ending is switched off for tier chains. Thesis: it adds
        # nothing here. A wall hit is already punished when it happens, the
        # score can only fall because the one positive reward ends the episode,
        # and under fog of war the agent cannot see the score that would end it.
        # TODO: once the repo is stable, do one training run without it, then
        # either delete this line or restore it.
        # terminated = world.droid.score <= 0
        terminated = False

        if engine.count(ChainBroken) > 0 or engine.count(ChainCompleted) > 0:
            terminated = True

        elif steps_left <= 0:
            reward = self.timeout_penalty
            terminated = True

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)


@dataclass(frozen=True)
class ContinuousTermination:
    """Ends a continuous episode: only the score running out, or the clock.

    There is no chain to break and nothing to complete, so the clock is the
    only interesting boundary. A timeout still settles a partially accumulated
    reward.
    """

    def evaluate(self, world: GridWorld, steps_left: int, reward: float) -> EpisodeOutcome:
        terminated = world.droid.score <= 0

        if steps_left <= 0:
            reward = world.droid.digestion_engine.get(TierOrbDigester).pending_reward
            terminated = True

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)
