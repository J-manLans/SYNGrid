"""
How a fixed field gets its tier orbs.

A tier chain has exactly one orb per tier and no weighting anywhere. This is
the implementation of ``OrbPopulation`` for it, shared by every scenario family
whose field holds one chain.
"""

from __future__ import annotations

from dataclasses import dataclass

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.synergy.chain.tier_orb import TierOrb

# ################### #
# Population Strategy #
# ################### #


@dataclass(frozen=True)
class TierChainPopulation:
    """One orb per tier, tiers 1...max_tier.

    A tier chain is a fixed sequence, so the population is the sequence: one
    orb for each tier and no weighting. `max_tier` is the number of orbs, so
    how many there are is never configured. Frozen so two scenarios built from
    the same config hold equal populations.

    Attributes:
        max_tier: the highest tier, and the number of orbs.
        base_reward: what a tier 1 orb is worth.
        growth_factor: how fast the worth grows with the tier. A tier is worth
            `base_reward * tier ** growth_factor`.
        cool_down: the number of steps an orb is off the field before it can
            appear again, or None for orbs that do not return in the episode.
        life_span: the number of steps an orb stays on the field before it
            leaves, or None for orbs that stay until they are eaten.
    """

    max_tier: int
    base_reward: float
    growth_factor: float
    cool_down: int | None
    life_span: int | None

    @property
    def count(self) -> int:
        """The number of orbs, `max_tier`: one for each tier."""

        return self.max_tier

    def create(self) -> list[BaseOrb]:
        """Return one tier orb for each tier, in order from 1 to `max_tier`.

        Each is worth `base_reward * tier ** growth_factor` and is handed this
        population's cool-down and lifespan. They come back off the field, and
        the world places them.
        """

        return [
            TierOrb(
                tier,
                self.max_tier,
                self.base_reward,
                self.growth_factor,
                self.cool_down,
                self.life_span,
            )
            for tier in range(1, self.max_tier + 1)
        ]
