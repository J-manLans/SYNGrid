"""
How a tier chain fills the orb pool.

A tier chain has exactly one orb per tier and no weighting anywhere. This is
the tier-chain implementation of ``OrbPopulation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syn_grid.core.orbs.base_orb import BaseOrb


# ################### #
# Population Strategy #
# ################### #


@dataclass(frozen=True)
class TierChainPopulation:
    """One orb per tier, tiers 1...max_tier.

    A tier chain is a fixed sequence, so the pool is the sequence. `max_tier`
    is the pool size, which is why a tier chain spawns `max_tier` orbs at reset
    rather than a configured count. Frozen so it compares by value.
    """

    max_tier: int

    def create(self) -> list[BaseOrb]:
        return []
