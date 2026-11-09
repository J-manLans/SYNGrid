"""
How a scenario says which tier digester its worlds get.

A scenario is built once and makes many worlds, and a digester holds episode
state, so a scenario holds a way to make a digester and never a digester. This
is that way for tier orbs, shared by every scenario family that has them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)

# ################## #
#  Digester Recipe   #
# ################## #


@dataclass(frozen=True)
class TierDigesterRecipe:
    """Makes a new tier digester each time it is called.

    It is what a tier orb's `OrbBundle` holds as `make_digester`. Frozen so
    that two recipes with the same scoring mode and values compare equal.

    Attributes:
        scoring_mode: the tier digester class to create, one of the scoring
            modes. It decides when a tier orb's worth is paid.
        max_tier: the highest tier, where a chain is complete.
        chain_break_penalty: what eating a tier orb out of order costs.
    """

    scoring_mode: Callable[[int, float], TierDigester]
    max_tier: int
    chain_break_penalty: float

    def __call__(self) -> TierDigester:
        return self.scoring_mode(self.max_tier, self.chain_break_penalty)
